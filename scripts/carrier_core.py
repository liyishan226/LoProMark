from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from PIL import Image, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    gaussian_blur,
    pil_to_tensor,
    rotate,
    to_pil_image,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.attacks import jpeg_roundtrip, resize_roundtrip
from lopromark.config import PATHS
from lopromark.data import random_rectangle_masks
from lopromark.messages import sample_binary_messages


PROMPTS = [
    "a cinematic photo of a small cabin beside a lake, detailed natural light",
    "a watercolor painting of a city street after rain, soft reflections",
    "a close up studio photo of a ceramic vase with blue flowers",
    "a realistic landscape photo of mountains under dramatic clouds",
    "a cozy library interior with warm lamps and wooden shelves",
    "a detailed digital painting of an old stone bridge over a river",
    "a product photo of a glass perfume bottle on a marble table",
    "a bright food photo of berries and pastries on a white plate",
]


def _row_prompt(row: dict[str, object], column: str) -> str:
    candidates = [column, "prompt", "Prompt", "caption", "Caption", "text", "Text"]
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for value in row.values():
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("Could not find a non-empty prompt field.")


def load_prompts(path: Path | None, column: str = "prompt") -> list[str]:
    if path is None:
        return PROMPTS
    suffix = path.suffix.lower()
    prompts: list[str] = []
    if suffix == ".txt":
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    prompt = item
                elif isinstance(item, dict):
                    prompt = _row_prompt(item, column)
                else:
                    raise ValueError(f"Unsupported JSONL item in {path}: {type(item).__name__}")
                prompts.append(prompt.strip())
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    prompts.append(item.strip())
                elif isinstance(item, dict):
                    prompts.append(_row_prompt(item, column))
                else:
                    raise ValueError(f"Unsupported JSON item in {path}: {type(item).__name__}")
        elif isinstance(data, dict):
            items = data.get("annotations") or data.get("prompts") or data.get("data") or data.get(column)
            if not isinstance(items, list):
                raise ValueError(f"Could not find a prompt list in {path}")
            for item in items:
                if isinstance(item, str):
                    prompts.append(item.strip())
                elif isinstance(item, dict):
                    prompts.append(_row_prompt(item, column))
                else:
                    raise ValueError(f"Unsupported JSON item in {path}: {type(item).__name__}")
        else:
            raise ValueError(f"Unsupported JSON root in {path}: {type(data).__name__}")
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                prompts.append(_row_prompt(row, column))
    elif suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        selected = column
        if selected not in frame.columns:
            for candidate in ["prompt", "Prompt", "caption", "Caption", "text", "Text"]:
                if candidate in frame.columns:
                    selected = candidate
                    break
            else:
                selected = str(frame.columns[0])
        prompts = [str(value).strip() for value in frame[selected].tolist() if str(value).strip()]
    else:
        raise ValueError(f"Unsupported prompt file extension: {path.suffix}")
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def parse_attacks(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def tensor_to_bchw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected a 4D image tensor, got {tuple(images.shape)}")
    if images.shape[1] in {1, 3, 4}:
        return images[:, :3].clamp(0.0, 1.0)
    return images.permute(0, 3, 1, 2)[:, :3].contiguous().clamp(0.0, 1.0)


def image_psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = torch.mean((reference.float() - estimate.float()) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def global_ssim(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference = reference.float()
    estimate = estimate.float()
    c1 = 0.01**2
    c2 = 0.03**2
    dims = (-2, -1)
    mu_x = reference.mean(dim=dims, keepdim=True)
    mu_y = estimate.mean(dim=dims, keepdim=True)
    sigma_x = ((reference - mu_x) ** 2).mean(dim=dims, keepdim=True)
    sigma_y = ((estimate - mu_y) ** 2).mean(dim=dims, keepdim=True)
    sigma_xy = ((reference - mu_x) * (estimate - mu_y)).mean(dim=dims, keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim.mean().detach().cpu())


def apply_image_attack(images: torch.Tensor, name: str) -> torch.Tensor:
    attacked, _ = apply_image_attack_with_meta(images, name)
    return attacked


def apply_image_attack_with_meta(
    images: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, tuple[float, float, float] | None]:
    name = name.lower()
    if name in {"identity", "none", "clean"}:
        return images, (1.0, 0.0, 0.0)
    if name == "jpeg" or name.startswith("jpeg_q"):
        quality = 75
        if name.startswith("jpeg_q"):
            quality = int(name.removeprefix("jpeg_q"))
        return jpeg_roundtrip(images, quality=quality), (1.0, 0.0, 0.0)
    if name == "resize" or name.startswith("resize_s"):
        scale = 0.75
        if name.startswith("resize_s"):
            scale = float(name.removeprefix("resize_s"))
        return resize_roundtrip(images, scale=scale), (1.0, 0.0, 0.0)
    if name == "gaussian_blur" or name.startswith("gaussian_blur_k"):
        kernel = 7
        if name.startswith("gaussian_blur_k"):
            kernel = int(name.removeprefix("gaussian_blur_k"))
        if kernel % 2 == 0:
            kernel += 1
        sigma = max(0.1, kernel / 6.0)
        return gaussian_blur(images.float(), [kernel, kernel], [sigma, sigma]).clamp(0, 1), (1.0, 0.0, 0.0)
    if name == "gaussian_noise" or name.startswith("gaussian_noise_s"):
        sigma = 0.05
        if name.startswith("gaussian_noise_s"):
            sigma = float(name.removeprefix("gaussian_noise_s"))
        return (images.float() + torch.randn_like(images.float()) * sigma).clamp(0, 1), (1.0, 0.0, 0.0)
    if name == "brightness" or name.startswith("brightness_f"):
        factor = 1.5
        if name.startswith("brightness_f"):
            factor = float(name.removeprefix("brightness_f"))
        return adjust_brightness(images.float(), factor).clamp(0, 1), (1.0, 0.0, 0.0)
    if name == "contrast" or name.startswith("contrast_f"):
        factor = 1.5
        if name.startswith("contrast_f"):
            factor = float(name.removeprefix("contrast_f"))
        return adjust_contrast(images.float(), factor).clamp(0, 1), (1.0, 0.0, 0.0)
    if name == "saturation" or name.startswith("saturation_f"):
        factor = 1.5
        if name.startswith("saturation_f"):
            factor = float(name.removeprefix("saturation_f"))
        return adjust_saturation(images.float(), factor).clamp(0, 1), (1.0, 0.0, 0.0)
    if name == "hue" or name.startswith("hue_h"):
        factor = 0.25
        if name.startswith("hue_h"):
            factor = float(name.removeprefix("hue_h"))
        factor = max(-0.5, min(0.5, factor))
        return adjust_hue(images.float(), factor).clamp(0, 1), (1.0, 0.0, 0.0)
    if name == "rotation" or name.startswith("rotation_d"):
        degrees = 10.0
        if name.startswith("rotation_d"):
            degrees = float(name.removeprefix("rotation_d"))
        return (
            rotate(images.float(), degrees, interpolation=InterpolationMode.BILINEAR, fill=[0.0, 0.0, 0.0]).clamp(0, 1),
            None,
        )
    if name == "median" or name.startswith("median_k"):
        kernel = 5
        if name.startswith("median_k"):
            kernel = int(name.removeprefix("median_k"))
        if kernel % 2 == 0:
            kernel += 1
        outputs = []
        for image in images.detach().cpu():
            pil = to_pil_image(image.clamp(0, 1))
            restored = pil.filter(ImageFilter.MedianFilter(size=kernel))
            outputs.append(pil_to_tensor(restored).float() / 255.0)
        return torch.stack(outputs, dim=0).to(images.device), (1.0, 0.0, 0.0)
    if name == "saltpepper" or name.startswith("saltpepper_p"):
        prob = 0.01
        if name.startswith("saltpepper_p"):
            prob = float(name.removeprefix("saltpepper_p"))
        out = images.float().clone()
        rnd = torch.rand(out.shape[0], 1, out.shape[2], out.shape[3], device=out.device)
        out = torch.where(rnd < prob / 2.0, torch.zeros_like(out), out)
        out = torch.where(rnd > 1.0 - prob / 2.0, torch.ones_like(out), out)
        return out.clamp(0, 1), (1.0, 0.0, 0.0)
    if name == "nlmeans" or name.startswith("nlmeans_h"):
        h = 10.0
        if name.startswith("nlmeans_h"):
            h = float(name.removeprefix("nlmeans_h"))
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            raise RuntimeError("cv2/numpy are required for nlmeans attack") from exc
        outputs = []
        for image in images.detach().cpu():
            arr = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
            rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).float() / 255.0
            outputs.append(tensor)
        return torch.stack(outputs, dim=0).to(images.device), (1.0, 0.0, 0.0)
    if name == "center_crop_resize" or name.startswith("center_crop_resize_c"):
        crop_frac = 0.85
        if name.startswith("center_crop_resize_c"):
            crop_frac = float(name.removeprefix("center_crop_resize_c"))
        _, _, height, width = images.shape
        crop_h = int(height * crop_frac)
        crop_w = int(width * crop_frac)
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        crop = images[:, :, top : top + crop_h, left : left + crop_w]
        return (
            torch.nn.functional.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False),
            (crop_frac, 0.5, 0.5),
        )
    if name == "random_crop_resize" or name.startswith("random_crop_resize_c"):
        crop_frac = 0.85
        if name.startswith("random_crop_resize_c"):
            crop_frac = float(name.removeprefix("random_crop_resize_c"))
        _, _, height, width = images.shape
        crop_h = int(height * crop_frac)
        crop_w = int(width * crop_frac)
        max_top = max(0, height - crop_h)
        max_left = max(0, width - crop_w)
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)
        crop = images[:, :, top : top + crop_h, left : left + crop_w]
        top_frac = 0.0 if max_top == 0 else top / max_top
        left_frac = 0.0 if max_left == 0 else left / max_left
        return (
            torch.nn.functional.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False),
            (crop_frac, top_frac, left_frac),
        )
    raise ValueError(f"Unknown attack: {name}")


@torch.no_grad()
def vae_roundtrip_attack(pipe: StableDiffusionPipeline, images: torch.Tensor) -> torch.Tensor:
    latents = encode_image_to_latent(pipe, images.float())
    decoded = pipe.vae.decode(latents / float(pipe.vae.config.scaling_factor)).sample
    return ((decoded.float() / 2.0) + 0.5).clamp(0, 1)


@torch.no_grad()
def diffusion_regen_attack(
    pipe: StableDiffusionPipeline,
    images: torch.Tensor,
    prompt: str,
    steps: int,
    guidance_scale: float,
    strength: float,
) -> torch.Tensor:
    device = pipe.device
    latents = encode_image_to_latent(pipe, images.float())
    do_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        [prompt],
        device,
        1,
        do_guidance,
        negative_prompt=[""],
    )
    if do_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pipe.scheduler.set_timesteps(steps, device=device)
    init_timestep = min(steps, max(1, int(round(steps * strength))))
    t_start = max(steps - init_timestep, 0)
    timesteps = pipe.scheduler.timesteps[t_start:]
    noise = torch.randn_like(latents)
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[:1])
    for timestep in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = pipe.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample
        if do_guidance:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
    decoded = pipe.vae.decode(latents / float(pipe.vae.config.scaling_factor)).sample
    return ((decoded.float() / 2.0) + 0.5).clamp(0, 1)


def crop_resize_bchw(
    tensor: torch.Tensor,
    crop_frac: float,
    top_frac: float,
    left_frac: float,
    mode: str,
) -> torch.Tensor:
    """Crop a BCHW tensor and resize back to the original spatial size."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(tensor.shape)}")
    _, _, height, width = tensor.shape
    crop_h = max(1, min(height, int(round(height * crop_frac))))
    crop_w = max(1, min(width, int(round(width * crop_frac))))
    max_top = max(0, height - crop_h)
    max_left = max(0, width - crop_w)
    top = int(round(max_top * top_frac))
    left = int(round(max_left * left_frac))
    crop = tensor[:, :, top : top + crop_h, left : left + crop_w]
    if mode == "nearest":
        return torch.nn.functional.interpolate(crop, size=(height, width), mode=mode)
    return torch.nn.functional.interpolate(crop, size=(height, width), mode=mode, align_corners=False)


def visible_region_fraction(region_masks: torch.Tensor, candidate: tuple[float, float, float] | None) -> torch.Tensor:
    if candidate is None or candidate[0] >= 0.999:
        return torch.ones(region_masks.shape[:2], device=region_masks.device)
    crop_frac, top_frac, left_frac = candidate
    _, _, height, width = region_masks.shape
    crop_h = max(1, min(height, int(round(height * crop_frac))))
    crop_w = max(1, min(width, int(round(width * crop_frac))))
    max_top = max(0, height - crop_h)
    max_left = max(0, width - crop_w)
    top = int(round(max_top * top_frac))
    left = int(round(max_left * left_frac))
    retained = region_masks[:, :, top : top + crop_h, left : left + crop_w].flatten(2).sum(dim=-1)
    total = region_masks.flatten(2).sum(dim=-1).clamp_min(1e-6)
    return retained / total


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def geometry_candidates(mode: str, fracs: list[float], grid: int) -> list[tuple[float, float, float]]:
    """Return (crop_frac, top_frac, left_frac) candidates in normalized crop coordinates."""
    if mode == "none":
        return [(1.0, 0.0, 0.0)]
    if mode == "center":
        return [(1.0, 0.0, 0.0)] + [(frac, 0.5, 0.5) for frac in fracs if frac < 1.0]
    if mode != "grid":
        raise ValueError(f"Unknown geometry search mode: {mode}")
    if grid <= 1:
        positions = [0.5]
    else:
        positions = [index / (grid - 1) for index in range(grid)]
    candidates = [(1.0, 0.0, 0.0)]
    for frac in fracs:
        if frac >= 1.0:
            continue
        for top_frac in positions:
            for left_frac in positions:
                candidates.append((frac, top_frac, left_frac))
    return candidates


def warp_geometry(
    carriers: torch.Tensor,
    region_masks: torch.Tensor,
    candidate: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    crop_frac, top_frac, left_frac = candidate
    if crop_frac >= 0.999:
        return carriers, region_masks
    warped_carriers = crop_resize_bchw(carriers, crop_frac, top_frac, left_frac, mode="bilinear")
    warped_masks = crop_resize_bchw(region_masks, crop_frac, top_frac, left_frac, mode="nearest")
    return warped_carriers, warped_masks


def make_carriers(bits: int, channels: int, height: int, width: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    carriers = torch.randn(bits, channels, height, width, generator=generator)
    carriers = carriers.to(device)
    return carriers


def make_smooth_carriers(
    bits: int,
    channels: int,
    height: int,
    width: int,
    base_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    low = torch.randn(bits, channels, base_size, base_size, generator=generator)
    carriers = torch.nn.functional.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    carriers = carriers - carriers.flatten(1).mean(dim=1)[:, None, None, None]
    carriers = carriers / carriers.flatten(1).std(dim=1).clamp_min(1e-6)[:, None, None, None]
    return carriers.to(device)


def masked_unit_carriers(carriers: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = carriers * mask[None, None]
    denom = masked.flatten(1).norm(dim=1).clamp_min(1e-6)
    return masked / denom[:, None, None, None]


def normalize_latent_noise(noise: torch.Tensor) -> torch.Tensor:
    mean = noise.flatten(1).mean(dim=1)[:, None, None, None]
    std = noise.flatten(1).std(dim=1).clamp_min(1e-6)[:, None, None, None]
    return (noise - mean) / std


def match_local_region_stats(
    watermarked: torch.Tensor,
    reference: torch.Tensor,
    region_masks: torch.Tensor,
) -> torch.Tensor:
    matched = watermarked.clone()
    for region in range(region_masks.shape[1]):
        mask = region_masks[:, region : region + 1]
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        wm_mean = (matched * mask).sum(dim=(2, 3), keepdim=True) / denom
        ref_mean = (reference * mask).sum(dim=(2, 3), keepdim=True) / denom
        wm_var = (((matched - wm_mean) * mask) ** 2).sum(dim=(2, 3), keepdim=True) / denom
        ref_var = (((reference - ref_mean) * mask) ** 2).sum(dim=(2, 3), keepdim=True) / denom
        wm_std = wm_var.clamp_min(1e-6).sqrt()
        ref_std = ref_var.clamp_min(1e-6).sqrt()
        local = (matched - wm_mean) / wm_std * ref_std + ref_mean
        matched = matched * (1.0 - mask) + local * mask
    return matched


def unit_carriers(carriers: torch.Tensor) -> torch.Tensor:
    denom = carriers.flatten(1).norm(dim=1).clamp_min(1e-6)
    return carriers / denom[:, None, None, None]


def add_local_noise_watermark(
    noise: torch.Tensor,
    region_masks: torch.Tensor,
    messages: torch.Tensor,
    carriers: torch.Tensor,
    alpha: float,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    watermarked = noise.clone()
    bits = messages.shape[-1]
    for region in range(region_masks.shape[1]):
        unit = masked_unit_carriers(carriers, region_masks[0, region])
        signs = messages[0, region].mul(2.0).sub(1.0)
        code = torch.einsum("k,kchw->chw", signs, unit) / math.sqrt(bits)
        watermarked[0] = watermarked[0] + alpha * code
    if normalize:
        return normalize_latent_noise(watermarked)
    return watermarked


def add_global_sync_pilot(noise: torch.Tensor, carriers: torch.Tensor, alpha: float) -> torch.Tensor:
    if carriers.numel() == 0 or alpha <= 0:
        return noise
    pilot = unit_carriers(carriers)
    code = pilot.sum(dim=0) / math.sqrt(carriers.shape[0])
    return noise + alpha * code[None]


def decode_local_noise_watermark(
    recovered_noise: torch.Tensor,
    region_masks: torch.Tensor,
    carriers: torch.Tensor,
) -> torch.Tensor:
    centered = recovered_noise - recovered_noise.flatten(1).mean(dim=1)[:, None, None, None]
    centered = centered / centered.flatten(1).std(dim=1).clamp_min(1e-6)[:, None, None, None]
    decoded = []
    for region in range(region_masks.shape[1]):
        unit = masked_unit_carriers(carriers, region_masks[0, region])
        corr = torch.einsum("bchw,kchw->bk", centered, unit)
        decoded.append((corr > 0).float())
    return torch.stack(decoded, dim=1)


def correlate_local_noise_watermark(
    recovered_noise: torch.Tensor,
    region_masks: torch.Tensor,
    carriers: torch.Tensor,
) -> torch.Tensor:
    centered = normalize_latent_noise(recovered_noise)
    correlations = []
    for region in range(region_masks.shape[1]):
        unit = masked_unit_carriers(carriers, region_masks[0, region])
        correlations.append(torch.einsum("bchw,kchw->bk", centered, unit))
    return torch.stack(correlations, dim=1)


def correlate_global_sync_pilot(recovered_noise: torch.Tensor, carriers: torch.Tensor) -> torch.Tensor:
    centered = normalize_latent_noise(recovered_noise)
    return torch.einsum("bchw,kchw->bk", centered, unit_carriers(carriers))


def decode_with_geometry_search(
    recovered_noise: torch.Tensor,
    region_masks: torch.Tensor,
    carriers: torch.Tensor,
    candidates: list[tuple[float, float, float]],
    sync_carriers: torch.Tensor | None = None,
    *,
    score_mode: str = "carrier_abs",
    outer_bits: int | None = None,
    repeat: int = 1,
    ecc: str = "none",
    codebook: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float], float]:
    best_score = -float("inf")
    best_candidate = candidates[0]
    best_decoded: torch.Tensor | None = None
    best_corr: torch.Tensor | None = None
    for candidate in candidates:
        warped_carriers, warped_masks = warp_geometry(carriers, region_masks, candidate)
        corr = correlate_local_noise_watermark(recovered_noise, warped_masks, warped_carriers)
        if score_mode == "sync_pilot" and sync_carriers is not None and sync_carriers.numel() > 0:
            crop_frac, top_frac, left_frac = candidate
            if crop_frac >= 0.999:
                warped_sync = sync_carriers
            else:
                warped_sync = crop_resize_bchw(sync_carriers, crop_frac, top_frac, left_frac, mode="bilinear")
            sync_corr = correlate_global_sync_pilot(recovered_noise, warped_sync)
            score = float(sync_corr.mean().detach().cpu())
        elif score_mode == "ecc_likelihood":
            if outer_bits is None:
                raise ValueError("outer_bits must be provided for ecc_likelihood geometry scoring.")
            score = geometry_ecc_likelihood_score(corr, outer_bits, repeat, ecc, codebook)
        elif score_mode == "carrier_abs":
            score = float(corr.abs().mean().detach().cpu())
        else:
            raise ValueError(f"Unknown geometry score mode: {score_mode}")
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_corr = corr
            best_decoded = (corr > 0).float()
    if best_decoded is None or best_corr is None:
        raise RuntimeError("No geometry candidate was evaluated.")
    return best_decoded, best_corr, best_candidate, best_score


def encode_repetition(messages: torch.Tensor, repeat: int) -> torch.Tensor:
    if repeat <= 1:
        return messages
    return messages.repeat_interleave(repeat, dim=-1)


def decode_repetition(decoded_code: torch.Tensor, output_bits: int, repeat: int) -> torch.Tensor:
    if repeat <= 1:
        return decoded_code
    batch, regions, code_bits = decoded_code.shape
    expected = output_bits * repeat
    if code_bits != expected:
        raise ValueError(f"Decoded {code_bits} bits, expected {expected} for repeat={repeat}.")
    votes = decoded_code.view(batch, regions, output_bits, repeat).mean(dim=-1)
    return (votes >= 0.5).float()


def decode_repetition_soft(code_corr: torch.Tensor, output_bits: int, repeat: int) -> torch.Tensor:
    if repeat <= 1:
        return (code_corr > 0).float()
    batch, regions, code_bits = code_corr.shape
    expected = output_bits * repeat
    if code_bits != expected:
        raise ValueError(f"Decoded {code_bits} scores, expected {expected} for repeat={repeat}.")
    votes = code_corr.view(batch, regions, output_bits, repeat).sum(dim=-1)
    return (votes > 0).float()


def hamming74_encode(messages: torch.Tensor) -> torch.Tensor:
    """Encode payload bits with systematic Hamming(7,4), using positions 3,5,6,7 as data."""
    if messages.shape[-1] % 4 != 0:
        raise ValueError("--payload-bits must be divisible by 4 for hamming74 ECC.")
    blocks = messages.view(*messages.shape[:-1], -1, 4).long()
    d1 = blocks[..., 0]
    d2 = blocks[..., 1]
    d3 = blocks[..., 2]
    d4 = blocks[..., 3]
    p1 = (d1 ^ d2 ^ d4)
    p2 = (d1 ^ d3 ^ d4)
    p4 = (d2 ^ d3 ^ d4)
    encoded = torch.stack([p1, p2, d1, p4, d2, d3, d4], dim=-1)
    return encoded.flatten(-2).float()


def hamming74_decode_hard(decoded_code: torch.Tensor) -> torch.Tensor:
    if decoded_code.shape[-1] % 7 != 0:
        raise ValueError("Hamming(7,4) code length must be divisible by 7.")
    blocks = decoded_code.long().view(*decoded_code.shape[:-1], -1, 7).clone()
    s1 = blocks[..., 0] ^ blocks[..., 2] ^ blocks[..., 4] ^ blocks[..., 6]
    s2 = blocks[..., 1] ^ blocks[..., 2] ^ blocks[..., 5] ^ blocks[..., 6]
    s4 = blocks[..., 3] ^ blocks[..., 4] ^ blocks[..., 5] ^ blocks[..., 6]
    syndrome = s1 + 2 * s2 + 4 * s4
    for position in range(1, 8):
        mask = syndrome == position
        blocks[..., position - 1] = torch.where(mask, 1 - blocks[..., position - 1], blocks[..., position - 1])
    return torch.stack([blocks[..., 2], blocks[..., 4], blocks[..., 5], blocks[..., 6]], dim=-1).flatten(-2).float()


def hamming74_codebooks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(
        [[(index >> shift) & 1 for shift in (3, 2, 1, 0)] for index in range(16)],
        device=device,
        dtype=torch.float32,
    )
    code = hamming74_encode(data[None, None]).view(16, 7)
    return data, code


def hamming74_decode_soft(code_corr: torch.Tensor) -> torch.Tensor:
    if code_corr.shape[-1] % 7 != 0:
        raise ValueError("Hamming(7,4) score length must be divisible by 7.")
    blocks = code_corr.view(*code_corr.shape[:-1], -1, 7)
    data_book, code_book = hamming74_codebooks(code_corr.device)
    signs = code_book.mul(2.0).sub(1.0)
    scores = torch.einsum("brnq,cq->brnc", blocks, signs)
    indices = scores.argmax(dim=-1)
    decoded = data_book[indices]
    return decoded.flatten(-2)


def binary_data_book(chunk_bits: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[(index >> shift) & 1 for shift in range(chunk_bits - 1, -1, -1)] for index in range(2**chunk_bits)],
        device=device,
        dtype=torch.float32,
    )


def make_binary_codebook(chunk_bits: int, code_bits: int, seed: int, device: torch.device) -> torch.Tensor:
    if chunk_bits > 10:
        raise ValueError("--codebook-chunk-bits above 10 makes exhaustive soft decoding too expensive.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    codebook = torch.randint(0, 2, (2**chunk_bits, code_bits), generator=generator, dtype=torch.float32)
    return codebook.to(device)


def codebook_encode(messages: torch.Tensor, codebook: torch.Tensor, chunk_bits: int) -> torch.Tensor:
    if messages.shape[-1] % chunk_bits != 0:
        raise ValueError("--payload-bits must be divisible by --codebook-chunk-bits for codebook ECC.")
    chunks = messages.long().view(*messages.shape[:-1], -1, chunk_bits)
    weights = (2 ** torch.arange(chunk_bits - 1, -1, -1, device=messages.device)).long()
    indices = (chunks * weights).sum(dim=-1)
    return codebook[indices].flatten(-2).float()


def codebook_decode_soft(code_corr: torch.Tensor, codebook: torch.Tensor, chunk_bits: int) -> torch.Tensor:
    code_bits = codebook.shape[-1]
    if code_corr.shape[-1] % code_bits != 0:
        raise ValueError("Codebook score length must be divisible by --codebook-code-bits.")
    blocks = code_corr.view(*code_corr.shape[:-1], -1, code_bits)
    signs = codebook.mul(2.0).sub(1.0)
    scores = torch.einsum("brnm,cm->brnc", blocks, signs)
    indices = scores.argmax(dim=-1)
    data = binary_data_book(chunk_bits, code_corr.device)
    return data[indices].flatten(-2)


def codebook_likelihood_margin(code_corr: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    code_bits = codebook.shape[-1]
    if code_corr.shape[-1] % code_bits != 0:
        raise ValueError("Codebook score length must be divisible by --codebook-code-bits.")
    blocks = code_corr.view(*code_corr.shape[:-1], -1, code_bits)
    signs = codebook.mul(2.0).sub(1.0)
    scores = torch.einsum("brnm,cm->brnc", blocks, signs)
    top2 = scores.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def hamming74_likelihood_margin(code_corr: torch.Tensor) -> torch.Tensor:
    if code_corr.shape[-1] % 7 != 0:
        raise ValueError("Hamming(7,4) score length must be divisible by 7.")
    blocks = code_corr.view(*code_corr.shape[:-1], -1, 7)
    _, code_book = hamming74_codebooks(code_corr.device)
    signs = code_book.mul(2.0).sub(1.0)
    scores = torch.einsum("brnq,cq->brnc", blocks, signs)
    top2 = scores.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def geometry_ecc_likelihood_score(
    code_corr: torch.Tensor,
    outer_bits: int,
    repeat: int,
    ecc: str,
    codebook: torch.Tensor | None = None,
) -> float:
    if repeat <= 1:
        outer_corr = code_corr
    else:
        expected = outer_bits * repeat
        if code_corr.shape[-1] != expected:
            raise ValueError(f"Decoded {code_corr.shape[-1]} scores, expected {expected} for repeat={repeat}.")
        outer_corr = code_corr.view(*code_corr.shape[:-1], outer_bits, repeat).sum(dim=-1)
    if ecc == "none":
        score = outer_corr.abs().mean()
    elif ecc == "hamming74":
        score = hamming74_likelihood_margin(outer_corr).mean()
    elif ecc == "codebook":
        if codebook is None:
            raise ValueError("codebook must be provided for codebook ECC scoring.")
        score = codebook_likelihood_margin(outer_corr, codebook).mean()
    else:
        raise ValueError(f"Unknown ECC mode: {ecc}")
    return float(score.detach().cpu())


def encode_payload(
    messages: torch.Tensor,
    ecc: str,
    codebook: torch.Tensor | None = None,
    codebook_chunk_bits: int = 8,
) -> torch.Tensor:
    if ecc == "none":
        return messages
    if ecc == "hamming74":
        return hamming74_encode(messages)
    if ecc == "codebook":
        if codebook is None:
            raise ValueError("codebook must be provided for codebook ECC.")
        return codebook_encode(messages, codebook, codebook_chunk_bits)
    raise ValueError(f"Unknown ECC mode: {ecc}")


def decode_payload_hard(
    decoded_outer: torch.Tensor,
    ecc: str,
    codebook: torch.Tensor | None = None,
    codebook_chunk_bits: int = 8,
) -> torch.Tensor:
    if ecc == "none":
        return decoded_outer
    if ecc == "hamming74":
        return hamming74_decode_hard(decoded_outer)
    if ecc == "codebook":
        if codebook is None:
            raise ValueError("codebook must be provided for codebook ECC.")
        decoded_corr = decoded_outer.mul(2.0).sub(1.0)
        return codebook_decode_soft(decoded_corr, codebook, codebook_chunk_bits)
    raise ValueError(f"Unknown ECC mode: {ecc}")


def decode_payload_soft(
    outer_corr: torch.Tensor,
    ecc: str,
    codebook: torch.Tensor | None = None,
    codebook_chunk_bits: int = 8,
) -> torch.Tensor:
    if ecc == "none":
        return (outer_corr > 0).float()
    if ecc == "hamming74":
        return hamming74_decode_soft(outer_corr)
    if ecc == "codebook":
        if codebook is None:
            raise ValueError("codebook must be provided for codebook ECC.")
        return codebook_decode_soft(outer_corr, codebook, codebook_chunk_bits)
    raise ValueError(f"Unknown ECC mode: {ecc}")


def fuse_region_correlations(
    outer_corr: torch.Tensor,
    region_masks: torch.Tensor,
    candidate: tuple[float, float, float],
    mode: str,
) -> torch.Tensor:
    if mode == "none":
        return outer_corr
    if mode == "mean":
        return outer_corr.mean(dim=1, keepdim=True)
    if mode == "visibility":
        weights = visible_region_fraction(region_masks, candidate).to(device=outer_corr.device, dtype=outer_corr.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (outer_corr * weights[..., None]).sum(dim=1, keepdim=True)
    raise ValueError(f"Unknown region fusion mode: {mode}")


@torch.no_grad()
def encode_image_to_latent(pipe: StableDiffusionPipeline, images: torch.Tensor) -> torch.Tensor:
    images = images.to(device=pipe.device, dtype=pipe.vae.dtype)
    normalized = images * 2.0 - 1.0
    posterior = pipe.vae.encode(normalized).latent_dist
    latents = posterior.mode() if hasattr(posterior, "mode") else posterior.sample()
    return latents * float(pipe.vae.config.scaling_factor)


@torch.no_grad()
def invert_image(
    pipe: StableDiffusionPipeline,
    inverse_scheduler: DDIMInverseScheduler,
    images: torch.Tensor,
    prompt: str,
    steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    device = pipe.device
    latents = encode_image_to_latent(pipe, images)
    do_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        [prompt],
        device,
        1,
        do_guidance,
        negative_prompt=[""],
    )
    if do_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    inverse_scheduler.set_timesteps(steps, device=device)
    for timestep in inverse_scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_guidance else latents
        latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = pipe.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample
        if do_guidance:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = inverse_scheduler.step(noise_pred, timestep, latents).prev_sample
    return latents


def save_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = to_pil_image(image.squeeze(0).detach().cpu().clamp(0, 1))
    pil.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=PATHS.sd21_model)
    parser.add_argument("--output-dir", type=Path, default=PATHS.runs / "noise_watermark_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--bits", type=int, default=32)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--ecc", choices=["none", "hamming74", "codebook"], default="none")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--codebook-chunk-bits", type=int, default=8)
    parser.add_argument("--codebook-code-bits", type=int, default=64)
    parser.add_argument("--regions", type=int, default=2)
    parser.add_argument("--shared-payload", action="store_true")
    parser.add_argument("--region-fusion", choices=["none", "mean", "visibility"], default="none")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--local-stat-match", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--carrier-kind", choices=["random", "smooth"], default="random")
    parser.add_argument("--carrier-base-size", type=int, default=16)
    parser.add_argument("--attacks", default="identity,jpeg,resize,center_crop_resize")
    parser.add_argument("--geometry-search", choices=["none", "center", "grid"], default="none")
    parser.add_argument(
        "--geometry-score",
        choices=["carrier_abs", "ecc_likelihood", "sync_pilot"],
        default="carrier_abs",
    )
    parser.add_argument("--search-fracs", default="1.0,0.95,0.9,0.85,0.8")
    parser.add_argument("--search-grid", type=int, default=3)
    parser.add_argument("--sync-pilot-bits", type=int, default=0)
    parser.add_argument("--sync-pilot-alpha", type=float, default=0.0)
    parser.add_argument("--sync-pilot-base-size", type=int, default=8)
    parser.add_argument("--oracle-geometry", action="store_true")
    parser.add_argument("--quality-reference", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--prompt-start-index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_root),
        local_files_only=True,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    latent_h = args.height // pipe.vae_scale_factor
    latent_w = args.width // pipe.vae_scale_factor
    payload_bits = int(args.payload_bits) if args.payload_bits is not None else args.bits
    if args.ecc == "hamming74" and payload_bits % 4 != 0:
        raise ValueError("--payload-bits must be divisible by 4 for hamming74 ECC.")
    if args.ecc == "codebook" and payload_bits % args.codebook_chunk_bits != 0:
        raise ValueError("--payload-bits must be divisible by --codebook-chunk-bits for codebook ECC.")
    if args.ecc == "none":
        outer_bits = payload_bits
    elif args.ecc == "hamming74":
        outer_bits = payload_bits // 4 * 7
    elif args.ecc == "codebook":
        outer_bits = payload_bits // args.codebook_chunk_bits * args.codebook_code_bits
    else:
        raise ValueError(f"Unknown ECC mode: {args.ecc}")
    code_bits = outer_bits * args.repeat
    if args.payload_bits is None and args.ecc == "none":
        code_bits = args.bits
        payload_bits = args.bits
        outer_bits = args.bits
    payload_codebook = None
    if args.ecc == "codebook":
        payload_codebook = make_binary_codebook(
            args.codebook_chunk_bits,
            args.codebook_code_bits,
            args.seed + 571,
            device,
        )
    if args.carrier_kind == "smooth":
        carriers = make_smooth_carriers(
            code_bits,
            pipe.unet.config.in_channels,
            latent_h,
            latent_w,
            args.carrier_base_size,
            args.seed + 17,
            device,
        )
    else:
        carriers = make_carriers(code_bits, pipe.unet.config.in_channels, latent_h, latent_w, args.seed + 17, device)
    sync_carriers = None
    if args.sync_pilot_bits > 0 and args.sync_pilot_alpha > 0:
        sync_carriers = make_smooth_carriers(
            args.sync_pilot_bits,
            pipe.unet.config.in_channels,
            latent_h,
            latent_w,
            args.sync_pilot_base_size,
            args.seed + 137,
            device,
        )
    attacks = parse_attacks(args.attacks)
    prompts = load_prompts(args.prompt_file, args.prompt_column)
    candidates = geometry_candidates(
        args.geometry_search,
        parse_float_list(args.search_fracs),
        args.search_grid,
    )
    csv_path = args.output_dir / f"results_{args.device.replace(':', '')}_seed{args.seed}.csv"

    rows: list[dict[str, str | float | int]] = []
    for index in range(args.num_samples):
        global_sample = args.prompt_start_index + index
        prompt = prompts[global_sample % len(prompts)]
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed * 1000 + global_sample)
        noise = torch.randn(
            1,
            pipe.unet.config.in_channels,
            latent_h,
            latent_w,
            generator=generator,
            device=device,
            dtype=torch.float16,
        )
        region_masks = random_rectangle_masks(1, args.regions, latent_h, latent_w, device=device)
        if args.shared_payload:
            payload_messages = sample_binary_messages(1, 1, payload_bits, device=device).repeat(1, args.regions, 1)
        else:
            payload_messages = sample_binary_messages(1, args.regions, payload_bits, device=device)
        outer_messages = encode_payload(
            payload_messages,
            args.ecc,
            payload_codebook,
            args.codebook_chunk_bits,
        )
        code_messages = encode_repetition(outer_messages, args.repeat)
        wm_noise = add_local_noise_watermark(
            noise.float(),
            region_masks,
            code_messages,
            carriers,
            args.alpha,
            normalize=False,
        )
        if sync_carriers is not None:
            wm_noise = add_global_sync_pilot(wm_noise, sync_carriers, args.sync_pilot_alpha)
        if args.local_stat_match:
            wm_noise = match_local_region_stats(wm_noise, noise.float(), region_masks)
        wm_noise = normalize_latent_noise(wm_noise).to(torch.float16)
        latent_rmse = torch.sqrt(torch.mean((wm_noise.float() - noise.float()) ** 2)).item()
        latent_mean = float(wm_noise.float().mean().detach().cpu())
        latent_std = float(wm_noise.float().std().detach().cpu())
        reference_image = None
        if args.quality_reference:
            reference_output = pipe(
                prompt=[prompt],
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                latents=noise,
                output_type="pt",
            )
            reference_image = tensor_to_bchw(reference_output.images).to(device)
        output = pipe(
            prompt=[prompt],
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            latents=wm_noise,
            output_type="pt",
        )
        image = tensor_to_bchw(output.images).to(device)
        ref_psnr = image_psnr(reference_image, image) if reference_image is not None else ""
        ref_ssim = global_ssim(reference_image, image) if reference_image is not None else ""
        if args.save_images:
            save_image(image, args.output_dir / f"sample_{args.seed}_{global_sample:06d}.png")
            if reference_image is not None:
                save_image(reference_image, args.output_dir / f"sample_{args.seed}_{global_sample:06d}_reference.png")

        for attack in attacks:
            attack_key = attack.lower()
            if attack_key in {"vae_sd", "vae_roundtrip", "vae"}:
                attacked_float = vae_roundtrip_attack(pipe, image.float())
                attack_candidate = (1.0, 0.0, 0.0)
            elif attack_key == "diffusion_regen" or attack_key.startswith("diffusion_regen_s"):
                strength = 0.25
                if attack_key.startswith("diffusion_regen_s"):
                    strength = float(attack_key.removeprefix("diffusion_regen_s"))
                attacked_float = diffusion_regen_attack(
                    pipe,
                    image.float(),
                    prompt,
                    args.steps,
                    args.guidance_scale,
                    strength,
                )
                attack_candidate = (1.0, 0.0, 0.0)
            else:
                attacked_float, attack_candidate = apply_image_attack_with_meta(image.float(), attack)
            attacked = attacked_float.to(dtype=torch.float16)
            visible = visible_region_fraction(region_masks, attack_candidate)
            recovered = invert_image(
                pipe,
                inverse_scheduler,
                attacked,
                prompt,
                args.steps,
                args.guidance_scale,
            ).float()
            decoded_code, code_corr, best_candidate, geometry_score = decode_with_geometry_search(
                recovered,
                region_masks,
                carriers,
                candidates,
                sync_carriers,
                score_mode=args.geometry_score,
                outer_bits=outer_bits,
                repeat=args.repeat,
                ecc=args.ecc,
                codebook=payload_codebook,
            )
            decoded_outer = decode_repetition(decoded_code, outer_bits, args.repeat)
            outer_corr = code_corr.view(*code_corr.shape[:-1], outer_bits, args.repeat).sum(dim=-1)
            decoded_payload = decode_payload_hard(
                decoded_outer,
                args.ecc,
                payload_codebook,
                args.codebook_chunk_bits,
            )
            decoded_payload_soft = decode_payload_soft(
                outer_corr,
                args.ecc,
                payload_codebook,
                args.codebook_chunk_bits,
            )
            fused_soft_payload_bit_acc: str | float = ""
            fused_soft_payload_exact_acc: str | float = ""
            fused_sample_extract_acc: str | float = ""
            if args.shared_payload and args.region_fusion != "none":
                fused_outer_corr = fuse_region_correlations(outer_corr, region_masks, best_candidate, args.region_fusion)
                fused_payload_soft = decode_payload_soft(
                    fused_outer_corr,
                    args.ecc,
                    payload_codebook,
                    args.codebook_chunk_bits,
                )
                target_payload = payload_messages[:, :1]
                fused_soft_payload_bit_acc = (fused_payload_soft == target_payload).float().mean().item()
                fused_soft_payload_exact_acc = (
                    (fused_payload_soft == target_payload).all(dim=-1).float().mean().item()
                )
                fused_sample_extract_acc = float((fused_payload_soft == target_payload).all().float().cpu())
            code_bit_acc = (decoded_code == code_messages).float().mean().item()
            payload_bit_acc = (decoded_payload == payload_messages).float().mean().item()
            payload_exact_acc = (decoded_payload == payload_messages).all(dim=-1).float().mean().item()
            soft_payload_bit_acc = (decoded_payload_soft == payload_messages).float().mean().item()
            soft_payload_exact_acc = (
                (decoded_payload_soft == payload_messages).all(dim=-1).float().mean().item()
            )
            soft_sample_extract_acc = float((decoded_payload_soft == payload_messages).all().float().cpu())
            oracle_soft_payload_bit_acc: str | float = ""
            oracle_soft_payload_exact_acc: str | float = ""
            oracle_soft_sample_extract_acc: str | float = ""
            if args.oracle_geometry and attack_candidate is not None:
                _, oracle_corr, _, _ = decode_with_geometry_search(
                    recovered,
                    region_masks,
                    carriers,
                    [attack_candidate],
                    sync_carriers,
                    score_mode=args.geometry_score,
                    outer_bits=outer_bits,
                    repeat=args.repeat,
                    ecc=args.ecc,
                    codebook=payload_codebook,
                )
                oracle_outer_corr = oracle_corr.view(*oracle_corr.shape[:-1], outer_bits, args.repeat).sum(dim=-1)
                oracle_payload_soft = decode_payload_soft(
                    oracle_outer_corr,
                    args.ecc,
                    payload_codebook,
                    args.codebook_chunk_bits,
                )
                oracle_soft_payload_bit_acc = (oracle_payload_soft == payload_messages).float().mean().item()
                oracle_soft_payload_exact_acc = (
                    (oracle_payload_soft == payload_messages).all(dim=-1).float().mean().item()
                )
                oracle_soft_sample_extract_acc = float((oracle_payload_soft == payload_messages).all().float().cpu())
            rows.append(
                {
                    "sample": index,
                    "global_sample": global_sample,
                    "prompt_file": "" if args.prompt_file is None else str(args.prompt_file),
                    "attack": attack,
                    "payload_bits": payload_bits,
                    "outer_bits": outer_bits,
                    "code_bits": code_bits,
                    "ecc": args.ecc,
                    "repeat": args.repeat,
                    "codebook_chunk_bits": args.codebook_chunk_bits if args.ecc == "codebook" else "",
                    "codebook_code_bits": args.codebook_code_bits if args.ecc == "codebook" else "",
                    "regions": args.regions,
                    "shared_payload": int(args.shared_payload),
                    "region_fusion": args.region_fusion,
                    "alpha": args.alpha,
                    "local_stat_match": int(args.local_stat_match),
                    "carrier_kind": args.carrier_kind,
                    "carrier_base_size": args.carrier_base_size,
                    "geometry_search": args.geometry_search,
                    "geometry_score_mode": args.geometry_score,
                    "sync_pilot_bits": args.sync_pilot_bits,
                    "sync_pilot_alpha": args.sync_pilot_alpha,
                    "sync_pilot_base_size": args.sync_pilot_base_size,
                    "best_crop_frac": best_candidate[0],
                    "best_top_frac": best_candidate[1],
                    "best_left_frac": best_candidate[2],
                    "attack_crop_frac": "" if attack_candidate is None else attack_candidate[0],
                    "attack_top_frac": "" if attack_candidate is None else attack_candidate[1],
                    "attack_left_frac": "" if attack_candidate is None else attack_candidate[2],
                    "visible_region_mean": float(visible.mean().detach().cpu()),
                    "visible_region_min": float(visible.min().detach().cpu()),
                    "geometry_score": geometry_score,
                    "ref_psnr": ref_psnr,
                    "ref_ssim": ref_ssim,
                    "latent_rmse": latent_rmse,
                    "watermarked_noise_mean": latent_mean,
                    "watermarked_noise_std": latent_std,
                    "code_bit_acc": code_bit_acc,
                    "payload_bit_acc": payload_bit_acc,
                    "payload_exact_acc": payload_exact_acc,
                    "soft_payload_bit_acc": soft_payload_bit_acc,
                    "soft_payload_exact_acc": soft_payload_exact_acc,
                    "soft_sample_extract_acc": soft_sample_extract_acc,
                    "fused_soft_payload_bit_acc": fused_soft_payload_bit_acc,
                    "fused_soft_payload_exact_acc": fused_soft_payload_exact_acc,
                    "fused_sample_extract_acc": fused_sample_extract_acc,
                    "oracle_soft_payload_bit_acc": oracle_soft_payload_bit_acc,
                    "oracle_soft_payload_exact_acc": oracle_soft_payload_exact_acc,
                    "oracle_soft_sample_extract_acc": oracle_soft_sample_extract_acc,
                    "prompt": prompt,
                }
            )
            fusion_text = ""
            if args.shared_payload and args.region_fusion != "none":
                fusion_text = (
                    f" fused_bit={float(fused_soft_payload_bit_acc):.4f}"
                    f" fused_exact={float(fused_soft_payload_exact_acc):.4f}"
                    f" fused_extract={float(fused_sample_extract_acc):.4f}"
                )
            print(
                f"sample={index} global={global_sample} attack={attack} payload={payload_bits} code={code_bits} "
                f"repeat={args.repeat} alpha={args.alpha} pilot={args.sync_pilot_bits}/{args.sync_pilot_alpha} "
                f"code_bit={code_bit_acc:.4f} "
                f"payload_bit={payload_bit_acc:.4f} payload_exact={payload_exact_acc:.4f} "
                f"soft_bit={soft_payload_bit_acc:.4f} soft_exact={soft_payload_exact_acc:.4f} "
                f"sample_extract={soft_sample_extract_acc:.4f} "
                f"sync=({best_candidate[0]:.2f},{best_candidate[1]:.2f},{best_candidate[2]:.2f}) "
                f"visible={float(visible.mean().detach().cpu()):.3f}/{float(visible.min().detach().cpu()):.3f}"
                f"{fusion_text}"
            )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
