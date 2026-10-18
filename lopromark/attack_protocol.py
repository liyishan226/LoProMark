from __future__ import annotations

import io
import random
from pathlib import Path

import torch
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

from .attacks import jpeg_roundtrip, resize_roundtrip


MODEL_ATTACKS = {"vae_sd", "diffusion_regen_s0.25"}


def pil_to_bchw(image: Image.Image, device: str | torch.device = "cpu") -> torch.Tensor:
    tensor = pil_to_tensor(image.convert("RGB")).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def bchw_to_pil(image: torch.Tensor) -> Image.Image:
    return to_pil_image(image.squeeze(0).detach().cpu().clamp(0, 1))


def tensor_to_bchw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected a 4D image tensor, got {tuple(images.shape)}")
    if images.shape[1] in {1, 3, 4}:
        return images[:, :3].contiguous().clamp(0.0, 1.0)
    return images.permute(0, 3, 1, 2)[:, :3].contiguous().clamp(0.0, 1.0)


def parse_attacks(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def apply_tensor_attack(
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
        rotated = rotate(images.float(), degrees, interpolation=InterpolationMode.BILINEAR, fill=[0.0, 0.0, 0.0])
        return rotated.clamp(0, 1), None
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
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
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
        resized = torch.nn.functional.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False)
        return resized.clamp(0, 1), (crop_frac, 0.5, 0.5)
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
        resized = torch.nn.functional.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False)
        return resized.clamp(0, 1), (crop_frac, top_frac, left_frac)
    raise ValueError(f"Unknown attack: {name}")


@torch.no_grad()
def encode_image_to_latent(pipe, images: torch.Tensor) -> torch.Tensor:
    images = images.to(device=pipe.device, dtype=pipe.vae.dtype)
    normalized = images * 2.0 - 1.0
    posterior = pipe.vae.encode(normalized).latent_dist
    latents = posterior.mode() if hasattr(posterior, "mode") else posterior.sample()
    return latents * float(pipe.vae.config.scaling_factor)


@torch.no_grad()
def vae_roundtrip_attack(pipe, images: torch.Tensor) -> torch.Tensor:
    latents = encode_image_to_latent(pipe, images.float())
    decoded = pipe.vae.decode(latents / float(pipe.vae.config.scaling_factor)).sample
    return ((decoded.float() / 2.0) + 0.5).clamp(0, 1)


@torch.no_grad()
def diffusion_regen_attack(
    pipe,
    images: torch.Tensor,
    prompt: str,
    steps: int = 25,
    guidance_scale: float = 7.5,
    strength: float = 0.25,
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


def apply_pil_attack(
    image: Image.Image,
    attack: str,
    *,
    device: str | torch.device = "cpu",
    pipe=None,
    prompt: str = "",
    regen_steps: int = 25,
    regen_guidance_scale: float = 7.5,
) -> Image.Image:
    tensor = pil_to_bchw(image, device=device)
    if attack == "vae_sd":
        if pipe is None:
            raise ValueError("vae_sd attack requires a diffusion pipeline")
        attacked = vae_roundtrip_attack(pipe, tensor.to(pipe.device))
        return bchw_to_pil(attacked)
    if attack == "diffusion_regen_s0.25":
        if pipe is None:
            raise ValueError("diffusion_regen_s0.25 attack requires a diffusion pipeline")
        attacked = diffusion_regen_attack(
            pipe,
            tensor.to(pipe.device),
            prompt=prompt,
            steps=regen_steps,
            guidance_scale=regen_guidance_scale,
            strength=0.25,
        )
        return bchw_to_pil(attacked)
    attacked, _ = apply_tensor_attack(tensor, attack)
    return bchw_to_pil(attacked)


def png_roundtrip(image: Image.Image, path: Path) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return Image.open(path).convert("RGB")
