"""COCO object-mask and multi-prompt benchmark for the coded pipeline.

Pairs each marked output with an unmarked output under semantic object
masks and caption or category prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw
from scipy import ndimage
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.mask_head import MaskHead, normalize_input


DEFAULT_COVER_DIR = Path("data/val2017")
DEFAULT_INSTANCES = Path(
    "data/annotations/instances_val2017.json"
)
DEFAULT_CAPTIONS = Path(
    "data/annotations/captions_val2017.json"
)


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


P = _load("scripts/carrier_core.py", "om_carrier_core")
B = _load("scripts/build_maskhead_dataset.py", "om_maskhead_dataset")
E = _load("scripts/eval_per_attack.py", "om_per_attack_eval")
CRC = _load("scripts/crc_listdecode.py", "om_crc_listdecode")


CATEGORY_TEMPLATES = (
    "A realistic photograph featuring {article} {category}.",
    "A detailed natural photo of {article} {category}.",
    "A documentary-style photograph centered on {article} {category}.",
    "A high-quality photograph showing {article} {category} in context.",
)


@dataclass
class SelectedInstance:
    index: int
    global_index: int
    cover_path: Path
    image_id: int
    image_width: int
    image_height: int
    ann_id: int
    category_id: int
    category_name: str
    caption_id: int
    caption: str
    prompt_source: str
    prompt: str
    full_mask: torch.Tensor  # [1, size, size] bool, one connected component
    latent_mask: torch.Tensor  # [1, H, W] bool, one connected component
    original_component_area: float


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def article_for(category: str) -> str:
    return "an" if category[:1].lower() in "aeiou" else "a"


def stable_rank(seed: int, ann_id: int) -> str:
    # Seeded deterministic shuffle of annotation ids.
    return hashlib.sha256(f"{seed}:{ann_id}".encode("utf-8")).hexdigest()


def polygon_mask(segmentation: Any, width: int, height: int) -> np.ndarray | None:
    """Rasterize COCO polygon segmentation without requiring pycocotools.

    Crowd/RLE annotations are excluded by the selector. Multiple polygons are
    unioned here, then the selector keeps the largest connected component.
    """
    if not isinstance(segmentation, list) or not segmentation:
        return None
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    valid = False
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        draw.polygon(points, outline=1, fill=1)
        valid = True
    if not valid:
        return None
    return np.asarray(canvas, dtype=np.uint8).astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.reshape(-1))
    areas[0] = 0
    return labels == int(areas.argmax())


def resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((width, height), resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) >= 128


def choose_prompt(
    prompt_mode: str,
    global_index: int,
    category: str,
    caption: str,
) -> tuple[str, str]:
    if prompt_mode == "caption":
        source = "caption"
    elif prompt_mode == "category":
        source = "category"
    else:
        source = "caption" if global_index % 2 == 0 else "category"
    if source == "caption":
        return source, clean_text(caption)
    template = CATEGORY_TEMPLATES[global_index % len(CATEGORY_TEMPLATES)]
    return source, template.format(article=article_for(category), category=category)


def select_instances(args: argparse.Namespace, latent_height: int, latent_width: int) -> list[SelectedInstance]:
    instances = json.loads(args.instances_json.read_text(encoding="utf-8"))
    captions_data = json.loads(args.captions_json.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in instances["images"]}
    categories = {int(item["id"]): str(item["name"]) for item in instances["categories"]}
    captions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in captions_data["annotations"]:
        captions_by_image[int(item["image_id"])].append(item)
    for values in captions_by_image.values():
        values.sort(key=lambda item: int(item["id"]))

    candidates = []
    for ann in instances["annotations"]:
        image_id = int(ann["image_id"])
        image = images.get(image_id)
        if image is None or image_id not in captions_by_image:
            continue
        if int(ann.get("iscrowd", 0)) != 0 or not isinstance(ann.get("segmentation"), list):
            continue
        width, height = int(image["width"]), int(image["height"])
        coarse_fraction = float(ann.get("area", 0.0)) / max(1, width * height)
        if coarse_fraction < args.frac_min * 0.35 or coarse_fraction > min(1.0, args.frac_max * 2.0):
            continue
        cover_path = args.cover_dir / str(image["file_name"])
        if not cover_path.exists():
            continue
        candidates.append(ann)
    candidates.sort(key=lambda ann: stable_rank(args.seed, int(ann["id"])))

    required = args.sample_offset + args.num_samples
    selected_all: list[SelectedInstance] = []
    used_images: set[int] = set()
    category_counts: dict[int, int] = defaultdict(int)
    for ann in candidates:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id in used_images:
            continue
        if args.max_per_category > 0 and category_counts[category_id] >= args.max_per_category:
            continue
        image = images[image_id]
        width, height = int(image["width"]), int(image["height"])
        raw = polygon_mask(ann["segmentation"], width, height)
        if raw is None:
            continue
        component = largest_component(raw)
        original_area = float(component.mean())
        full = largest_component(resize_binary_mask(component, args.size, args.size))
        full_area = float(full.mean())
        if not (args.frac_min <= full_area <= args.frac_max):
            continue
        latent = largest_component(resize_binary_mask(full, latent_width, latent_height))
        latent_area = float(latent.mean())
        if latent.sum() == 0 or not (args.frac_min * 0.8 <= latent_area <= args.frac_max * 1.2):
            continue

        captions = captions_by_image[image_id]
        caption = captions[(args.seed + int(ann["id"])) % len(captions)]
        global_index = len(selected_all)
        category_name = categories[category_id]
        prompt_source, prompt = choose_prompt(
            args.prompt_mode,
            global_index,
            category_name,
            clean_text(caption["caption"]),
        )
        selected_all.append(
            SelectedInstance(
                index=-1,
                global_index=global_index,
                cover_path=args.cover_dir / str(image["file_name"]),
                image_id=image_id,
                image_width=width,
                image_height=height,
                ann_id=int(ann["id"]),
                category_id=category_id,
                category_name=category_name,
                caption_id=int(caption["id"]),
                caption=clean_text(caption["caption"]),
                prompt_source=prompt_source,
                prompt=prompt,
                full_mask=torch.from_numpy(full.copy()).bool().unsqueeze(0),
                latent_mask=torch.from_numpy(latent.copy()).bool().unsqueeze(0),
                original_component_area=original_area,
            )
        )
        used_images.add(image_id)
        category_counts[category_id] += 1
        if len(selected_all) >= required:
            break

    if len(selected_all) < required:
        raise RuntimeError(
            f"Only selected {len(selected_all)} valid unique-image instances; need {required}. "
            "Relax --max-per-category or the area interval."
        )
    result = selected_all[args.sample_offset:required]
    for local_index, item in enumerate(result):
        item.index = local_index
    return result


def selection_rows(selected: list[SelectedInstance]) -> list[dict[str, Any]]:
    rows = []
    for item in selected:
        rows.append(
            {
                "idx": item.index,
                "global_idx": item.global_index,
                "image_id": item.image_id,
                "image_width": item.image_width,
                "image_height": item.image_height,
                "image_file": item.cover_path.name,
                "ann_id": item.ann_id,
                "category_id": item.category_id,
                "category_name": item.category_name,
                "caption_id": item.caption_id,
                "caption": item.caption,
                "prompt_source": item.prompt_source,
                "prompt": item.prompt,
                "original_largest_component_area": item.original_component_area,
                "resized_full_area": float(item.full_mask.float().mean()),
                "latent_area": float(item.latent_mask.float().mean()),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_alphas(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--alphas must contain unique values")
    if 0.0 not in values:
        raise ValueError("--alphas must include 0 for strict paired quality")
    return values


def alpha_name(alpha: float) -> str:
    return f"a{alpha:g}"


def load_cover(path: Path, size: int, device: torch.device) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image)
    image = Image.open(path).convert("RGB").resize((size, size), resampling.BICUBIC)
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return tensor.to(device)


def make_callback(edit_mask: torch.Tensor, z_real: torch.Tensor, background_noise: torch.Tensor):
    def callback(pipe, step, timestep, kwargs):
        latents = kwargs["latents"].float()
        timesteps = pipe.scheduler.timesteps
        if step + 1 < len(timesteps):
            background = pipe.scheduler.add_noise(
                z_real,
                background_noise,
                timesteps[step + 1].unsqueeze(0),
            )
        else:
            background = z_real
        blended = edit_mask * latents + (1.0 - edit_mask) * background
        return {"latents": blended.to(kwargs["latents"].dtype)}

    return callback


def load_pipe(args: argparse.Namespace, device: torch.device) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_root),
        local_files_only=True,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def build_artifacts(
    args: argparse.Namespace,
    selected: list[SelectedInstance],
    alphas: list[float],
    pipe: StableDiffusionPipeline,
    device: torch.device,
) -> None:
    height = args.size // pipe.vae_scale_factor
    width = args.size // pipe.vae_scale_factor
    total_bits = args.data_bits + args.crc_bits
    if total_bits % args.chunk_bits:
        raise ValueError("data bits plus CRC bits must be divisible by chunk bits")
    outer_bits = total_bits // args.chunk_bits * args.code_bits
    codebook = P.make_binary_codebook(args.chunk_bits, args.code_bits, args.seed + 571, device)
    carriers = P.make_carriers(
        outer_bits,
        pipe.unet.config.in_channels,
        height,
        width,
        args.seed + 17,
        device,
    )

    images_by_alpha: dict[float, list[torch.Tensor]] = {alpha: [] for alpha in alphas}
    payloads: list[torch.Tensor] = []
    data_payloads: list[torch.Tensor] = []
    pair_rows: list[dict[str, Any]] = []
    started = time.time()

    for item in selected:
        cover = load_cover(item.cover_path, args.size, device)
        z_real = P.encode_image_to_latent(pipe, cover.to(torch.float16)).float()
        edit_mask = item.latent_mask.unsqueeze(0).float().to(device)
        data_seed = args.seed + 1_000_003 + item.global_index
        base_seed = args.seed + 3_000_003 + item.global_index
        background_seed = args.seed + 4_000_003 + item.global_index
        generation_seed = args.seed + 5_000_003 + item.global_index

        data_generator = torch.Generator(device="cpu")
        data_generator.manual_seed(data_seed)
        data_cpu = torch.randint(
            0,
            2,
            (args.data_bits,),
            generator=data_generator,
            dtype=torch.float32,
        )
        payload_1d = CRC.make_crc_payload(data_cpu, args.crc_bits, device)
        encoded = P.encode_payload(
            payload_1d.view(1, 1, -1),
            "codebook",
            codebook,
            args.chunk_bits,
        )
        base_generator = torch.Generator(device=device)
        base_generator.manual_seed(base_seed)
        base_noise = torch.randn(
            1,
            pipe.unet.config.in_channels,
            height,
            width,
            device=device,
            generator=base_generator,
        ).float()
        background_generator = torch.Generator(device=device)
        background_generator.manual_seed(background_seed)
        background_noise = torch.randn(
            z_real.shape,
            device=device,
            dtype=z_real.dtype,
            generator=background_generator,
        )
        pipe.scheduler.set_timesteps(args.steps, device=device)
        first_timestep = pipe.scheduler.timesteps[0].unsqueeze(0)
        background_initial = pipe.scheduler.add_noise(z_real, background_noise, first_timestep)
        for alpha in alphas:
            random.seed(generation_seed)
            torch.manual_seed(generation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(generation_seed)
            watermarked = P.add_local_noise_watermark(
                base_noise.clone(),
                edit_mask,
                encoded,
                carriers,
                alpha,
                normalize=False,
            )
            watermarked = P.match_local_region_stats(watermarked, base_noise, edit_mask)
            watermarked = P.normalize_latent_noise(watermarked)
            x_t = edit_mask * (watermarked * pipe.scheduler.init_noise_sigma) + (
                1.0 - edit_mask
            ) * background_initial
            pipe.scheduler.set_timesteps(args.steps, device=device)
            generated = P.tensor_to_bchw(
                pipe(
                    prompt=[item.prompt],
                    height=args.size,
                    width=args.size,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    latents=x_t.to(torch.float16),
                    output_type="pt",
                    callback_on_step_end=make_callback(edit_mask, z_real, background_noise),
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
            )
            image_u8 = (generated[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            images_by_alpha[alpha].append(image_u8)
            if item.index < args.save_previews:
                preview_dir = args.outdir / "previews"
                P.save_image(generated, preview_dir / f"{item.index:03d}_{alpha_name(alpha)}.png")

        if item.index < args.save_previews:
            P.save_image(cover, args.outdir / "previews" / f"{item.index:03d}_cover.png")
            mask_image = item.full_mask.unsqueeze(0).float().to(device).repeat(1, 3, 1, 1)
            P.save_image(mask_image, args.outdir / "previews" / f"{item.index:03d}_mask.png")

        payloads.append(payload_1d.bool().cpu())
        data_payloads.append(data_cpu.bool().cpu())
        pair_rows.append(
            {
                "idx": item.index,
                "global_idx": item.global_index,
                "image_id": item.image_id,
                "ann_id": item.ann_id,
                "category_id": item.category_id,
                "caption_id": item.caption_id,
                "prompt_source": item.prompt_source,
                "data_seed": data_seed,
                "base_seed": base_seed,
                "background_seed": background_seed,
                "generation_seed": generation_seed,
            }
        )
        elapsed = (time.time() - started) / (item.index + 1)
        print(
            f"[build {item.index + 1}/{len(selected)}] image={item.image_id} "
            f"ann={item.ann_id} source={item.prompt_source} {elapsed:.2f}s/sample",
            flush=True,
        )

    latent_masks = torch.stack([item.latent_mask for item in selected])
    full_masks = torch.stack([item.full_mask for item in selected])
    shared_meta = {
        "seed": args.seed,
        "payload_bits": total_bits,
        "data_bits": args.data_bits,
        "crc_bits": args.crc_bits,
        "chunk": args.chunk_bits,
        "code_bits": args.code_bits,
        "size": args.size,
        "H": height,
        "W": width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strict_pairing": True,
        "mask_definition": (
            "largest 8-connected component of one non-crowd COCO polygon instance; "
            "resized to 512 and latent support with nearest-neighbor"
        ),
        "prompt_protocol": args.prompt_mode,
        "prompts": [item.prompt for item in selected],
        "prompt_sources": [item.prompt_source for item in selected],
        "global_indices": [item.global_index for item in selected],
        "image_ids": [item.image_id for item in selected],
        "ann_ids": [item.ann_id for item in selected],
        "category_ids": [item.category_id for item in selected],
        "caption_ids": [item.caption_id for item in selected],
    }
    for alpha in alphas:
        alpha_dir = args.outdir / alpha_name(alpha)
        alpha_dir.mkdir(parents=True, exist_ok=False)
        torch.save(
            {
                "images": torch.stack(images_by_alpha[alpha]),
                "masks": latent_masks,
                "full_masks": full_masks,
                "payloads": torch.stack(payloads),
                "data_payloads": torch.stack(data_payloads),
                "meta": {**shared_meta, "alpha": alpha},
            },
            alpha_dir / "pos.pt",
        )
        print(f"saved {alpha_dir / 'pos.pt'}", flush=True)
    write_rows(args.outdir / "pairing_manifest.csv", pair_rows)


def validate_artifacts(artifacts: dict[float, dict[str, Any]]) -> dict[str, Any]:
    reference = artifacts[0.0]
    if not reference["meta"].get("strict_pairing"):
        raise ValueError("alpha0 artifact does not declare strict pairing")
    invariant_meta = (
        "seed",
        "payload_bits",
        "data_bits",
        "crc_bits",
        "chunk",
        "code_bits",
        "size",
        "H",
        "W",
        "steps",
        "guidance_scale",
        "prompts",
        "prompt_sources",
        "global_indices",
        "image_ids",
        "ann_ids",
        "category_ids",
        "caption_ids",
    )
    for alpha, artifact in artifacts.items():
        if float(artifact["meta"].get("alpha", math.nan)) != alpha:
            raise ValueError(f"Artifact alpha mismatch for {alpha}")
        for key in invariant_meta:
            if artifact["meta"].get(key) != reference["meta"].get(key):
                raise ValueError(f"Metadata pairing mismatch: alpha={alpha}, key={key}")
        for key in ("masks", "full_masks", "payloads", "data_payloads"):
            if not torch.equal(artifact[key], reference[key]):
                raise ValueError(f"Tensor pairing mismatch: alpha={alpha}, key={key}")
    return reference["meta"]


def masked_rmse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    denominator = (mask.sum() * a.shape[1]).clamp_min(1.0)
    return float(((((a - b) ** 2) * mask).sum() / denominator).sqrt())


def crop_with_context(image: torch.Tensor, mask: torch.Tensor, padding: float) -> torch.Tensor:
    positions = torch.nonzero(mask[0, 0] > 0.5, as_tuple=False)
    if positions.numel() == 0:
        return image
    y0, x0 = positions.min(dim=0).values.tolist()
    y1, x1 = positions.max(dim=0).values.tolist()
    height, width = image.shape[-2:]
    pad = int(round(max(y1 - y0 + 1, x1 - x0 + 1) * padding))
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(height - 1, y1 + pad), min(width - 1, x1 + pad)
    return image[:, :, y0 : y1 + 1, x0 : x1 + 1]


@torch.no_grad()
def clip_text_similarity(
    model,
    processor,
    image: torch.Tensor,
    text_value: str,
    device: torch.device,
) -> float:
    pil = to_pil_image(image[0].detach().cpu().clamp(0, 1))
    inputs = processor(
        text=[text_value],
        images=[pil],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    image_features = F.normalize(outputs.image_embeds.float(), dim=-1)
    text_features = F.normalize(outputs.text_embeds.float(), dim=-1)
    return float((image_features * text_features).sum())


def binary_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target.float(), size=prediction.shape[-2:], mode="nearest")
    pred = prediction.bool()
    truth = target.bool()
    intersection = (pred & truth).sum().float()
    union = (pred | truth).sum().float().clamp_min(1.0)
    return float(intersection / union)


def reset_attack_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def mean_or_nan(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") not in ("", None)]
    return sum(values) / len(values) if values else float("nan")


def summarize_results(
    quality_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    sources = ["all", "caption", "category"]
    keys = sorted({(float(row["alpha"]), str(row["attack"])) for row in eval_rows})
    for alpha, attack in keys:
        for source in sources:
            eval_group = [
                row
                for row in eval_rows
                if float(row["alpha"]) == alpha
                and str(row["attack"]) == attack
                and (source == "all" or row["prompt_source"] == source)
            ]
            quality_group = [
                row
                for row in quality_rows
                if float(row["alpha"]) == alpha
                and (source == "all" or row["prompt_source"] == source)
            ]
            if not eval_group:
                continue
            valid = [row for row in eval_group if int(row["crc_valid"])]
            output.append(
                {
                    "alpha": alpha,
                    "attack": attack,
                    "prompt_source": source,
                    "n": len(eval_group),
                    "crc_valid": mean_or_nan(eval_group, "crc_valid"),
                    "crc_reject": mean_or_nan(eval_group, "crc_reject"),
                    "data_exact": mean_or_nan(eval_group, "data_exact"),
                    "data_bit_given_crc": mean_or_nan(valid, "data_bit"),
                    "latent_iou": mean_or_nan(eval_group, "latent_iou"),
                    "det": mean_or_nan(eval_group, "det"),
                    "full_rmse_vs_alpha0": mean_or_nan(quality_group, "full_rmse_vs_alpha0"),
                    "region_rmse_vs_alpha0": mean_or_nan(quality_group, "region_rmse_vs_alpha0"),
                    "background_rmse_vs_alpha0": mean_or_nan(
                        quality_group, "background_rmse_vs_alpha0"
                    ),
                    "crop_clip_to_text": mean_or_nan(quality_group, "crop_clip_to_text"),
                    "crop_clip_delta_vs_alpha0": mean_or_nan(
                        quality_group, "crop_clip_delta_vs_alpha0"
                    ),
                }
            )
    return output


@torch.no_grad()
def evaluate_artifacts(
    args: argparse.Namespace,
    alphas: list[float],
    pipe: StableDiffusionPipeline,
    device: torch.device,
) -> None:
    result_paths = [
        args.outdir / "quality_rows.csv",
        args.outdir / "eval_rows.csv",
        args.outdir / "summary.csv",
        args.outdir / "eval_config.json",
    ]
    existing = [str(path) for path in result_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite evaluation outputs: {existing}")
    artifact_paths = {alpha: args.outdir / alpha_name(alpha) / "pos.pt" for alpha in alphas}
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing built artifacts: {missing}")
    artifacts = {
        alpha: torch.load(path, map_location="cpu", mmap=True)
        for alpha, path in artifact_paths.items()
    }
    meta = validate_artifacts(artifacts)
    count = int(artifacts[0.0]["images"].shape[0])
    prompts = list(meta["prompts"])
    prompt_sources = list(meta["prompt_sources"])

    clip_model = clip_processor = None
    clip_device = torch.device(args.clip_device or args.device)
    clip_error = None
    if not args.skip_clip:
        try:
            from transformers import CLIPModel, CLIPProcessor

            clip_model = CLIPModel.from_pretrained(
                args.clip_model,
                local_files_only=True,
            ).to(clip_device).eval()
            clip_processor = CLIPProcessor.from_pretrained(
                args.clip_model,
                local_files_only=True,
            )
        except Exception as error:  # CLIP is optional
            clip_error = f"{type(error).__name__}: {error}"
            warnings.warn(f"CLIP unavailable; crop metric will be blank: {clip_error}")

    quality_rows: list[dict[str, Any]] = []
    for index in range(count):
        reference = artifacts[0.0]["images"][index : index + 1].float().to(device) / 255.0
        full_mask = artifacts[0.0]["full_masks"][index : index + 1].float().to(device)
        clip_reference = None
        if clip_model is not None:
            reference_crop = crop_with_context(reference, full_mask, args.crop_padding)
            clip_reference = clip_text_similarity(
                clip_model,
                clip_processor,
                reference_crop.to(clip_device),
                prompts[index],
                clip_device,
            )
        for alpha in alphas:
            image = artifacts[alpha]["images"][index : index + 1].float().to(device) / 255.0
            clip_score: float | str = ""
            clip_delta: float | str = ""
            if clip_model is not None:
                crop = crop_with_context(image, full_mask, args.crop_padding)
                clip_score = clip_text_similarity(
                    clip_model,
                    clip_processor,
                    crop.to(clip_device),
                    prompts[index],
                    clip_device,
                )
                assert clip_reference is not None
                clip_delta = clip_score - clip_reference
            quality_rows.append(
                {
                    "alpha": alpha,
                    "idx": index,
                    "global_idx": meta["global_indices"][index],
                    "image_id": meta["image_ids"][index],
                    "ann_id": meta["ann_ids"][index],
                    "category_id": meta["category_ids"][index],
                    "caption_id": meta["caption_ids"][index],
                    "prompt_source": prompt_sources[index],
                    "full_rmse_vs_alpha0": float(torch.sqrt(torch.mean((image - reference) ** 2))),
                    "region_rmse_vs_alpha0": masked_rmse(image, reference, full_mask),
                    "background_rmse_vs_alpha0": masked_rmse(
                        image, reference, 1.0 - full_mask
                    ),
                    "crop_clip_to_text": clip_score,
                    "crop_clip_delta_vs_alpha0": clip_delta,
                }
            )
        print(f"[quality {index + 1}/{count}]", flush=True)
    if clip_model is not None:
        del clip_model, clip_processor
        if clip_device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint = torch.load(args.ckpt, map_location=device)
    model = MaskHead(in_ch=checkpoint["meta"]["in_ch"], base=checkpoint["base"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    carriers, codebook = CRC.make_key(meta, device)
    crc_config = CRC.Config(meta["data_bits"], meta["crc_bits"], (args.topk,))
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    attacks = ("clean", f"diffusion_regen_s{args.regen_strength:g}")
    eval_rows: list[dict[str, Any]] = []
    for attack_index, attack in enumerate(attacks):
        for index in range(count):
            target = artifacts[0.0]["data_payloads"][index].float().to(device)
            true_mask = artifacts[0.0]["masks"][index : index + 1].float().to(device)
            attack_seed = args.attack_seed + attack_index * 1_000_003 + int(
                meta["global_indices"][index]
            )
            for alpha in alphas:
                image = artifacts[alpha]["images"][index : index + 1].float().to(device) / 255.0
                reset_attack_seed(attack_seed, device)
                if attack == "clean":
                    received = image
                else:
                    received = E.apply_attack(
                        pipe,
                        attack,
                        image,
                        prompts[index],
                        meta["steps"],
                        meta["guidance_scale"],
                    )
                recovered = P.invert_image(
                    pipe,
                    inverse,
                    received.to(torch.float16),
                    prompts[index],
                    meta["steps"],
                    meta["guidance_scale"],
                ).float()
                margins = B.margin_maps_fast(
                    recovered,
                    carriers,
                    codebook,
                    meta["H"],
                    meta["W"],
                    checkpoint["meta"]["margin_wins"],
                    checkpoint["meta"]["margin_stride"],
                )[None]
                probability = torch.sigmoid(
                    model(normalize_input(torch.cat([recovered, margins], dim=1)))
                )
                correlation = E._corr(recovered, probability, carriers)
                detection_margin = float(P.codebook_likelihood_margin(correlation, codebook).mean())
                scores = CRC.codebook_scores(correlation, codebook)
                selected, _ = CRC.select_crc_candidate(
                    scores,
                    args.topk,
                    crc_config,
                    meta["chunk"],
                    device,
                )
                crc_valid = int(selected is not None)
                data_bit: float | str = ""
                data_exact = 0
                if selected is not None:
                    decoded = selected[: crc_config.data_bits]
                    data_bit = float((decoded == target).float().mean())
                    data_exact = int(bool((decoded == target).all()))
                eval_rows.append(
                    {
                        "alpha": alpha,
                        "attack": attack,
                        "idx": index,
                        "global_idx": meta["global_indices"][index],
                        "image_id": meta["image_ids"][index],
                        "ann_id": meta["ann_ids"][index],
                        "category_id": meta["category_ids"][index],
                        "caption_id": meta["caption_ids"][index],
                        "prompt_source": prompt_sources[index],
                        "attack_seed": attack_seed,
                        "det": detection_margin,
                        "crc_valid": crc_valid,
                        "crc_reject": 1 - crc_valid,
                        "data_bit": data_bit,
                        "data_exact": data_exact,
                        "latent_iou": binary_iou(
                            probability > args.mask_threshold,
                            true_mask,
                        ),
                    }
                )
            print(f"[eval {attack} {index + 1}/{count}]", flush=True)

    summary = summarize_results(quality_rows, eval_rows)
    write_rows(args.outdir / "quality_rows.csv", quality_rows)
    write_rows(args.outdir / "eval_rows.csv", eval_rows)
    write_rows(args.outdir / "summary.csv", summary)
    eval_config = {
        "artifacts": {alpha_name(alpha): str(path) for alpha, path in artifact_paths.items()},
        "checkpoint": str(args.ckpt),
        "alphas": alphas,
        "attacks": list(attacks),
        "attack_seed": args.attack_seed,
        "topk": args.topk,
        "mask_threshold": args.mask_threshold,
        "clip_model": None if args.skip_clip else args.clip_model,
        "clip_device": None if args.skip_clip else str(clip_device),
        "clip_load_error": clip_error,
    }
    (args.outdir / "eval_config.json").write_text(
        json.dumps(eval_config, indent=2), encoding="utf-8"
    )
    for row in summary:
        print(row, flush=True)


def make_config(args: argparse.Namespace, alphas: list[float]) -> dict[str, Any]:
    return {
        "phase": args.phase,
        "num_samples": args.num_samples,
        "sample_offset": args.sample_offset,
        "seed": args.seed,
        "alphas": alphas,
        "size": args.size,
        "latent_size_assumption": [args.size // 8, args.size // 8],
        "frac_min": args.frac_min,
        "frac_max": args.frac_max,
        "max_per_category": args.max_per_category,
        "prompt_mode": args.prompt_mode,
        "cover_dir": str(args.cover_dir.resolve()),
        "instances_json": str(args.instances_json),
        "captions_json": str(args.captions_json),
        "model_root": str(args.model_root.resolve()),
        "checkpoint": str(args.ckpt.resolve()),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "data_bits": args.data_bits,
        "crc_bits": args.crc_bits,
        "chunk_bits": args.chunk_bits,
        "code_bits": args.code_bits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "build", "eval", "all"), default="all")
    parser.add_argument("--cover-dir", type=Path, default=DEFAULT_COVER_DIR)
    parser.add_argument("--instances-json", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--captions-json", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--model-root", type=Path, default=PATHS.sd21_model)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=PATHS.runs / "maskhead_ckpt" / "train2017_a128_classical_FINAL.pt",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--data-bits", type=int, default=32)
    parser.add_argument("--crc-bits", type=int, default=16)
    parser.add_argument("--chunk-bits", type=int, default=8)
    parser.add_argument("--code-bits", type=int, default=64)
    parser.add_argument("--alphas", default="0,128")
    parser.add_argument("--frac-min", type=float, default=0.05)
    parser.add_argument("--frac-max", type=float, default=0.35)
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--prompt-mode", choices=("mixed", "caption", "category"), default="mixed")
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--attack-seed", type=int, default=930001)
    parser.add_argument("--regen-strength", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--save-previews", type=int, default=4)
    args = parser.parse_args()
    if args.num_samples < 1 or args.sample_offset < 0:
        parser.error("--num-samples must be positive and --sample-offset nonnegative")
    if not (0.0 < args.frac_min < args.frac_max < 1.0):
        parser.error("Require 0 < frac-min < frac-max < 1")
    if args.size <= 0 or args.size % 8:
        parser.error("--size must be a positive multiple of 8")
    return args


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    for required in (args.cover_dir, args.instances_json, args.captions_json):
        if not required.exists():
            raise FileNotFoundError(required)

    if args.phase in {"prepare", "build", "all"}:
        if args.outdir.exists() and any(args.outdir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.outdir}")
        args.outdir.mkdir(parents=True, exist_ok=True)
        latent_height = args.size // 8
        latent_width = args.size // 8
        selected = select_instances(args, latent_height, latent_width)
        write_rows(args.outdir / "selection_manifest.csv", selection_rows(selected))
        config = make_config(args, alphas)
        (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(
            f"selected {len(selected)} instances; "
            f"prompt_sources={dict((s, sum(x.prompt_source == s for x in selected)) for s in ('caption', 'category'))}",
            flush=True,
        )
        if args.phase == "prepare":
            print("prepare complete", flush=True)
            return
    else:
        if not args.outdir.exists():
            raise FileNotFoundError(args.outdir)
        selected = []

    device = torch.device(args.device)
    pipe = load_pipe(args, device)
    if args.phase in {"build", "all"}:
        build_artifacts(args, selected, alphas, pipe, device)
    if args.phase in {"eval", "all"}:
        evaluate_artifacts(args, alphas, pipe, device)


if __name__ == "__main__":
    main()
