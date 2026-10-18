from __future__ import annotations

import io
import random

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.nn import functional as F


def jpeg_roundtrip(images: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Apply a PIL JPEG roundtrip to a [B, C, H, W] tensor in [0, 1]."""
    outputs = []
    for image in images.detach().cpu():
        pil = to_pil_image(image.clamp(0, 1))
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        restored = Image.open(buffer).convert("RGB")
        outputs.append(pil_to_tensor(restored).float() / 255.0)
    return torch.stack(outputs, dim=0).to(images.device)


def resize_roundtrip(images: torch.Tensor, scale: float = 0.75) -> torch.Tensor:
    height, width = images.shape[-2:]
    small_h = max(16, int(height * scale))
    small_w = max(16, int(width * scale))
    small = F.interpolate(images, size=(small_h, small_w), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)


def crop_resize(images: torch.Tensor, crop_frac: float = 0.85) -> torch.Tensor:
    batch, _, height, width = images.shape
    crop_h = max(16, int(height * crop_frac))
    crop_w = max(16, int(width * crop_frac))
    crops = []
    for b in range(batch):
        top = random.randint(0, max(0, height - crop_h))
        left = random.randint(0, max(0, width - crop_w))
        crop = images[b : b + 1, :, top : top + crop_h, left : left + crop_w]
        crops.append(F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False))
    return torch.cat(crops, dim=0)


def crop_resize_bundle(
    images: torch.Tensor,
    region_masks: torch.Tensor,
    union_mask: torch.Tensor,
    bit_targets: torch.Tensor,
    crop_frac: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same crop-resize geometry to images and spatial labels."""
    batch, _, height, width = images.shape
    crop_h = max(16, int(height * crop_frac))
    crop_w = max(16, int(width * crop_frac))
    image_crops = []
    region_crops = []
    union_crops = []
    bit_crops = []
    for b in range(batch):
        top = random.randint(0, max(0, height - crop_h))
        left = random.randint(0, max(0, width - crop_w))
        image_crop = images[b : b + 1, :, top : top + crop_h, left : left + crop_w]
        region_crop = region_masks[b : b + 1, :, top : top + crop_h, left : left + crop_w]
        union_crop = union_mask[b : b + 1, :, top : top + crop_h, left : left + crop_w]
        bit_crop = bit_targets[b : b + 1, :, top : top + crop_h, left : left + crop_w]
        image_crops.append(
            F.interpolate(image_crop, size=(height, width), mode="bilinear", align_corners=False)
        )
        region_crops.append(F.interpolate(region_crop, size=(height, width), mode="nearest"))
        union_crops.append(F.interpolate(union_crop, size=(height, width), mode="nearest"))
        bit_crops.append(F.interpolate(bit_crop, size=(height, width), mode="nearest"))
    return (
        torch.cat(image_crops, dim=0),
        torch.cat(region_crops, dim=0),
        torch.cat(union_crops, dim=0),
        torch.cat(bit_crops, dim=0),
    )


def blur(images: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    return F.avg_pool2d(images, kernel_size=kernel_size, stride=1, padding=padding)


def gaussian_noise(images: torch.Tensor, sigma: float = 0.015) -> torch.Tensor:
    return (images + torch.randn_like(images) * sigma).clamp(0.0, 1.0)


def brightness(images: torch.Tensor, factor: float = 1.12) -> torch.Tensor:
    return (images * factor).clamp(0.0, 1.0)


def apply_attack(images: torch.Tensor, name: str) -> torch.Tensor:
    name = name.lower()
    if name in {"none", "identity", "clean"}:
        return images
    if name == "jpeg":
        return jpeg_roundtrip(images, quality=75)
    if name == "resize":
        return resize_roundtrip(images, scale=0.75)
    if name == "crop_resize":
        return crop_resize(images, crop_frac=0.85)
    if name == "blur":
        return blur(images, kernel_size=5)
    if name == "noise":
        return gaussian_noise(images, sigma=0.015)
    if name == "brightness":
        return brightness(images, factor=1.12)
    raise ValueError(f"Unknown attack: {name}")


def apply_attack_bundle(
    images: torch.Tensor,
    region_masks: torch.Tensor,
    union_mask: torch.Tensor,
    bit_targets: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an attack and keep spatial labels aligned for geometric transforms."""
    name = name.lower()
    if name == "crop_resize":
        return crop_resize_bundle(images, region_masks, union_mask, bit_targets)
    return apply_attack(images, name), region_masks, union_mask, bit_targets
