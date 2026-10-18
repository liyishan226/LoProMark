from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_bit_bce(
    bit_logits: torch.Tensor,
    bit_targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(bit_logits, bit_targets, reduction="none")
    weighted = per_pixel * valid_mask
    return weighted.sum() / valid_mask.sum().clamp_min(1.0) / bit_logits.shape[1]


def region_message_bce(
    bit_logits: torch.Tensor,
    region_masks: torch.Tensor,
    messages: torch.Tensor,
) -> torch.Tensor:
    """Pool bit logits inside each region and supervise region-level messages."""
    masks = region_masks.float()
    denom = masks.flatten(2).sum(dim=-1).clamp_min(1.0)
    pooled_logits = torch.einsum("bkhw,brhw->brk", bit_logits, masks) / denom[:, :, None]
    return F.binary_cross_entropy_with_logits(pooled_logits, messages)


def balanced_mask_bce(mask_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(mask_logits, target_mask, reduction="none")
    positive = target_mask
    negative = 1.0 - target_mask
    positive_loss = (per_pixel * positive).sum() / positive.sum().clamp_min(1.0)
    negative_loss = (per_pixel * negative).sum() / negative.sum().clamp_min(1.0)
    return 0.5 * (positive_loss + negative_loss)


def training_loss(
    cover: torch.Tensor,
    watermarked: torch.Tensor,
    mask_logits: torch.Tensor,
    bit_logits: torch.Tensor,
    union_mask: torch.Tensor,
    bit_targets: torch.Tensor,
    *,
    lambda_image: float = 2.0,
    lambda_mask: float = 1.0,
    lambda_bits: float = 5.0,
    region_masks: torch.Tensor | None = None,
    messages: torch.Tensor | None = None,
    lambda_region_bits: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_loss = F.mse_loss(watermarked, cover)
    mask_loss = nn.BCEWithLogitsLoss()(mask_logits, union_mask)
    bits_loss = masked_bit_bce(bit_logits, bit_targets, union_mask)
    region_bits_loss = torch.zeros((), device=cover.device)
    if region_masks is not None and messages is not None and lambda_region_bits > 0:
        region_bits_loss = region_message_bce(bit_logits, region_masks, messages)
    total = (
        lambda_image * image_loss
        + lambda_mask * mask_loss
        + lambda_bits * bits_loss
        + lambda_region_bits * region_bits_loss
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "image": float(image_loss.detach().cpu()),
        "mask": float(mask_loss.detach().cpu()),
        "bits": float(bits_loss.detach().cpu()),
        "region_bits": float(region_bits_loss.detach().cpu()),
    }
    return total, stats
