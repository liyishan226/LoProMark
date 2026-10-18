from __future__ import annotations

import math

import torch

from .messages import decode_region_messages


@torch.no_grad()
def psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = torch.mean((reference - estimate) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def mask_iou(mask_logits: torch.Tensor, target_mask: torch.Tensor) -> float:
    pred = (mask_logits.sigmoid() >= 0.5).float()
    target = (target_mask >= 0.5).float()
    intersection = (pred * target).sum()
    union = ((pred + target) > 0).float().sum().clamp_min(1.0)
    return float((intersection / union).cpu())


@torch.no_grad()
def pixel_bit_accuracy(
    bit_logits: torch.Tensor,
    bit_targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    pred = (bit_logits.sigmoid() >= 0.5).float()
    correct = ((pred == bit_targets).float() * valid_mask).sum()
    total = valid_mask.sum().clamp_min(1.0) * bit_logits.shape[1]
    return float((correct / total).cpu())


@torch.no_grad()
def region_message_accuracy(
    bit_logits: torch.Tensor,
    region_masks: torch.Tensor,
    messages: torch.Tensor,
) -> tuple[float, float]:
    decoded = decode_region_messages(bit_logits, region_masks)
    bit_acc = (decoded == messages).float().mean()
    exact = (decoded == messages).all(dim=-1).float().mean()
    return float(bit_acc.cpu()), float(exact.cpu())
