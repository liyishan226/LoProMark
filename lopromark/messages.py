from __future__ import annotations

import torch


def sample_binary_messages(
    batch_size: int,
    num_regions: int,
    num_bits: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample region-level binary messages with shape [B, R, K]."""
    return torch.randint(
        low=0,
        high=2,
        size=(batch_size, num_regions, num_bits),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )


def build_bit_targets(
    region_masks: torch.Tensor,
    messages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize per-region messages into pixel-level bit targets.

    Args:
        region_masks: Float/bool tensor [B, R, H, W].
        messages: Float tensor [B, R, K] with values in {0, 1}.

    Returns:
        union_mask: [B, 1, H, W].
        bit_targets: [B, K, H, W].
        valid_mask: [B, 1, H, W].
    """
    if region_masks.ndim != 4:
        raise ValueError(f"region_masks must be [B, R, H, W], got {region_masks.shape}")
    if messages.ndim != 3:
        raise ValueError(f"messages must be [B, R, K], got {messages.shape}")
    if region_masks.shape[:2] != messages.shape[:2]:
        raise ValueError("region_masks and messages disagree on batch/region axes")

    masks = region_masks.float().clamp(0, 1)
    weighted_bits = masks.unsqueeze(2) * messages[:, :, :, None, None]
    mask_sum = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
    bit_targets = weighted_bits.sum(dim=1) / mask_sum
    union_mask = (masks.sum(dim=1, keepdim=True) > 0).float()
    return union_mask, bit_targets, union_mask


@torch.no_grad()
def decode_region_messages(bit_logits: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
    """Average pixel logits inside each region and threshold them to bits."""
    if bit_logits.ndim != 4:
        raise ValueError(f"bit_logits must be [B, K, H, W], got {bit_logits.shape}")
    if region_masks.ndim != 4:
        raise ValueError(f"region_masks must be [B, R, H, W], got {region_masks.shape}")

    probs = bit_logits.sigmoid()
    masks = region_masks.float()
    denom = masks.flatten(2).sum(dim=-1).clamp_min(1.0)
    pooled = torch.einsum("bkhw,brhw->brk", probs, masks) / denom[:, :, None]
    return (pooled >= 0.5).float()
