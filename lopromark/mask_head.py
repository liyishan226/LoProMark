"""Learned mask-recovery head for localized diffusion-watermark provenance.

Given the DDIM-recovered latent of a partially-edited (inpainted) image plus
multi-scale carrier-margin feature maps, predict the edited region mask (latent
resolution). Payload is then decoded over the predicted mask. A compact U-Net
refines the multi-scale margin evidence into the region mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
    )


class MaskHead(nn.Module):
    """Compact U-Net: [B, in_ch, H, W] -> [B, 1, H, W] mask logits."""

    def __init__(self, in_ch: int = 7, base: int = 32):
        super().__init__()
        self.enc1 = _block(in_ch, base)
        self.enc2 = _block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bott = _block(base * 2, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)              # H
        e2 = self.enc2(self.pool(e1))  # H/2
        b = self.bott(self.pool(e2))   # H/4
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))  # H/2
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # H
        return self.head(d1)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()


def tversky_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                     alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    """Tversky with beta>alpha penalizes false negatives -> recall-favoring coverage,
    which helps downstream payload decode (mask must cover the carrier support)."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=(1, 2, 3))
    fp = (p * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * target).sum(dim=(1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return bce + (1.0 - tversky).mean()


def dilate_mask(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Morphological dilation via max-pool; widens coverage for decode robustness.
    Crops back to the input spatial size so even kernels stay shape-safe."""
    if k <= 1:
        return mask
    out = nn.functional.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    return out[..., : mask.shape[-2], : mask.shape[-1]]


@torch.no_grad()
def normalize_input(x: torch.Tensor) -> torch.Tensor:
    """Per-channel standardize over spatial dims, per sample."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std
