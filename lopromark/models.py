from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def make_dct_carrier(
    num_bits: int,
    size: int,
    *,
    min_freq: int = 1,
    max_freq: int = 16,
) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) + 0.5
    freq_pairs: list[tuple[int, int]] = []
    for total in range(min_freq, max_freq + 1):
        for u in range(total + 1):
            v = total - u
            if u == 0 and v == 0:
                continue
            freq_pairs.append((u, v))
    carriers = torch.zeros(num_bits, 3, size, size)
    for bit in range(num_bits):
        channel = bit % 3
        u, v = freq_pairs[(bit // 3) % len(freq_pairs)]
        basis_x = torch.cos(torch.pi * coords * u / size)
        basis_y = torch.cos(torch.pi * coords * v / size)
        carriers[bit, channel] = basis_y[:, None] * basis_x[None, :]
    carriers = carriers - carriers.mean(dim=(-2, -1), keepdim=True)
    carriers = carriers / carriers.flatten(1).std(dim=1).clamp_min(1e-6)[:, None, None, None]
    return carriers


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegionMessageAdapter(nn.Module):
    """Localized adapter applied to decoder-output RGB images.

    It learns a bounded residual conditioned on a spatial mask and rasterized
    region bits.
    """

    def __init__(self, num_bits: int = 32, width: int = 48, strength: float = 0.04) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.strength = strength
        in_channels = 3 + 1 + num_bits
        self.net = nn.Sequential(
            ConvBlock(in_channels, width),
            ConvBlock(width, width),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bit_targets.shape[1] != self.num_bits:
            raise ValueError(f"Expected {self.num_bits} bits, got {bit_targets.shape[1]}")
        cond_bits = (bit_targets * 2.0 - 1.0) * union_mask
        x = torch.cat([image, union_mask, cond_bits], dim=1)
        residual = torch.tanh(self.net(x)) * self.strength * union_mask
        watermarked = (image + residual).clamp(0.0, 1.0)
        return watermarked, residual


class SpreadSpectrumAdapter(nn.Module):
    """Keyed fixed carrier adapter for higher-capacity localized payloads."""

    def __init__(
        self,
        num_bits: int = 64,
        strength: float = 0.08,
        carrier_size: int = 32,
        seed: int = 20260521,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.strength = strength
        self.carrier_size = carrier_size
        self.seed = seed
        self._cache: dict[tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}

    def carrier_bank(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype, height, width)
        if key in self._cache:
            return self._cache[key]
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + self.num_bits * 1009 + height * 9176 + width * 6113
        )
        low = torch.randn(self.num_bits, 3, self.carrier_size, self.carrier_size, generator=generator)
        low = low - low.mean(dim=(-2, -1), keepdim=True)
        low = low / low.flatten(1).std(dim=1).clamp_min(1e-6)[:, None, None, None]
        carrier = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
        carrier = carrier - carrier.mean(dim=(-2, -1), keepdim=True)
        carrier = carrier / carrier.flatten(1).std(dim=1).clamp_min(1e-6)[:, None, None, None]
        carrier = carrier.to(device=device, dtype=dtype)
        self._cache[key] = carrier
        return carrier

    def forward(
        self,
        image: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bit_targets.shape[1] != self.num_bits:
            raise ValueError(f"Expected {self.num_bits} bits, got {bit_targets.shape[1]}")
        _, _, height, width = image.shape
        carrier = self.carrier_bank(height, width, image.device, image.dtype)
        bit_signs = (bit_targets * 2.0 - 1.0) * union_mask
        residual = torch.einsum("bkhw,kchw->bchw", bit_signs, carrier)
        residual = residual * (self.strength / (self.num_bits**0.5))
        residual = residual * union_mask
        watermarked = (image + residual).clamp(0.0, 1.0)
        return watermarked, residual


class RegionNormalizedCarrierAdapter(nn.Module):
    """Canonical region-coordinate carrier for crop/resize synchronized payloads."""

    def __init__(
        self,
        num_bits: int = 64,
        strength: float = 0.08,
        carrier_size: int = 32,
        carrier_base_size: int = 12,
        carrier_kind: str = "random",
        dct_min_freq: int = 1,
        dct_max_freq: int = 16,
        seed: int = 20260521,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.strength = strength
        self.carrier_size = carrier_size
        self.carrier_base_size = carrier_base_size
        self.carrier_kind = carrier_kind
        self.dct_min_freq = dct_min_freq
        self.dct_max_freq = dct_max_freq
        self.seed = seed
        if carrier_kind == "dct":
            carrier = make_dct_carrier(
                num_bits,
                carrier_size,
                min_freq=dct_min_freq,
                max_freq=dct_max_freq,
            )
        elif carrier_kind == "random":
            generator = torch.Generator(device="cpu").manual_seed(
                seed + num_bits * 7919 + carrier_size * 101 + carrier_base_size
            )
            carrier = torch.randn(num_bits, 3, carrier_base_size, carrier_base_size, generator=generator)
            carrier = F.interpolate(
                carrier,
                size=(carrier_size, carrier_size),
                mode="bilinear",
                align_corners=False,
            )
            carrier = carrier - carrier.mean(dim=(-2, -1), keepdim=True)
            carrier = carrier / carrier.flatten(1).std(dim=1).clamp_min(1e-6)[:, None, None, None]
        else:
            raise ValueError(f"Unknown carrier_kind: {carrier_kind}")
        self.register_buffer("canonical_carrier", carrier)

    def forward(
        self,
        image: torch.Tensor,
        region_masks: torch.Tensor,
        messages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if messages.shape[-1] != self.num_bits:
            raise ValueError(f"Expected {self.num_bits} bits, got {messages.shape[-1]}")
        residual = torch.zeros_like(image)
        batch, num_regions, _, _ = region_masks.shape
        carrier = self.canonical_carrier.to(device=image.device, dtype=image.dtype)
        for b in range(batch):
            for r in range(num_regions):
                coords = torch.nonzero(region_masks[b, r] > 0.5, as_tuple=False)
                if coords.numel() == 0:
                    continue
                top = int(coords[:, 0].min().item())
                bottom = int(coords[:, 0].max().item()) + 1
                left = int(coords[:, 1].min().item())
                right = int(coords[:, 1].max().item()) + 1
                bit_sign = messages[b, r].to(dtype=image.dtype) * 2.0 - 1.0
                canonical = torch.einsum("k,kchw->chw", bit_sign, carrier)
                canonical = canonical / (self.num_bits**0.5)
                patch = F.interpolate(
                    canonical[None],
                    size=(bottom - top, right - left),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                mask_patch = region_masks[b, r, top:bottom, left:right].to(dtype=image.dtype)
                residual[b, :, top:bottom, left:right] += patch * mask_patch[None]
        residual = residual * self.strength
        watermarked = (image + residual).clamp(0.0, 1.0)
        return watermarked, residual


class LocalizedExtractor(nn.Module):
    """Lightweight U-Net style extractor for mask and message logits."""

    def __init__(self, num_bits: int = 32, width: int = 48) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.mid = ConvBlock(width * 4, width * 4)
        self.dec2 = ConvBlock(width * 4 + width * 2, width * 2)
        self.dec1 = ConvBlock(width * 2 + width, width)
        self.out = nn.Conv2d(width, 1 + num_bits, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(image)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self.out(d1)
        return logits[:, :1], logits[:, 1:]


class CarrierAwareExtractor(nn.Module):
    """Extractor that receives keyed carrier correlations as an explicit prior."""

    def __init__(
        self,
        num_bits: int = 64,
        width: int = 64,
        carrier_size: int = 32,
        carrier_seed: int = 20260521,
    ) -> None:
        super().__init__()
        self.adapter = SpreadSpectrumAdapter(
            num_bits=num_bits,
            strength=1.0,
            carrier_size=carrier_size,
            seed=carrier_seed,
        )
        self.enc1 = ConvBlock(3 + num_bits, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.mid = ConvBlock(width * 4, width * 4)
        self.dec2 = ConvBlock(width * 4 + width * 2, width * 2)
        self.dec1 = ConvBlock(width * 2 + width, width)
        self.out = nn.Conv2d(width, 1 + num_bits, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = image.shape
        carrier = self.adapter.carrier_bank(height, width, image.device, image.dtype)
        centered = image - F.avg_pool2d(image, kernel_size=15, stride=1, padding=7)
        correlations = torch.einsum("bchw,kchw->bkhw", centered, carrier)
        x = torch.cat([image, correlations], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self.out(d1)
        return logits[:, :1], logits[:, 1:]
