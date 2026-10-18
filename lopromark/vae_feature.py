from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch import nn
from torch.nn import functional as F

from .models import LocalizedExtractor


class FeatureConditioner(nn.Module):
    """Inject mask/message-conditioned residuals into VAE decoder feature maps."""

    def __init__(
        self,
        num_bits: int,
        channels: list[int],
        width: int = 64,
        strength: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.strength = strength
        self.blocks = nn.ModuleList()
        in_channels = 1 + num_bits
        for channel in channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, channel, 1),
            )
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            self.blocks.append(block)

    def forward(
        self,
        index: int,
        hidden_states: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> torch.Tensor:
        cond_bits = (bit_targets * 2.0 - 1.0) * union_mask
        cond = torch.cat([union_mask, cond_bits], dim=1)
        cond = F.interpolate(cond, size=hidden_states.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.blocks[index](cond).to(dtype=hidden_states.dtype)
        return hidden_states + torch.tanh(residual) * self.strength


class VAEFeatureWatermarker(nn.Module):
    """Frozen SD VAE with trainable payload-conditioned decoder feature injection."""

    def __init__(
        self,
        vae_root: str | Path,
        num_bits: int = 32,
        conditioner_width: int = 64,
        extractor_width: int = 64,
        strength: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.vae = AutoencoderKL.from_pretrained(
            str(vae_root),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        channels = [block.resnets[-1].out_channels for block in self.vae.decoder.up_blocks]
        self.conditioner = FeatureConditioner(
            num_bits=num_bits,
            channels=channels,
            width=conditioner_width,
            strength=strength,
        )
        self.extractor = LocalizedExtractor(num_bits=num_bits, width=extractor_width)

    @property
    def scaling_factor(self) -> float:
        return float(self.vae.config.scaling_factor)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        normalized = images * 2.0 - 1.0
        posterior = self.vae.encode(normalized).latent_dist
        if hasattr(posterior, "mode"):
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        return latents * self.scaling_factor

    @staticmethod
    def _decoder_dtype(decoder: nn.Module, fallback: torch.dtype) -> torch.dtype:
        for module_name in ("conv_norm_out", "conv_out"):
            module = getattr(decoder, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters(recurse=True):
                return parameter.dtype
        return fallback

    def decode_with_features(
        self,
        latents: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> torch.Tensor:
        z = latents / self.scaling_factor
        if self.vae.post_quant_conv is not None:
            z = self.vae.post_quant_conv(z)

        decoder = self.vae.decoder
        sample = decoder.conv_in(z)
        sample = decoder.mid_block(sample, None)
        sample = sample.to(self._decoder_dtype(decoder, sample.dtype))
        for index, up_block in enumerate(decoder.up_blocks):
            sample = up_block(sample, None)
            sample = self.conditioner(index, sample, union_mask, bit_targets)
        sample = decoder.conv_norm_out(sample)
        sample = decoder.conv_act(sample)
        sample = decoder.conv_out(sample)
        return (sample / 2.0 + 0.5).clamp(0.0, 1.0)

    def decode_plain(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents / self.scaling_factor
        if self.vae.post_quant_conv is not None:
            z = self.vae.post_quant_conv(z)

        decoder = self.vae.decoder
        sample = decoder.conv_in(z)
        sample = decoder.mid_block(sample, None)
        sample = sample.to(self._decoder_dtype(decoder, sample.dtype))
        for up_block in decoder.up_blocks:
            sample = up_block(sample, None)
        sample = decoder.conv_norm_out(sample)
        sample = decoder.conv_act(sample)
        sample = decoder.conv_out(sample)
        return (sample / 2.0 + 0.5).clamp(0.0, 1.0)

    def embed(
        self,
        cover: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode_images(cover)
        baseline = self.decode_plain(latents)
        watermarked = self.decode_with_features(latents, union_mask, bit_targets)
        return baseline, watermarked

    def forward(
        self,
        cover: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline, watermarked = self.embed(cover, union_mask, bit_targets)
        mask_logits, bit_logits = self.extractor(watermarked)
        return baseline, watermarked, mask_logits, bit_logits


class FeatureCarrierInjector(nn.Module):
    """Fixed spread-spectrum carriers injected into VAE decoder feature maps."""

    def __init__(
        self,
        num_bits: int,
        channels: list[int],
        strength: float = 0.05,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.strength = strength
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for index, channel in enumerate(channels):
            carrier = torch.randn(num_bits, channel, generator=generator)
            carrier = F.normalize(carrier, dim=1)
            self.register_buffer(f"carrier_{index}", carrier)

    def forward(
        self,
        index: int,
        hidden_states: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> torch.Tensor:
        cond_bits = (bit_targets * 2.0 - 1.0) * union_mask
        cond_bits = F.interpolate(
            cond_bits,
            size=hidden_states.shape[-2:],
            mode="nearest",
        )
        mask = F.interpolate(
            union_mask,
            size=hidden_states.shape[-2:],
            mode="nearest",
        )
        carrier = getattr(self, f"carrier_{index}").to(dtype=hidden_states.dtype)
        residual = torch.einsum("bkhw,kc->bchw", cond_bits, carrier)
        residual = residual * mask
        return hidden_states + residual * self.strength


class VAEFeatureCarrierWatermarker(nn.Module):
    """Frozen SD VAE with deterministic payload carriers inside decoder features."""

    def __init__(
        self,
        vae_root: str | Path,
        num_bits: int = 32,
        extractor_width: int = 64,
        strength: float = 0.05,
        carrier_seed: int = 2026,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.vae = AutoencoderKL.from_pretrained(
            str(vae_root),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        channels = [block.resnets[-1].out_channels for block in self.vae.decoder.up_blocks]
        self.injector = FeatureCarrierInjector(
            num_bits=num_bits,
            channels=channels,
            strength=strength,
            seed=carrier_seed,
        )
        self.extractor = LocalizedExtractor(num_bits=num_bits, width=extractor_width)

    @property
    def scaling_factor(self) -> float:
        return float(self.vae.config.scaling_factor)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        normalized = images * 2.0 - 1.0
        posterior = self.vae.encode(normalized).latent_dist
        if hasattr(posterior, "mode"):
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        return latents * self.scaling_factor

    @staticmethod
    def _decoder_dtype(decoder: nn.Module, fallback: torch.dtype) -> torch.dtype:
        return VAEFeatureWatermarker._decoder_dtype(decoder, fallback)

    def decode_plain(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents / self.scaling_factor
        if self.vae.post_quant_conv is not None:
            z = self.vae.post_quant_conv(z)

        decoder = self.vae.decoder
        sample = decoder.conv_in(z)
        sample = decoder.mid_block(sample, None)
        sample = sample.to(self._decoder_dtype(decoder, sample.dtype))
        for up_block in decoder.up_blocks:
            sample = up_block(sample, None)
        sample = decoder.conv_norm_out(sample)
        sample = decoder.conv_act(sample)
        sample = decoder.conv_out(sample)
        return (sample / 2.0 + 0.5).clamp(0.0, 1.0)

    def decode_with_carriers(
        self,
        latents: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> torch.Tensor:
        z = latents / self.scaling_factor
        if self.vae.post_quant_conv is not None:
            z = self.vae.post_quant_conv(z)

        decoder = self.vae.decoder
        sample = decoder.conv_in(z)
        sample = decoder.mid_block(sample, None)
        sample = sample.to(self._decoder_dtype(decoder, sample.dtype))
        for index, up_block in enumerate(decoder.up_blocks):
            sample = up_block(sample, None)
            sample = self.injector(index, sample, union_mask, bit_targets)
        sample = decoder.conv_norm_out(sample)
        sample = decoder.conv_act(sample)
        sample = decoder.conv_out(sample)
        return (sample / 2.0 + 0.5).clamp(0.0, 1.0)

    def embed(
        self,
        cover: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode_images(cover)
        baseline = self.decode_plain(latents)
        watermarked = self.decode_with_carriers(latents, union_mask, bit_targets)
        return baseline, watermarked

    def forward(
        self,
        cover: torch.Tensor,
        union_mask: torch.Tensor,
        bit_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline, watermarked = self.embed(cover, union_mask, bit_targets)
        mask_logits, bit_logits = self.extractor(watermarked)
        return baseline, watermarked, mask_logits, bit_logits
