"""Build and cache watermarked composites once per cover.

Caches the clean composite, edit mask, and payload so the per-attack
evaluator can re-invert each composite under every attack without
regenerating.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.messages import sample_binary_messages


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "carrier_core")
M = _load("scripts/blended_inpaint.py", "blended_inpaint")
C = _load("scripts/mask_gate.py", "mask_gate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=PATHS.sd21_model)
    ap.add_argument("--cover-dir", type=Path, default=ROOT / "data" / "val2017")
    ap.add_argument("--out", type=Path, default=PATHS.runs / "composites" / "pos.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--cover-offset", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--prompt", default="a natural photo, realistic, detailed")
    ap.add_argument("--payload-bits", type=int, default=64)
    ap.add_argument("--codebook-chunk-bits", type=int, default=8)
    ap.add_argument("--codebook-code-bits", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=48.0)
    ap.add_argument("--frac-min", type=float, default=0.05)
    ap.add_argument("--frac-max", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=7001)
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed + args.cover_offset)
    torch.manual_seed(args.seed + args.cover_offset)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_root), local_files_only=True, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    H = args.size // pipe.vae_scale_factor
    W = args.size // pipe.vae_scale_factor
    chunk = args.codebook_chunk_bits
    outer_bits = args.payload_bits // chunk * args.codebook_code_bits
    codebook = P.make_binary_codebook(chunk, args.codebook_code_bits, args.seed + 571, device)
    carriers = P.make_carriers(outer_bits, pipe.unet.config.in_channels, H, W, args.seed + 17, device)
    covers = sorted(args.cover_dir.glob("*.jpg"))

    images, masks, payloads = [], [], []
    for i in range(args.num_samples):
        cover = M.load_cover(covers[(args.cover_offset + i) % len(covers)], args.size, device)
        z_real = P.encode_image_to_latent(pipe, cover.to(torch.float16)).float()
        frac = random.uniform(args.frac_min, args.frac_max)
        edit_mask = C.irregular_mask(frac, H, W, device)
        payload = sample_binary_messages(1, 1, args.payload_bits, device=device)
        outer = P.encode_payload(payload, "codebook", codebook, chunk)
        base = torch.randn(1, pipe.unet.config.in_channels, H, W, device=device).float()
        wm = P.add_local_noise_watermark(base, edit_mask, outer, carriers, args.alpha, normalize=False)
        wm = P.match_local_region_stats(wm, base, edit_mask)
        wm = P.normalize_latent_noise(wm)
        eps = torch.randn_like(z_real)
        init_sigma = pipe.scheduler.init_noise_sigma
        pipe.scheduler.set_timesteps(args.steps, device=device)

        def make_cb(em, zr, e):
            def cb(pipe_, step, timestep, kwargs):
                lat = kwargs["latents"].float()
                ts = pipe_.scheduler.timesteps
                bg = pipe_.scheduler.add_noise(zr, e, ts[step + 1].unsqueeze(0)) if step + 1 < len(ts) else zr
                return {"latents": (em * lat + (1.0 - em) * bg).to(kwargs["latents"].dtype)}
            return cb

        x_T = edit_mask * (wm * init_sigma) + (1.0 - edit_mask) * pipe.scheduler.add_noise(
            z_real, eps, pipe.scheduler.timesteps[0].unsqueeze(0))
        composite = P.tensor_to_bchw(pipe(
            prompt=[args.prompt], height=args.size, width=args.size,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            latents=x_T.to(torch.float16), output_type="pt",
            callback_on_step_end=make_cb(edit_mask, z_real, eps),
            callback_on_step_end_tensor_inputs=["latents"]).images)

        images.append((composite[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu())  # [3,H,W] uint8
        masks.append(edit_mask[0].bool().cpu())
        payloads.append(payload[0, 0].bool().cpu())
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{args.num_samples}]", flush=True)

    torch.save(dict(
        images=torch.stack(images), masks=torch.stack(masks), payloads=torch.stack(payloads),
        meta=dict(seed=args.seed, payload_bits=args.payload_bits, chunk=chunk,
                  code_bits=args.codebook_code_bits, alpha=args.alpha, size=args.size,
                  H=H, W=W, steps=args.steps, guidance_scale=args.guidance_scale, prompt=args.prompt),
    ), args.out)
    print(f"saved {args.out}  images={tuple(torch.stack(images).shape)}")


if __name__ == "__main__":
    main()
