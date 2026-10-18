"""Build and cache the mask-head training dataset.

Per sample: real cover, random irregular mask, carrier embedding,
blended-latent generation, classical attack, DDIM inversion, cached
input features plus true mask. Also computes the four codebook-margin
maps.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.messages import sample_binary_messages


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "carrier_core")
K = _load("scripts/decode_helpers.py", "decode_helpers")
M = _load("scripts/blended_inpaint.py", "blended_inpaint")
C = _load("scripts/mask_gate.py", "mask_gate")

# Classical value-domain augmentations only. diffusion_regen and vae are not used
# for training.
DEFAULT_ATTACK_POOL = [
    "identity", "jpeg_q40", "jpeg_q60", "jpeg_q80", "resize_s0.5", "resize_s0.75",
    "gaussian_blur_k3", "gaussian_blur_k7", "gaussian_noise_s0.05", "brightness_f1.5",
    "contrast_f1.5", "hue_h0.25", "saturation_f1.5", "median_k5",
]


def margin_maps(recovered, carriers, codebook, H, W, wins, stride) -> torch.Tensor:
    """Multi-scale sliding-window codebook-margin maps -> [len(wins), H, W]."""
    out = []
    for win in wins:
        acc = torch.zeros((H, W), device=recovered.device)
        tops = sorted(set(list(range(0, H - win + 1, stride)) + [H - win]))
        lefts = sorted(set(list(range(0, W - win + 1, stride)) + [W - win]))
        for t in tops:
            for l in lefts:
                m = K.rect_mask(t, l, win, win, H, W, recovered.device)
                s = K.detect_score_at(recovered, m, carriers, codebook)
                blk = acc[t : t + win, l : l + win]
                acc[t : t + win, l : l + win] = torch.maximum(blk, torch.full_like(blk, s))
        out.append(acc)
    return torch.stack(out, dim=0)


def margin_maps_fast(recovered, carriers, codebook, H, W, wins, stride) -> torch.Tensor:
    """Same output as margin_maps, computed on window crops for speed."""
    centered = P.normalize_latent_noise(recovered)  # full-tensor centering, once
    out = []
    for win in wins:
        acc = torch.zeros((H, W), device=recovered.device)
        tops = sorted(set(list(range(0, H - win + 1, stride)) + [H - win]))
        lefts = sorted(set(list(range(0, W - win + 1, stride)) + [W - win]))
        for t in tops:
            for l in lefts:
                c = centered[:, :, t : t + win, l : l + win]      # [1,4,win,win]
                k = carriers[:, :, t : t + win, l : l + win]      # [code_bits,4,win,win]
                norm = k.flatten(1).norm(dim=1).clamp_min(1e-6)   # [code_bits]
                corr = torch.einsum("bchw,kchw->bk", c, k) / norm[None]  # [1,code_bits]
                s = float(P.codebook_likelihood_margin(corr[:, None, :], codebook).mean())
                blk = acc[t : t + win, l : l + win]
                acc[t : t + win, l : l + win] = torch.maximum(blk, torch.full_like(blk, s))
        out.append(acc)
    return torch.stack(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=PATHS.sd21_model)
    ap.add_argument("--sdxl-model", type=Path, default=PATHS.sdxl_model)
    ap.add_argument(
        "--sd15-model",
        type=Path,
        default=ROOT / "weights" / "stable-diffusion-v1-5",
    )
    ap.add_argument("--cover-dir", type=Path, default=ROOT / "data" / "val2017")
    ap.add_argument("--out", type=Path, default=PATHS.runs / "maskhead_dataset" / "shard.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--cover-offset", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--prompt", default="a natural photo, realistic, detailed")
    ap.add_argument(
        "--object-masks", action="store_true",
        help=(
            "Use unique-image COCO polygon instances and matched caption/category "
            "prompts instead of random irregular masks."
        ),
    )
    ap.add_argument("--instances-json", type=Path)
    ap.add_argument("--captions-json", type=Path)
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--max-per-category", type=int, default=0)
    ap.add_argument(
        "--prompt-mode", choices=("mixed", "caption", "category"), default="mixed",
    )
    ap.add_argument("--payload-bits", type=int, default=64)
    ap.add_argument("--codebook-chunk-bits", type=int, default=8)
    ap.add_argument("--codebook-code-bits", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=48.0)
    ap.add_argument("--frac-min", type=float, default=0.05)
    ap.add_argument("--frac-max", type=float, default=0.35)
    ap.add_argument("--margin-wins", default="8,14,22")
    ap.add_argument("--margin-stride", type=int, default=3)
    ap.add_argument(
        "--attack-pool",
        default=",".join(DEFAULT_ATTACK_POOL),
        help=(
            "Comma-separated training attacks. The default uses classical attacks "
            "only. Regeneration entries such as diffusion_regen_s0.1 can be added."
        ),
    )
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    attack_pool = [item.strip() for item in args.attack_pool.split(",") if item.strip()]
    if not attack_pool:
        raise ValueError("--attack-pool must contain at least one attack")

    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed + args.cover_offset)
    torch.manual_seed(args.seed + args.cover_offset)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_root), local_files_only=True, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    cross_pipes = {}

    def cross_regen_attack(family: str, image: torch.Tensor, prompt: str,
                           strength: float, sample_seed: int) -> torch.Tensor:
        """Downstream img2img regeneration with a backbone different from the
        SD2.1 embedding/verifier. Pipelines are loaded lazily so a shard can target
        one family without paying the memory cost of the other."""
        if family not in cross_pipes:
            if family == "sdxl":
                from diffusers import StableDiffusionXLImg2ImgPipeline

                attack_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    str(args.sdxl_model), local_files_only=True,
                    torch_dtype=torch.float16,
                ).to(device)
            elif family == "sd15":
                from diffusers import StableDiffusionImg2ImgPipeline

                attack_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    str(args.sd15_model), local_files_only=True,
                    torch_dtype=torch.float16, safety_checker=None,
                    requires_safety_checker=False,
                ).to(device)
            else:
                raise ValueError(f"unsupported cross-regenerator family: {family}")
            attack_pipe.set_progress_bar_config(disable=True)
            cross_pipes[family] = attack_pipe
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        regenerated = cross_pipes[family](
            prompt=prompt, image=image.float(), strength=strength,
            guidance_scale=args.guidance_scale, num_inference_steps=args.steps,
            generator=generator, output_type="pt",
        ).images
        return torch.nn.functional.interpolate(
            regenerated.float(), size=(args.size, args.size), mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

    H = args.size // pipe.vae_scale_factor
    W = args.size // pipe.vae_scale_factor
    chunk = args.codebook_chunk_bits
    outer_bits = args.payload_bits // chunk * args.codebook_code_bits
    codebook = P.make_binary_codebook(chunk, args.codebook_code_bits, args.seed + 571, device)
    carriers = P.make_carriers(outer_bits, pipe.unet.config.in_channels, H, W, args.seed + 17, device)
    wins = [int(x) for x in args.margin_wins.split(",") if x.strip()]
    covers = sorted(args.cover_dir.glob("*.jpg"))
    selected_objects = None
    if args.object_masks:
        if args.instances_json is None or args.captions_json is None:
            raise ValueError("--object-masks requires --instances-json and --captions-json")
        if not args.instances_json.exists() or not args.captions_json.exists():
            raise FileNotFoundError("COCO instance/caption annotation file is missing")
        O = _load("scripts/object_mask_benchmark.py", "object_benchmark_for_builder")
        selection_args = SimpleNamespace(
            instances_json=args.instances_json,
            captions_json=args.captions_json,
            cover_dir=args.cover_dir,
            seed=args.seed,
            frac_min=args.frac_min,
            frac_max=args.frac_max,
            max_per_category=args.max_per_category,
            sample_offset=args.sample_offset,
            num_samples=args.num_samples,
            size=args.size,
            prompt_mode=args.prompt_mode,
        )
        selected_objects = O.select_instances(selection_args, H, W)
        print(
            f"selected {len(selected_objects)} unique-image COCO objects "
            f"with prompt_mode={args.prompt_mode}", flush=True,
        )

    # Chunked, resumable saving: write a part file every `save_every` samples so a
    # death/restart loses at most one chunk. The trainer globs all *part*.pt files.
    save_every = 200
    out_dir = args.out.parent; out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out.stem
    meta = dict(seed=args.seed, payload_bits=args.payload_bits, chunk=chunk,
                 code_bits=args.codebook_code_bits, alpha=args.alpha,
                 margin_wins=wins, margin_stride=args.margin_stride, H=H, W=W,
                 in_ch=4 + len(wins), attack_pool=attack_pool,
                 cover_dir=str(args.cover_dir.resolve()), cover_offset=args.cover_offset,
                 mask_source="coco_object" if args.object_masks else "random_irregular",
                 instances_json=(str(args.instances_json.resolve()) if args.instances_json else None),
                 captions_json=(str(args.captions_json.resolve()) if args.captions_json else None),
                 sample_offset=args.sample_offset, prompt_mode=args.prompt_mode,
                 sdxl_model=str(args.sdxl_model), sd15_model=str(args.sd15_model))
    part_k = len(list(out_dir.glob(f"{stem}_part*.pt")))
    start_i = part_k * save_every
    if start_i >= args.num_samples:
        print(f"already complete ({part_k} parts)"); return
    if start_i:
        print(f"resuming from sample {start_i} ({part_k} parts done)", flush=True)

    buf_in, buf_m, buf_p, buf_a, buf_r = [], [], [], [], []

    def flush_part():
        nonlocal part_k
        if not buf_in:
            return
        torch.save(dict(inputs=torch.stack(buf_in), masks=torch.stack(buf_m),
                        payloads=torch.stack(buf_p), attacks=list(buf_a),
                        records=list(buf_r), meta=meta),
                   out_dir / f"{stem}_part{part_k:03d}.pt")
        print(f"saved {stem}_part{part_k:03d} ({len(buf_in)})", flush=True)
        part_k += 1
        buf_in.clear(); buf_m.clear(); buf_p.clear(); buf_a.clear(); buf_r.clear()

    for i in range(start_i, args.num_samples):
        if selected_objects is None:
            cover_path = covers[(args.cover_offset + i) % len(covers)]
            cover = M.load_cover(cover_path, args.size, device)
            frac = random.uniform(args.frac_min, args.frac_max)
            edit_mask = C.irregular_mask(frac, H, W, device)
            sample_prompt = args.prompt
            sample_record = {
                "index": i, "cover": str(cover_path), "prompt": sample_prompt,
                "mask_source": "random_irregular",
            }
        else:
            item = selected_objects[i]
            cover = M.load_cover(item.cover_path, args.size, device)
            edit_mask = item.latent_mask.unsqueeze(0).float().to(device)
            sample_prompt = item.prompt
            sample_record = {
                "index": i, "image_id": item.image_id, "ann_id": item.ann_id,
                "category_id": item.category_id, "caption_id": item.caption_id,
                "prompt_source": item.prompt_source, "prompt": sample_prompt,
                "cover": str(item.cover_path), "mask_source": "coco_object",
            }
        z_real = P.encode_image_to_latent(pipe, cover.to(torch.float16)).float()
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
            prompt=[sample_prompt], height=args.size, width=args.size,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            latents=x_T.to(torch.float16), output_type="pt",
            callback_on_step_end=make_cb(edit_mask, z_real, eps),
            callback_on_step_end_tensor_inputs=["latents"]).images).to(device)

        attack = random.choice(attack_pool)
        if attack == "identity":
            attacked = composite.float()
        elif attack in {"vae_roundtrip", "vae_sd", "vae"}:
            attacked = P.vae_roundtrip_attack(pipe, composite.float())
        elif attack.startswith("cross_sdxl") or attack.startswith("cross_sd15"):
            family = "sdxl" if attack.startswith("cross_sdxl") else "sd15"
            strength = float(attack.rsplit("_s", 1)[1]) if "_s" in attack else 0.25
            attacked = cross_regen_attack(
                family, composite.float(), sample_prompt, strength,
                args.seed + args.cover_offset + i + 9_000_007,
            )
        elif attack.startswith("diffusion_regen"):
            strength = float(attack.removeprefix("diffusion_regen_s")) if attack.startswith("diffusion_regen_s") else 0.25
            attacked = P.diffusion_regen_attack(pipe, composite.float(), sample_prompt, args.steps, args.guidance_scale, strength)
        else:
            attacked = P.apply_image_attack_with_meta(composite.float(), attack)[0]
        recovered = P.invert_image(pipe, inverse_scheduler, attacked.to(torch.float16),
                                   sample_prompt, args.steps, args.guidance_scale).float()

        mm = margin_maps(recovered, carriers, codebook, H, W, wins, args.margin_stride)  # [S,H,W]
        feat = torch.cat([recovered[0], mm], dim=0)  # [4+S, H, W]
        buf_in.append(feat.half().cpu())
        buf_m.append(edit_mask[0].bool().cpu())     # [1,H,W]
        buf_p.append(payload[0, 0].bool().cpu())    # [P]
        buf_a.append(attack)
        sample_record["attack"] = attack
        sample_record["attack_seed"] = args.seed + args.cover_offset + i + 9_000_007
        buf_r.append(sample_record)
        if (i + 1) % 8 == 0:
            print(f"[{i + 1}/{args.num_samples}] built ({attack})", flush=True)
        if len(buf_in) >= save_every:
            flush_part()

    flush_part()  # final partial chunk
    print(f"done: {part_k} parts under {out_dir}/{stem}_part*.pt", flush=True)


if __name__ == "__main__":
    main()
