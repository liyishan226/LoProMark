"""Mask gating utilities for unknown-size, irregular edit masks.

Predicts the carrier mask from recovered latents and margin evidence
without knowing mask size or shape.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
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
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "carrier_core")
K = _load("scripts/decode_helpers.py", "decode_helpers")
M = _load("scripts/blended_inpaint.py", "blended_inpaint")  # for load_cover


def irregular_mask(frac, H, W, device, n_rects=3):
    """Blob mask = union of n_rects random rectangles totalling ~frac area."""
    mask = torch.zeros(1, 1, H, W, device=device)
    target = frac * H * W
    guard = 0
    while mask.sum().item() < target and guard < 50:
        side_h = random.randint(max(3, H // 12), max(4, H // 5))
        side_w = random.randint(max(3, W // 12), max(4, W // 5))
        t = random.randint(0, H - side_h)
        l = random.randint(0, W - side_w)
        mask[0, 0, t : t + side_h, l : l + side_w] = 1.0
        guard += 1
    return mask


def multiscale_detect(recovered, carriers, codebook, H, W, sides, stride_div):
    best_score, best = -float("inf"), None
    for side in sides:
        stride = max(1, side // stride_div)
        tops = sorted(set(list(range(0, H - side + 1, stride)) + [H - side]))
        lefts = sorted(set(list(range(0, W - side + 1, stride)) + [W - side]))
        for t in tops:
            for l in lefts:
                m = K.rect_mask(t, l, side, side, H, W, recovered.device)
                s = K.detect_score_at(recovered, m, carriers, codebook)
                if s > best_score:
                    best_score, best = s, (t, l, side)
    t, l, side = best
    return best_score, K.rect_mask(t, l, side, side, H, W, recovered.device), side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=PATHS.sd21_model)
    ap.add_argument("--cover-dir", type=Path, default=ROOT / "data" / "val2017")
    ap.add_argument("--output-dir", type=Path, default=PATHS.runs / "mask_gate")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--prompt", default="a natural photo, realistic, detailed")
    ap.add_argument("--payload-bits", type=int, default=64)
    ap.add_argument("--codebook-chunk-bits", type=int, default=8)
    ap.add_argument("--codebook-code-bits", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=48.0)
    ap.add_argument("--shapes", default="square,irregular")
    ap.add_argument("--region-fracs", default="0.10,0.20")
    ap.add_argument("--attacks", default="identity,jpeg_q50")
    ap.add_argument("--search-sides", default="12,18,24,30,36")  # true size is unknown to the detector
    ap.add_argument("--stride-div", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_root), local_files_only=True, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    H = args.size // pipe.vae_scale_factor
    W = args.size // pipe.vae_scale_factor
    chunk = args.codebook_chunk_bits
    outer_bits = args.payload_bits // chunk * args.codebook_code_bits
    codebook = P.make_binary_codebook(chunk, args.codebook_code_bits, args.seed + 571, device)
    carriers = P.make_carriers(outer_bits, pipe.unet.config.in_channels, H, W, args.seed + 17, device)

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    fracs = [float(x) for x in args.region_fracs.split(",") if x.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    sides = [int(s) for s in args.search_sides.split(",") if s.strip()]
    covers = sorted(args.cover_dir.glob("*.jpg"))[: max(args.num_samples * 3, 64)]

    rows: list[dict] = []
    for idx in range(args.num_samples):
        cover = M.load_cover(covers[idx % len(covers)], args.size, device)
        z_real = P.encode_image_to_latent(pipe, cover.to(torch.float16)).float()

        for shape in shapes:
            for frac in fracs:
                if shape == "square":
                    side = K.square_side_for_frac(frac, H, W)
                    top = random.randint(0, H - side)
                    left = random.randint(0, W - side)
                    edit_mask = K.rect_mask(top, left, side, side, H, W, device)
                else:
                    edit_mask = irregular_mask(frac, H, W, device)
                true_area = float(edit_mask.mean().item())

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
                    z_real, eps, pipe.scheduler.timesteps[0].unsqueeze(0)
                )
                composite = P.tensor_to_bchw(
                    pipe(
                        prompt=[args.prompt], height=args.size, width=args.size,
                        num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                        latents=x_T.to(torch.float16), output_type="pt",
                        callback_on_step_end=make_cb(edit_mask, z_real, eps),
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images
                ).to(device)

                for attack in attacks:
                    if attack == "identity":
                        attacked = composite.float()
                    else:
                        attacked, _ = P.apply_image_attack_with_meta(composite.float(), attack)
                    recovered = P.invert_image(
                        pipe, inverse_scheduler, attacked.to(torch.float16),
                        args.prompt, args.steps, args.guidance_scale,
                    ).float()

                    pred_score, pred_mask, pred_side = multiscale_detect(
                        recovered, carriers, codebook, H, W, sides, args.stride_div
                    )
                    iou = K.mask_iou(pred_mask, edit_mask)
                    pred_payload = K.decode_payload_at(recovered, pred_mask, carriers, codebook, chunk)
                    pred_bit, pred_exact = K.bit_acc_exact(pred_payload, payload[0, 0])
                    oracle_payload = K.decode_payload_at(recovered, edit_mask, carriers, codebook, chunk)
                    oracle_bit, oracle_exact = K.bit_acc_exact(oracle_payload, payload[0, 0])

                    rows.append(dict(
                        sample=idx, shape=shape, attack=attack, region_frac=round(frac, 3),
                        true_area=round(true_area, 4), pred_side=pred_side,
                        iou=round(iou, 4), det_hit=int(iou >= 0.5),
                        pred_score=round(pred_score, 4),
                        pred_bit_acc=round(pred_bit, 4), pred_exact=int(pred_exact),
                        oracle_bit_acc=round(oracle_bit, 4), oracle_exact=int(oracle_exact),
                    ))
        print(f"[{idx + 1}/{args.num_samples}] done", flush=True)

    csv_path = args.output_dir / f"gate_{args.device.replace(':', '')}_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nmean over samples, multi-scale detector with unknown size")
    print(f"{'shape':>10} {'frac':>5} {'attack':>10} {'det_hit':>8} {'iou':>6} {'pred_ex':>8} {'oracle_ex':>10}")
    for sh, fr, at in sorted({(r["shape"], r["region_frac"], r["attack"]) for r in rows}):
        sub = [r for r in rows if r["shape"] == sh and r["region_frac"] == fr and r["attack"] == at]
        n = len(sub)
        g = lambda k: sum(r[k] for r in sub) / n
        print(f"{sh:>10} {fr:>5} {at:>10} {g('det_hit'):>8.3f} {g('iou'):>6.3f} {g('pred_exact'):>8.3f} {g('oracle_exact'):>10.3f}")
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()
