"""Detector and payload-decode helpers for latent-region carriers.

Provides sub-region scoring and payload decoding reused by the dataset
builder and the evaluators.
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


def _load_probe_module():
    """Load a sibling script module by file path."""
    path = ROOT / "scripts" / "carrier_core.py"
    spec = importlib.util.spec_from_file_location("carrier_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ != "__main__", so main() does not run
    return mod


P = _load_probe_module()


def square_side_for_frac(frac: float, height: int, width: int) -> int:
    side = int(round(math.sqrt(frac * height * width)))
    return max(4, min(side, min(height, width)))


def rect_mask(top: int, left: int, rh: int, rw: int, height: int, width: int, device) -> torch.Tensor:
    mask = torch.zeros(1, 1, height, width, device=device)
    mask[0, 0, top : top + rh, left : left + rw] = 1.0
    return mask


def candidate_positions(side: int, height: int, width: int, stride: int) -> list[tuple[int, int]]:
    tops = list(range(0, max(1, height - side + 1), stride))
    lefts = list(range(0, max(1, width - side + 1), stride))
    if (height - side) not in tops:
        tops.append(max(0, height - side))
    if (width - side) not in lefts:
        lefts.append(max(0, width - side))
    return [(t, l) for t in tops for l in lefts]


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a * b).sum().item()
    union = ((a + b) > 0).float().sum().item()
    return inter / max(union, 1.0)


def decode_payload_at(recovered, mask, carriers, codebook, chunk_bits):
    corr = P.correlate_local_noise_watermark(recovered, mask, carriers)  # [1,1,code_bits]
    payload = P.codebook_decode_soft(corr, codebook, chunk_bits)  # [1,1,payload_bits]
    return payload[0, 0]


def detect_score_at(recovered, mask, carriers, codebook):
    corr = P.correlate_local_noise_watermark(recovered, mask, carriers)
    margin = P.codebook_likelihood_margin(corr, codebook)  # [1,1,n_chunks]
    return float(margin.mean().detach().cpu())


def bit_acc_exact(decoded: torch.Tensor, target: torch.Tensor):
    match = (decoded == target).float()
    return float(match.mean().item()), float((match.mean() >= 0.999).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=PATHS.sd21_model)
    ap.add_argument("--output-dir", type=Path, default=PATHS.runs / "rect_detection")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--payload-bits", type=int, default=64)
    ap.add_argument("--codebook-chunk-bits", type=int, default=8)
    ap.add_argument("--codebook-code-bits", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=48.0)  # main 64-bit setting
    ap.add_argument("--local-stat-match", action="store_true", default=True)
    ap.add_argument("--region-fracs", default="0.10,0.20,0.30")
    ap.add_argument("--attacks", default="identity,jpeg_q50,resize_s0.5")
    ap.add_argument("--detect-stride-div", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_root),
        local_files_only=True,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    H = args.height // pipe.vae_scale_factor
    W = args.width // pipe.vae_scale_factor
    chunk = args.codebook_chunk_bits
    outer_bits = args.payload_bits // chunk * args.codebook_code_bits
    code_bits = outer_bits  # repeat = 1

    codebook = P.make_binary_codebook(chunk, args.codebook_code_bits, args.seed + 571, device)
    carriers = P.make_carriers(code_bits, pipe.unet.config.in_channels, H, W, args.seed + 17, device)

    fracs = [float(x) for x in args.region_fracs.split(",") if x.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    prompts = P.PROMPTS

    rows: list[dict] = []
    for idx in range(args.num_samples):
        prompt = prompts[idx % len(prompts)]
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed * 1000 + idx)
        noise = torch.randn(1, pipe.unet.config.in_channels, H, W, generator=gen, device=device, dtype=torch.float16)

        for frac in fracs:
            side = square_side_for_frac(frac, H, W)
            top = random.randint(0, H - side)
            left = random.randint(0, W - side)
            true_mask = rect_mask(top, left, side, side, H, W, device)
            true_area = float(true_mask.mean().item())

            payload = sample_binary_messages(1, 1, args.payload_bits, device=device)  # [1,1,P]
            outer = P.encode_payload(payload, "codebook", codebook, chunk)  # [1,1,code_bits]

            wm = P.add_local_noise_watermark(noise.float(), true_mask, outer, carriers, args.alpha, normalize=False)
            if args.local_stat_match:
                wm = P.match_local_region_stats(wm, noise.float(), true_mask)
            wm = P.normalize_latent_noise(wm).to(torch.float16)

            image = P.tensor_to_bchw(
                pipe(
                    prompt=[prompt],
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    latents=wm,
                    output_type="pt",
                ).images
            ).to(device)

            stride = max(1, side // args.detect_stride_div)
            cands = candidate_positions(side, H, W, stride)

            for attack in attacks:
                if attack == "identity":
                    attacked = image.float()
                else:
                    attacked, _ = P.apply_image_attack_with_meta(image.float(), attack)
                attacked = attacked.to(torch.float16)

                recovered = P.invert_image(
                    pipe, inverse_scheduler, attacked, prompt, args.steps, args.guidance_scale
                ).float()

                # detection: scan candidate positions and keep the best score
                best_score = -float("inf")
                best_pos = cands[0]
                wrong_scores = []
                for (ct, cl) in cands:
                    cmask = rect_mask(ct, cl, side, side, H, W, device)
                    s = detect_score_at(recovered, cmask, carriers, codebook)
                    is_true = abs(ct - top) <= stride // 2 and abs(cl - left) <= stride // 2
                    if not is_true:
                        wrong_scores.append(s)
                    if s > best_score:
                        best_score = s
                        best_pos = (ct, cl)
                pred_mask = rect_mask(best_pos[0], best_pos[1], side, side, H, W, device)
                iou = mask_iou(pred_mask, true_mask)
                center_dist = math.hypot(best_pos[0] - top, best_pos[1] - left)
                true_score = detect_score_at(recovered, true_mask, carriers, codebook)
                best_wrong = max(wrong_scores) if wrong_scores else float("nan")

                # payload: predicted, true and random masks
                pred_payload = decode_payload_at(recovered, pred_mask, carriers, codebook, chunk)
                pred_bit, pred_exact = bit_acc_exact(pred_payload, payload[0, 0])
                oracle_payload = decode_payload_at(recovered, true_mask, carriers, codebook, chunk)
                oracle_bit, oracle_exact = bit_acc_exact(oracle_payload, payload[0, 0])
                rt = random.randint(0, H - side)
                rl = random.randint(0, W - side)
                rand_mask = rect_mask(rt, rl, side, side, H, W, device)
                rand_payload = decode_payload_at(recovered, rand_mask, carriers, codebook, chunk)
                rand_bit, _ = bit_acc_exact(rand_payload, payload[0, 0])

                rows.append(
                    dict(
                        sample=idx,
                        attack=attack,
                        region_frac=round(frac, 3),
                        true_area=round(true_area, 4),
                        iou=round(iou, 4),
                        center_dist=round(center_dist, 2),
                        det_hit=int(iou >= 0.5),
                        pred_score=round(best_score, 4),
                        true_score=round(true_score, 4),
                        best_wrong_score=round(best_wrong, 4) if best_wrong == best_wrong else "",
                        pred_bit_acc=round(pred_bit, 4),
                        pred_exact=int(pred_exact),
                        oracle_bit_acc=round(oracle_bit, 4),
                        oracle_exact=int(oracle_exact),
                        rand_bit_acc=round(rand_bit, 4),
                    )
                )
        print(f"[{idx + 1}/{args.num_samples}] done", flush=True)

    csv_path = args.output_dir / f"rect_detection_{args.device.replace(':', '')}_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nmean over samples")
    print(f"{'frac':>6} {'attack':>12} {'det_hit':>8} {'iou':>6} {'pred_ex':>8} {'oracle_ex':>10} {'rand_bit':>9}")
    keys = sorted({(r["region_frac"], r["attack"]) for r in rows})
    for fr, at in keys:
        sub = [r for r in rows if r["region_frac"] == fr and r["attack"] == at]
        n = len(sub)
        det = sum(r["det_hit"] for r in sub) / n
        iou = sum(r["iou"] for r in sub) / n
        pex = sum(r["pred_exact"] for r in sub) / n
        oex = sum(r["oracle_exact"] for r in sub) / n
        rb = sum(r["rand_bit_acc"] for r in sub) / n
        print(f"{fr:>6} {at:>12} {det:>8.3f} {iou:>6.3f} {pex:>8.3f} {oex:>10.3f} {rb:>9.3f}")
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()
