"""Blended-latent partial generation over a pinned real cover.

Provides cover loading and the masked denoising loop used by the
composite builders.
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.messages import sample_binary_messages


def _load(path_rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path_rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "carrier_core")
K = _load("scripts/decode_helpers.py", "decode_helpers")  # reuse helpers


def load_cover(path: Path, size: int, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.from_numpy(
        __import__("numpy").asarray(img)
    ).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return t.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=PATHS.sd21_model)
    ap.add_argument("--cover-dir", type=Path, default=ROOT / "data" / "val2017")
    ap.add_argument("--output-dir", type=Path, default=PATHS.runs / "blended_inpaint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--prompt", default="a natural photo, realistic, detailed")
    ap.add_argument("--payload-bits", type=int, default=64)
    ap.add_argument("--codebook-chunk-bits", type=int, default=8)
    ap.add_argument("--codebook-code-bits", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=48.0)
    ap.add_argument("--local-stat-match", action="store_true", default=True)
    ap.add_argument("--region-fracs", default="0.10,0.20,0.30")
    ap.add_argument("--attacks", default="identity,jpeg_q50,resize_s0.5")
    ap.add_argument("--detect-stride-div", type=int, default=3)
    ap.add_argument("--save-composites", type=int, default=4)
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
    code_bits = outer_bits

    codebook = P.make_binary_codebook(chunk, args.codebook_code_bits, args.seed + 571, device)
    carriers = P.make_carriers(code_bits, pipe.unet.config.in_channels, H, W, args.seed + 17, device)

    fracs = [float(x) for x in args.region_fracs.split(",") if x.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    covers = sorted(args.cover_dir.glob("*.jpg"))[: max(args.num_samples * 3, 64)]

    rows: list[dict] = []
    saved = 0
    for idx in range(args.num_samples):
        cover_path = covers[idx % len(covers)]
        cover = load_cover(cover_path, args.size, device)
        z_real = P.encode_image_to_latent(pipe, cover.to(torch.float16)).float()  # [1,4,H,W]

        for frac in fracs:
            side = K.square_side_for_frac(frac, H, W)
            top = random.randint(0, H - side)
            left = random.randint(0, W - side)
            edit_mask = K.rect_mask(top, left, side, side, H, W, device)  # [1,1,H,W] 1=edited

            payload = sample_binary_messages(1, 1, args.payload_bits, device=device)
            outer = P.encode_payload(payload, "codebook", codebook, chunk)

            base_noise = torch.randn(1, pipe.unet.config.in_channels, H, W, device=device).float()
            wm_noise = P.add_local_noise_watermark(base_noise, edit_mask, outer, carriers, args.alpha, normalize=False)
            if args.local_stat_match:
                wm_noise = P.match_local_region_stats(wm_noise, base_noise, edit_mask)
            wm_noise = P.normalize_latent_noise(wm_noise)

            # blended-latent inpainting: pin background to noised z_real each step
            eps = torch.randn_like(z_real)
            init_sigma = pipe.scheduler.init_noise_sigma

            def make_cb(edit_mask, z_real, eps):
                def cb(pipe_, step, timestep, kwargs):
                    lat = kwargs["latents"].float()
                    ts = pipe_.scheduler.timesteps
                    nxt = step + 1
                    if nxt < len(ts):
                        bg = pipe_.scheduler.add_noise(z_real, eps, ts[nxt].unsqueeze(0))
                    else:
                        bg = z_real  # final step -> clean background
                    blended = edit_mask * lat + (1.0 - edit_mask) * bg
                    return {"latents": blended.to(kwargs["latents"].dtype)}
                return cb

            # build x_T: edited region carries the carrier; background = fully-noised real latent
            pipe.scheduler.set_timesteps(args.steps, device=device)
            x_T = edit_mask * (wm_noise * init_sigma) + (1.0 - edit_mask) * pipe.scheduler.add_noise(
                z_real, eps, pipe.scheduler.timesteps[0].unsqueeze(0)
            )

            out = pipe(
                prompt=[args.prompt], height=args.size, width=args.size,
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                latents=x_T.to(torch.float16), output_type="pt",
                callback_on_step_end=make_cb(edit_mask, z_real, eps),
                callback_on_step_end_tensor_inputs=["latents"],
            )
            composite = P.tensor_to_bchw(out.images).to(device)

            if saved < args.save_composites:
                P.save_image(composite, args.output_dir / f"composite_{idx}_{frac}.png")
                P.save_image(cover, args.output_dir / f"cover_{idx}_{frac}.png")
                # also dump the edit mask at image resolution for visual check
                up = torch.nn.functional.interpolate(edit_mask, size=(args.size, args.size), mode="nearest")
                P.save_image(up.repeat(1, 3, 1, 1), args.output_dir / f"mask_{idx}_{frac}.png")
                saved += 1

            stride = max(1, side // args.detect_stride_div)
            cands = K.candidate_positions(side, H, W, stride)

            for attack in attacks:
                if attack == "identity":
                    attacked = composite.float()
                else:
                    attacked, _ = P.apply_image_attack_with_meta(composite.float(), attack)
                attacked = attacked.to(torch.float16)
                recovered = P.invert_image(
                    pipe, inverse_scheduler, attacked, args.prompt, args.steps, args.guidance_scale
                ).float()

                best_score, best_pos, wrong = -float("inf"), cands[0], []
                for (ct, cl) in cands:
                    cmask = K.rect_mask(ct, cl, side, side, H, W, device)
                    s = K.detect_score_at(recovered, cmask, carriers, codebook)
                    if not (abs(ct - top) <= stride // 2 and abs(cl - left) <= stride // 2):
                        wrong.append(s)
                    if s > best_score:
                        best_score, best_pos = s, (ct, cl)
                pred_mask = K.rect_mask(best_pos[0], best_pos[1], side, side, H, W, device)
                iou = K.mask_iou(pred_mask, edit_mask)

                pred_payload = K.decode_payload_at(recovered, pred_mask, carriers, codebook, chunk)
                pred_bit, pred_exact = K.bit_acc_exact(pred_payload, payload[0, 0])
                oracle_payload = K.decode_payload_at(recovered, edit_mask, carriers, codebook, chunk)
                oracle_bit, oracle_exact = K.bit_acc_exact(oracle_payload, payload[0, 0])

                rows.append(dict(
                    sample=idx, attack=attack, region_frac=round(frac, 3),
                    iou=round(iou, 4), det_hit=int(iou >= 0.5),
                    pred_score=round(best_score, 4),
                    best_wrong_score=round(max(wrong), 4) if wrong else "",
                    pred_bit_acc=round(pred_bit, 4), pred_exact=int(pred_exact),
                    oracle_bit_acc=round(oracle_bit, 4), oracle_exact=int(oracle_exact),
                ))
        print(f"[{idx + 1}/{args.num_samples}] {cover_path.name} done", flush=True)

    csv_path = args.output_dir / f"blended_inpaint_{args.device.replace(':', '')}_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nmean over samples, real background and marked region")
    print(f"{'frac':>5} {'attack':>12} {'det_hit':>8} {'iou':>6} {'pred_ex':>8} {'oracle_ex':>10} {'pred_bit':>9}")
    for fr, at in sorted({(r["region_frac"], r["attack"]) for r in rows}, key=lambda x: (x[0], x[1])):
        sub = [r for r in rows if r["region_frac"] == fr and r["attack"] == at]
        n = len(sub)
        g = lambda k: sum(r[k] for r in sub) / n
        print(f"{fr:>5} {at:>12} {g('det_hit'):>8.3f} {g('iou'):>6.3f} {g('pred_exact'):>8.3f} {g('oracle_exact'):>10.3f} {g('pred_bit_acc'):>9.3f}")
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()
