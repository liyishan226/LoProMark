"""Evaluate every attack on the cached composites.

Re-inverts each composite per attack, recovers the mask with the
learned head, decodes the payload, and records bit, exact, and IoU
metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.mask_head import MaskHead, normalize_input


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "carrier_core")
B = _load("scripts/build_maskhead_dataset.py", "maskhead_dataset")  # margin_maps_fast

ATTACK_SUITE = [
    "identity", "jpeg_q40", "jpeg_q60", "jpeg_q80",
    "resize_s0.5", "resize_s0.75", "gaussian_blur_k3", "gaussian_blur_k7",
    "gaussian_noise_s0.05", "brightness_f1.5", "contrast_f1.5", "hue_h0.25",
    "saturation_f1.5", "median_k5", "vae_roundtrip", "diffusion_regen_s0.25",
]


def apply_attack(pipe, name, image, prompt, steps, gs):
    if name in {"vae_roundtrip", "vae_sd", "vae"}:
        return P.vae_roundtrip_attack(pipe, image)
    if name.startswith("diffusion_regen"):
        strength = 0.25
        if name.startswith("diffusion_regen_s"):
            strength = float(name.removeprefix("diffusion_regen_s"))
        return P.diffusion_regen_attack(pipe, image, prompt, steps, gs, strength)
    return P.apply_image_attack_with_meta(image, name)[0]


def roc_metrics(pos, neg):
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], reverse=True)
    Pn, Nn = len(pos), len(neg)
    tp = fp = 0; auc = 0.0; prev = 0.0; t1 = t01 = 0.0
    for s, lab in scored:
        if lab: tp += 1
        else: fp += 1
        tpr = tp / max(Pn, 1); fpr = fp / max(Nn, 1)
        auc += tpr * (fpr - prev); prev = fpr
        if fpr <= 0.01: t1 = max(t1, tpr)
        if fpr <= 0.001: t01 = max(t01, tpr)
    return auc, t1, t01


@torch.no_grad()
def make_key(meta, device):
    outer = meta["payload_bits"] // meta["chunk"] * meta["code_bits"]
    cb = P.make_binary_codebook(meta["chunk"], meta["code_bits"], meta["seed"] + 571, device)
    ph = P.make_carriers(outer, 4, meta["H"], meta["W"], meta["seed"] + 17, device)
    return ph, cb


@torch.no_grad()
def eval_one(pipe, inv, model, comp, key, attack, device, wins, stride, chunk, want_payload):
    ph, cb = key
    images, masks, payloads, meta = comp["images"], comp["masks"], comp["payloads"], comp["meta"]
    out = []
    for i in range(images.shape[0]):
        img = images[i:i+1].float().to(device) / 255.0
        att = apply_attack(pipe, attack, img, meta["prompt"], meta["steps"], meta["guidance_scale"]).to(torch.float16)
        rec = P.invert_image(pipe, inv, att, meta["prompt"], meta["steps"], meta["guidance_scale"]).float()
        mm = B.margin_maps_fast(rec, ph, cb, meta["H"], meta["W"], wins, stride)[None]
        x = torch.cat([rec, mm], dim=1)
        prob = torch.sigmoid(model(normalize_input(x)))
        corr = _corr(rec, prob, ph)
        det = float(P.codebook_likelihood_margin(corr, cb).mean())
        row = dict(det=det)
        if want_payload:
            pred = (prob > 0.5).float(); true = masks[i:i+1].float().to(device)
            tgt = payloads[i].float().to(device)
            dp = P.codebook_decode_soft(corr, cb, chunk)[0, 0]
            do = P.codebook_decode_soft(_corr(rec, true, ph), cb, chunk)[0, 0]
            inter = (pred*true).sum(); union = ((pred+true)>0).float().sum().clamp_min(1)
            row.update(area=float(true.mean()), bit=float((dp==tgt).float().mean()),
                       exact=int((dp==tgt).float().mean()>=0.999),
                       obit=float((do==tgt).float().mean()),
                       oexact=int((do==tgt).float().mean()>=0.999), iou=float(inter/union))
        out.append(row)
    return out


def _corr(recovered, weight, carriers):
    centered = P.normalize_latent_noise(recovered)
    masked = carriers * weight[0, 0][None, None]
    denom = masked.flatten(1).norm(dim=1).clamp_min(1e-6)
    unit = masked / denom[:, None, None, None]
    return torch.einsum("bchw,kchw->bk", centered, unit)[None]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=PATHS.runs / "maskhead_ckpt" / "fine3000_aug_FINAL.pt")
    ap.add_argument("--pos", type=Path, default=PATHS.runs / "composites" / "pos.pt")
    ap.add_argument("--neg", type=Path, default=PATHS.runs / "composites" / "neg.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attacks", default=",".join(ATTACK_SUITE))
    ap.add_argument("--max-samples", type=int, default=0)  # 0 = all (both)
    ap.add_argument("--max-pos", type=int, default=0)  # 0 = all
    ap.add_argument("--max-neg", type=int, default=0)  # 0 = all
    ap.add_argument("--out-csv", type=Path, default=PATHS.runs / "per_attack_eval" / "summary.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    model = MaskHead(in_ch=ck["meta"]["in_ch"], base=ck["base"]).to(device); model.load_state_dict(ck["model"]); model.eval()
    wins, stride = ck["meta"]["margin_wins"], ck["meta"]["margin_stride"]
    chunk = ck["meta"]["chunk"]

    # mmap=True keeps the large image tensors on disk (paged in per sample) so two
    # parallel eval processes don't exhaust system RAM.
    pos = torch.load(args.pos, map_location="cpu", mmap=True)
    neg = torch.load(args.neg, map_location="cpu", mmap=True)
    cap_pos = args.max_pos or args.max_samples
    cap_neg = args.max_neg or args.max_samples
    if cap_pos:
        for k in ("images", "masks", "payloads"):
            pos[k] = pos[k][:cap_pos]
    if cap_neg:
        for k in ("images", "masks", "payloads"):
            neg[k] = neg[k][:cap_neg]
    # Positives and negatives are both scored with the owner key.
    pos_key = make_key(pos["meta"], device)
    neg_key = pos_key

    pipe = StableDiffusionPipeline.from_pretrained(
        str(ck.get("model_root", PATHS.sd21_model)) if False else str(PATHS.sd21_model),
        local_files_only=True, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inv = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    import csv
    rows = []
    print(f"{'attack':>20} {'n':>4} {'bit_acc':>8} {'exact':>7} {'orc_bit':>8} {'IoU':>6} {'AUC':>7} {'TPR@1%':>7} {'t/smpl':>7}")
    for at in attacks:
        t0 = time.time()
        pr = eval_one(pipe, inv, model, pos, pos_key, at, device, wins, stride, chunk, want_payload=True)
        nr = eval_one(pipe, inv, model, neg, neg_key, at, device, wins, stride, chunk, want_payload=False)
        dt = (time.time() - t0) / max(1, len(pr) + len(nr))
        auc, t1, t01 = roc_metrics([r["det"] for r in pr], [r["det"] for r in nr])
        agg = lambda k: sum(r[k] for r in pr) / len(pr)
        rows.append(dict(attack=at, n=len(pr), bit_acc=agg("bit"), exact=agg("exact"),
                         oracle_bit=agg("obit"), iou=agg("iou"), auc=auc, tpr1=t1, tpr01=t01))
        print(f"{at:>20} {len(pr):>4} {agg('bit'):>8.4f} {agg('exact'):>7.3f} {agg('obit'):>8.4f} {agg('iou'):>6.3f} {auc:>7.4f} {t1:>7.3f} {dt:>6.2f}s", flush=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nsaved {args.out_csv}")


if __name__ == "__main__":
    main()
