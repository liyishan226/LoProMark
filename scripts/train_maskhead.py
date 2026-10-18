"""Train the mask head from cached shards.

Trains the compact U-Net and selects the checkpoint by validation
payload recovery.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.mask_head import MaskHead, dice_bce_loss, tversky_bce_loss, dilate_mask, normalize_input


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "carrier_core")
K = _load("scripts/decode_helpers.py", "decode_helpers")


def augment_batch(xb, yb):
    """Random flips and rot90 applied identically to input features and mask."""
    import random as _r
    if _r.random() < 0.5:
        xb = xb.flip(-1); yb = yb.flip(-1)
    if _r.random() < 0.5:
        xb = xb.flip(-2); yb = yb.flip(-2)
    k = _r.randint(0, 3)
    if k:
        xb = torch.rot90(xb, k, (-2, -1)); yb = torch.rot90(yb, k, (-2, -1))
    return xb.contiguous(), yb.contiguous()


def load_shards(patterns: str):
    """Load one or more comma-separated shard globs without duplicating files."""
    requested = [item.strip() for item in patterns.split(",") if item.strip()]
    files = sorted({f for pattern in requested for f in glob.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"no shards match {patterns}")
    inputs, masks, payloads, domains, meta = [], [], [], [], None
    compatibility_keys = ("seed", "payload_bits", "chunk", "code_bits", "alpha",
                          "margin_wins", "margin_stride", "H", "W", "in_ch")
    for f in files:
        d = torch.load(f, map_location="cpu")
        inputs.append(d["inputs"]); masks.append(d["masks"]); payloads.append(d["payloads"])
        attacks = list(d.get("attacks", ["unknown"] * len(d["inputs"])))
        if len(attacks) != len(d["inputs"]):
            raise ValueError(f"attack metadata length mismatch in {f}")
        for attack in attacks:
            if str(attack).startswith("cross_sdxl"):
                domains.append("sdxl_regen")
            elif str(attack).startswith("diffusion_regen"):
                domains.append("sd21_regen")
            else:
                domains.append("classical")
        shard_meta = d["meta"]
        if meta is None:
            meta = dict(shard_meta)
        else:
            mismatched = [key for key in compatibility_keys if meta.get(key) != shard_meta.get(key)]
            if mismatched:
                raise ValueError(f"incompatible shard metadata in {f}: {mismatched}")
    meta["training_shard_patterns"] = requested
    meta["training_shard_files"] = files
    return torch.cat(inputs), torch.cat(masks), torch.cat(payloads), meta, domains


def select_feature_view(inputs, meta, mode: str, margin_index: int):
    """Select a reduced input feature view for retraining ablations."""
    original_in_ch = int(meta["in_ch"])
    if mode == "all":
        selected = inputs
    elif mode == "latent-only":
        selected = inputs[:, :4]
    elif mode == "single-margin":
        margin_count = original_in_ch - 4
        if not 0 <= margin_index < margin_count:
            raise ValueError(f"--margin-index must be in [0,{margin_count - 1}]")
        selected = torch.cat(
            [inputs[:, :4], inputs[:, 4 + margin_index : 5 + margin_index]], dim=1,
        )
    else:
        raise ValueError(f"unknown feature mode: {mode}")
    meta = dict(meta)
    meta["original_in_ch"] = original_in_ch
    meta["feature_mode"] = mode
    meta["margin_index"] = int(margin_index)
    meta["in_ch"] = int(selected.shape[1])
    return selected, meta


def decode_weighted(recovered, weight, carriers, codebook, chunk):
    """Decode payload using a soft per-location weight (probability) over carriers,
    instead of a hard binary mask. Down-weights uncertain border pixels."""
    centered = P.normalize_latent_noise(recovered)
    w = weight[0, 0]
    masked = carriers * w[None, None]
    denom = masked.flatten(1).norm(dim=1).clamp_min(1e-6)
    unit = masked / denom[:, None, None, None]
    corr = torch.einsum("bchw,kchw->bk", centered, unit)[None]  # [1,1,code_bits]
    return P.codebook_decode_soft(corr, codebook, chunk)[0, 0]


@torch.no_grad()
def evaluate(model, inputs, masks, payloads, carriers, codebook, chunk, device, thresh=0.5, dilate=0):
    model.eval()
    iou_sum = exact_hard = exact_soft = oracle_exact = 0
    n = inputs.shape[0]
    for i in range(n):
        x = inputs[i : i + 1].float().to(device)
        prob = torch.sigmoid(model(normalize_input(x)))
        pred = (prob > thresh).float()  # [1,1,H,W]
        true = masks[i : i + 1].float().to(device)
        inter = (pred * true).sum(); union = ((pred + true) > 0).float().sum().clamp_min(1)
        iou_sum += float(inter / union)  # IoU on the raw predicted mask
        recovered = x[:, :4]
        tgt = payloads[i].float().to(device)
        dec_mask = dilate_mask(pred, dilate) if dilate > 1 else pred
        if dec_mask.sum() > 0:
            dh = K.decode_payload_at(recovered, dec_mask, carriers, codebook, chunk)
            exact_hard += int((dh == tgt).float().mean() >= 0.999)
        ds = decode_weighted(recovered, prob, carriers, codebook, chunk)  # soft prob-weighted
        exact_soft += int((ds == tgt).float().mean() >= 0.999)
        do = K.decode_payload_at(recovered, true, carriers, codebook, chunk)
        oracle_exact += int((do == tgt).float().mean() >= 0.999)
    return iou_sum / n, exact_hard / n, exact_soft / n, oracle_exact / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards",
        default=str(PATHS.runs / "maskhead_dataset" / "*.pt"),
        help="One glob or a comma-separated list of globs for mixed-domain training.",
    )
    ap.add_argument("--out", type=Path, default=PATHS.runs / "maskhead_ckpt" / "maskhead.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--loss", choices=["dice", "tversky"], default="dice")
    ap.add_argument("--tversky-beta", type=float, default=0.7)
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--save-by", choices=["payload", "iou"], default="payload")
    ap.add_argument(
        "--feature-mode", choices=["all", "latent-only", "single-margin"],
        default="all", help="Retrain on a declared reduced feature view.",
    )
    ap.add_argument(
        "--margin-index", type=int, default=1,
        help="Margin-map channel used by --feature-mode single-margin (0-based).",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--domain-balanced", action="store_true",
        help="Sample classical, SD2.1-regeneration, and SDXL-regeneration domains equally after the split.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    inputs, masks, payloads, meta, domains = load_shards(args.shards)
    inputs, meta = select_feature_view(inputs, meta, args.feature_mode, args.margin_index)
    in_ch = meta["in_ch"]; chunk = meta["chunk"]; H, W = meta["H"], meta["W"]
    n = inputs.shape[0]
    n_val = max(1, int(n * args.val_frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"dataset n={n} (train {len(tr_idx)}, val {len(val_idx)}), in_ch={in_ch}")

    # carriers/codebook for downstream decode (same key/seed as builder)
    outer_bits = meta["payload_bits"] // chunk * meta["code_bits"]
    codebook = P.make_binary_codebook(chunk, meta["code_bits"], meta["seed"] + 571, device)
    carriers = P.make_carriers(outer_bits, 4, H, W, meta["seed"] + 17, device)

    masks_f = masks.float()
    tr = TensorDataset(inputs[tr_idx], masks_f[tr_idx])
    if args.domain_balanced:
        tr_domains = [domains[int(i)] for i in tr_idx]
        counts = Counter(tr_domains)
        weights = torch.tensor([1.0 / counts[d] for d in tr_domains], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(tr_idx), replacement=True,
            generator=torch.Generator().manual_seed(args.seed + 991),
        )
        loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler)
        print(f"domain-balanced train counts={dict(sorted(counts.items()))}")
    else:
        loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    model = MaskHead(in_ch=in_ch, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb = xb.float().to(device); yb = yb.to(device)
            if args.augment:
                xb, yb = augment_batch(xb, yb)
            xb = normalize_input(xb)
            logit = model(xb)
            if args.loss == "tversky":
                loss = tversky_bce_loss(logit, yb, alpha=1.0 - args.tversky_beta, beta=args.tversky_beta)
            else:
                loss = dice_bce_loss(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * xb.shape[0]
        sched.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            iou, ph, ps, oex = evaluate(model, inputs[val_idx], masks[val_idx], payloads[val_idx],
                                        carriers, codebook, chunk, device, dilate=args.dilate)
            score = iou if args.save_by == "iou" else max(ph, ps)
            print(f"ep {ep+1:3d} loss {tot/len(tr_idx):.4f} | IoU {iou:.3f} exact_hard {ph:.3f} exact_soft {ps:.3f} oracle {oex:.3f}", flush=True)
            if score > best:
                best = score
                torch.save(
                    dict(
                        model=model.state_dict(), meta=meta, base=args.base,
                        training=dict(
                            shards=args.shards, seed=args.seed, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr, loss=args.loss,
                            augment=args.augment, save_by=args.save_by,
                            feature_mode=args.feature_mode, margin_index=args.margin_index,
                            domain_balanced=args.domain_balanced,
                        ),
                    ),
                    args.out,
                )
    print(f"\nbest val pred_exact {best:.3f}  saved {args.out}")


if __name__ == "__main__":
    main()
