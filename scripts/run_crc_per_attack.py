"""Resumable per-attack evaluation with CRC list decoding and raw fallback.

Reports effective data-payload metrics for composites carrying data
bits plus CRC parity bits.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.mask_head import MaskHead, normalize_input


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "carrier_core")
B = _load("scripts/build_maskhead_dataset.py", "maskhead_dataset")
E = _load("scripts/eval_per_attack.py", "per_attack_eval")
CRC = _load("scripts/crc_listdecode.py", "crc_listdecode_core")

FIELDS = [
    "kind",
    "idx",
    "attack_seed",
    "det",
    "area",
    "raw_payload_bit",
    "raw_payload_exact",
    "raw_data_bit",
    "raw_data_exact",
    "oracle_topk",
    "crc_valid",
    "crc_bit",
    "crc_exact",
    "false_accept",
    "crc_or_raw_bit",
    "crc_or_raw_exact",
    "obit",
    "oexact",
    "iou",
]


def done_indices(path: Path) -> set[tuple[str, int]]:
    done = set()
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                done.add((row["kind"], int(row["idx"])))
    return done


def blank_row(kind: str, idx: int, det: float, attack_seed: int | str = "") -> dict:
    row = {field: "" for field in FIELDS}
    row.update(kind=kind, idx=idx, attack_seed=attack_seed, det=round(det, 4))
    return row


def checkpoint_feature_view(features: torch.Tensor, checkpoint_meta: dict) -> torch.Tensor:
    """Apply the same declared feature selection used to retrain an ablation head."""
    mode = checkpoint_meta.get("feature_mode", "all")
    if mode == "all":
        return features
    if mode == "latent-only":
        return features[:, :4]
    if mode == "single-margin":
        index = int(checkpoint_meta.get("margin_index", 1))
        return torch.cat([features[:, :4], features[:, 4 + index : 5 + index]], dim=1)
    raise ValueError(f"unsupported checkpoint feature_mode: {mode}")


@torch.no_grad()
def run_attack(
    pipe: StableDiffusionPipeline,
    inv: DDIMInverseScheduler,
    model: MaskHead,
    pos: dict,
    neg: dict,
    key: tuple[torch.Tensor, torch.Tensor],
    attack: str,
    device: torch.device,
    wins: list[int],
    stride: int,
    topk: int,
    path: Path,
    flush_every: int,
    max_pos: int,
    max_neg: int,
    checkpoint_meta: dict,
    start_pos: int = 0,
    start_neg: int = 0,
    skip_pos: bool = False,
    skip_neg: bool = False,
    attack_seed: int | None = None,
    attack_index: int = 0,
) -> None:
    ph, cb = key
    meta = pos["meta"]
    chunk = meta["chunk"]
    cfg = CRC.Config(meta["data_bits"], meta["crc_bits"], (topk,))
    done = done_indices(path)
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    pos_total, neg_total = int(pos["images"].shape[0]), int(neg["images"].shape[0])
    pos_end = min(start_pos + (max_pos or pos_total), pos_total)
    neg_end = min(start_neg + (max_neg or neg_total), neg_total)
    buf = 0

    groups = []
    if not skip_pos:
        groups.append(("pos", pos, start_pos, pos_end, True))
    if not skip_neg:
        groups.append(("neg", neg, start_neg, neg_end, False))
    for kind, comp, start, end, want_payload in groups:
        for i in range(start, end):
            if (kind, i) in done:
                continue
            img = comp["images"][i : i + 1].float().to(device) / 255.0
            sample_seed = ""
            if attack_seed is not None:
                sample_seed = attack_seed + attack_index * 1_000_003 + i
                if kind == "neg":
                    sample_seed += 10_000_000
                random.seed(sample_seed)
                np.random.seed(sample_seed % (2**32 - 1))
                torch.manual_seed(sample_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(sample_seed)
            attacked = E.apply_attack(pipe, attack, img, meta["prompt"], meta["steps"], meta["guidance_scale"]).to(torch.float16)
            rec = P.invert_image(pipe, inv, attacked, meta["prompt"], meta["steps"], meta["guidance_scale"]).float()
            mm = B.margin_maps_fast(rec, ph, cb, meta["H"], meta["W"], wins, stride)[None]
            features = checkpoint_feature_view(torch.cat([rec, mm], dim=1), checkpoint_meta)
            prob = torch.sigmoid(model(normalize_input(features)))
            corr = E._corr(rec, prob, ph)
            det = float(P.codebook_likelihood_margin(corr, cb).mean())
            row = blank_row(kind, i, det, sample_seed)

            if want_payload:
                true_mask = comp["masks"][i : i + 1].float().to(device)
                true_full = comp["payloads"][i].float().to(device)
                true_data = comp["data_payloads"][i].float().to(device)
                pred_mask = (prob > 0.5).float()
                scores = CRC.codebook_scores(corr, cb)
                raw = P.codebook_decode_soft(corr, cb, chunk)[0, 0]
                raw_data = raw[: cfg.data_bits]

                selected, _ = CRC.select_crc_candidate(scores, topk, cfg, chunk, device)
                if selected is None:
                    crc_valid = 0
                    selected_data = raw_data
                    crc_bit = 0.0
                    crc_exact = 0
                    false_accept = 0
                else:
                    crc_valid = 1
                    selected_data = selected[: cfg.data_bits]
                    crc_bit = float((selected_data == true_data).float().mean())
                    crc_exact = int(bool((selected_data == true_data).all()))
                    false_accept = int(not bool((selected_data == true_data).all()))

                true_idx = CRC.chunk_indices(true_full, chunk)
                top_idx = scores.topk(topk, dim=-1).indices
                oracle_topk = int(bool(((true_idx[:, None] == top_idx[:, :topk]).any(-1)).all()))

                raw_oracle = P.codebook_decode_soft(E._corr(rec, true_mask, ph), cb, chunk)[0, 0]
                raw_oracle_data = raw_oracle[: cfg.data_bits]
                inter = (pred_mask * true_mask).sum()
                union = ((pred_mask + true_mask) > 0).float().sum().clamp_min(1)

                row.update(
                    area=round(float(true_mask.mean()), 4),
                    raw_payload_bit=round(float((raw == true_full).float().mean()), 4),
                    raw_payload_exact=int(bool((raw == true_full).all())),
                    raw_data_bit=round(float((raw_data == true_data).float().mean()), 4),
                    raw_data_exact=int(bool((raw_data == true_data).all())),
                    oracle_topk=oracle_topk,
                    crc_valid=crc_valid,
                    crc_bit=round(crc_bit, 4),
                    crc_exact=crc_exact,
                    false_accept=false_accept,
                    crc_or_raw_bit=round(float((selected_data == true_data).float().mean()), 4),
                    crc_or_raw_exact=int(bool((selected_data == true_data).all())),
                    obit=round(float((raw_oracle_data == true_data).float().mean()), 4),
                    oexact=int(bool((raw_oracle_data == true_data).all())),
                    iou=round(float(inter / union), 4),
                )

            writer.writerow(row)
            buf += 1
            if buf >= flush_every:
                handle.flush()
                buf = 0
    handle.flush()
    handle.close()
    print(f"[{attack}] done -> {path}", flush=True)


def summarize(outdir: Path, attacks: list[str]) -> None:
    header = (
        f"{'attack':>20} {'n':>5} {'raw_data':>8} {'oracleK':>7} "
        f"{'crc_ex':>7} {'valid':>7} {'false':>7} {'crc/raw':>7} "
        f"{'orc_bit':>8} {'IoU':>6} {'AUC':>7} {'TPR@1%':>7}"
    )
    print(header)
    rows = []
    for attack in attacks:
        path = outdir / f"rows_{attack}.csv"
        if not path.exists():
            continue
        pos_rows, pos_det, neg_det = [], [], []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["kind"] == "pos":
                    pos_rows.append(row)
                    pos_det.append(float(row["det"]))
                else:
                    neg_det.append(float(row["det"]))
        if not pos_rows:
            continue
        n = len(pos_rows)
        ag = lambda key: sum(float(row[key]) for row in pos_rows) / n
        auc, t1, t01 = E.roc_metrics(pos_det, neg_det) if neg_det else (float("nan"),) * 3
        item = dict(
            attack=attack,
            n=n,
            raw_payload_bit=ag("raw_payload_bit"),
            raw_payload_exact=ag("raw_payload_exact"),
            raw_data_bit=ag("raw_data_bit"),
            raw_data_exact=ag("raw_data_exact"),
            oracle_topk=ag("oracle_topk"),
            crc_valid=ag("crc_valid"),
            crc_bit=ag("crc_bit"),
            crc_exact=ag("crc_exact"),
            false_accept=ag("false_accept"),
            crc_or_raw_bit=ag("crc_or_raw_bit"),
            crc_or_raw_exact=ag("crc_or_raw_exact"),
            oracle_bit=ag("obit"),
            oracle_exact=ag("oexact"),
            iou=ag("iou"),
            auc=auc,
            tpr1=t1,
            tpr01=t01,
        )
        rows.append(item)
        print(
            f"{attack:>20} {n:>5} {item['raw_data_exact']:>8.3f} {item['oracle_topk']:>7.3f} "
            f"{item['crc_exact']:>7.3f} {item['crc_valid']:>7.3f} {item['false_accept']:>7.3f} "
            f"{item['crc_or_raw_exact']:>7.3f} {item['oracle_bit']:>8.4f} {item['iou']:>6.3f} "
            f"{auc:>7.4f} {t1:>7.3f}"
        )

    if not rows:
        print("no rows to summarize")
        return
    with (outdir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {outdir / 'summary.csv'}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=PATHS.runs / "maskhead_ckpt" / "train2017_a128_classical_FINAL.pt")
    ap.add_argument("--pos", type=Path, default=PATHS.runs / "crc_listdecode_full" / "data32_crc16_k3_n5000_a128" / "pos.pt")
    ap.add_argument("--neg", type=Path, default=PATHS.runs / "composites_val5k" / "neg.pt")
    ap.add_argument("--outdir", type=Path, default=PATHS.runs / "per_attack_crc32_val5k")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attacks", default=",".join(E.ATTACK_SUITE))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-pos", type=int, default=5000)
    ap.add_argument("--max-neg", type=int, default=2000)
    ap.add_argument("--start-pos", type=int, default=0)
    ap.add_argument("--start-neg", type=int, default=0)
    ap.add_argument("--skip-pos", action="store_true")
    ap.add_argument("--skip-neg", action="store_true")
    ap.add_argument("--attack-seed", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=25)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    attacks = [attack.strip() for attack in args.attacks.split(",") if attack.strip()]

    if args.summarize:
        summarize(args.outdir, attacks)
        return

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    model = MaskHead(in_ch=ck["meta"]["in_ch"], base=ck["base"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    wins, stride = ck["meta"]["margin_wins"], ck["meta"]["margin_stride"]

    pos = torch.load(args.pos, map_location="cpu", mmap=True)
    neg = torch.load(args.neg, map_location="cpu", mmap=True)
    key = CRC.make_key(pos["meta"], device)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(PATHS.sd21_model),
        local_files_only=True,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inv = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    for attack_index, attack in enumerate(attacks):
        run_attack(
            pipe,
            inv,
            model,
            pos,
            neg,
            key,
            attack,
            device,
            wins,
            stride,
            args.topk,
            args.outdir / f"rows_{attack}.csv",
            args.flush_every,
            args.max_pos,
            args.max_neg,
            ck["meta"],
            args.start_pos,
            args.start_neg,
            args.skip_pos,
            args.skip_neg,
            args.attack_seed,
            attack_index,
        )


if __name__ == "__main__":
    main()
