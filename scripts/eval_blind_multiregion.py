"""Blind multi-region instance discovery and payload association.

Inference sees only the received image and the verifier key and
metadata. It decodes every predicted connected component once, rejects
CRC-invalid components, and unites masks that decode to the same
identifier.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lopromark.config import PATHS
from lopromark.mask_head import MaskHead, normalize_input


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


P = _load("scripts/carrier_core.py", "mr_carrier_core")
B = _load("scripts/build_maskhead_dataset.py", "mr_maskhead_dataset")
E = _load("scripts/eval_per_attack.py", "mr_per_attack_eval")
CRC = _load("scripts/crc_listdecode.py", "mr_crc_listdecode")


IMAGE_FIELDS = [
    "attack",
    "idx",
    "n_gt",
    "n_pred",
    "matched_01",
    "matched_05",
    "missed_01",
    "spurious_01",
    "component_precision_01",
    "component_recall_01",
    "merge_count_01",
    "split_count_01",
    "union_iou",
    "mean_matched_iou",
    "per_gt_e2e_exact",
    "all_gt_exact",
    "strict_image_exact",
    "spurious_crc_valid",
]

GT_FIELDS = [
    "attack",
    "idx",
    "gt_region",
    "matched_pred",
    "matched_iou",
    "matched_01",
    "matched_05",
    "pred_crc_valid",
    "id_exact",
    "e2e_exact",
    "true_id",
    "decoded_id",
]

PRED_FIELDS = [
    "attack",
    "idx",
    "pred_component",
    "area",
    "crc_valid",
    "crc_candidate_score",
    "decoded_id",
    "matched_gt",
    "matched_iou",
    "matched_01",
    "matched_05",
    "is_spurious_01",
]


def bits_to_string(bits: torch.Tensor) -> str:
    return "".join(str(int(bit)) for bit in bits.detach().cpu().flatten().tolist())


def connected_components(mask: torch.Tensor, min_pixels: int) -> list[torch.Tensor]:
    """4-connected components of the predicted mask, sorted by area."""
    arr = mask.detach().cpu().bool()
    height, width = arr.shape
    seen = torch.zeros_like(arr)
    out: list[torch.Tensor] = []
    for y in range(height):
        for x in range(width):
            if not bool(arr[y, x]) or bool(seen[y, x]):
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and bool(arr[ny, nx])
                        and not bool(seen[ny, nx])
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) >= min_pixels:
                component = torch.zeros_like(arr)
                for py, px in pixels:
                    component[py, px] = True
                out.append(component)
    out.sort(key=lambda component: int(component.sum()), reverse=True)
    return out


def pairwise_iou(pred: list[torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros((len(pred), int(gt.shape[0])), dtype=torch.float32)
    for p_idx, p_mask in enumerate(pred):
        p_bool = p_mask.bool()
        for g_idx in range(int(gt.shape[0])):
            g_bool = gt[g_idx].bool().cpu()
            inter = (p_bool & g_bool).sum().float()
            union = (p_bool | g_bool).sum().float().clamp_min(1)
            matrix[p_idx, g_idx] = inter / union
    return matrix


def union_iou(pred_mask: torch.Tensor, gt_masks: torch.Tensor) -> float:
    gt_union = gt_masks.bool().any(dim=0).cpu()
    pred_union = pred_mask.bool().cpu()
    inter = (gt_union & pred_union).sum().float()
    union = (gt_union | pred_union).sum().float().clamp_min(1)
    return float(inter / union)


def decode_component(
    recovered: torch.Tensor,
    soft_prob: torch.Tensor,
    component: torch.Tensor,
    carriers: torch.Tensor,
    codebook: torch.Tensor,
    cfg: CRC.Config,
    chunk: int,
    topk: int,
    device: torch.device,
) -> dict:
    component_device = component[None, None].float().to(device)
    weight = soft_prob * component_device
    corr = E._corr(recovered, weight, carriers)
    scores = CRC.codebook_scores(corr, codebook)
    selected, score = CRC.select_crc_candidate(scores, topk, cfg, chunk, device)
    return {
        "crc_valid": int(selected is not None),
        "decoded_id": bits_to_string(selected[: cfg.data_bits]) if selected is not None else "",
        "score": score,
        "area": float(component.float().mean()),
    }


def frozen_hungarian(iou_matrix: torch.Tensor) -> tuple[dict[int, tuple[int, float]], dict[int, tuple[int, float]]]:
    """One-to-one assignment by mask IoU."""
    pred_to_gt: dict[int, tuple[int, float]] = {}
    gt_to_pred: dict[int, tuple[int, float]] = {}
    if iou_matrix.numel() == 0:
        return pred_to_gt, gt_to_pred
    rows, cols = linear_sum_assignment((1.0 - iou_matrix).numpy())
    for pred_idx, gt_idx in zip(rows.tolist(), cols.tolist()):
        value = float(iou_matrix[pred_idx, gt_idx])
        pred_to_gt[pred_idx] = (gt_idx, value)
        gt_to_pred[gt_idx] = (pred_idx, value)
    return pred_to_gt, gt_to_pred


def aggregate(image_rows: list[dict], gt_rows: list[dict], pred_rows: list[dict], out: Path) -> None:
    n_images = len(image_rows)
    n_gt = len(gt_rows)
    n_pred = len(pred_rows)
    mean = lambda rows, key: sum(float(row[key]) for row in rows) / max(1, len(rows))
    summary = {
        "attack": image_rows[0]["attack"],
        "images": n_images,
        "gt_regions": n_gt,
        "pred_components": n_pred,
        "component_precision_01": sum(int(row["matched_01"]) for row in pred_rows) / max(1, n_pred),
        "component_precision_05": sum(int(row["matched_05"]) for row in pred_rows) / max(1, n_pred),
        "component_recall_01": sum(int(row["matched_01"]) for row in gt_rows) / max(1, n_gt),
        "component_recall_05": sum(int(row["matched_05"]) for row in gt_rows) / max(1, n_gt),
        "matched_iou_over_matches": (
            sum(float(row["matched_iou"]) for row in gt_rows if int(row["matched_01"]))
            / max(1, sum(int(row["matched_01"]) for row in gt_rows))
        ),
        "per_gt_id_exact_given_match": (
            sum(int(row["id_exact"]) for row in gt_rows if int(row["matched_01"]))
            / max(1, sum(int(row["matched_01"]) for row in gt_rows))
        ),
        "per_gt_end_to_end_exact": sum(int(row["e2e_exact"]) for row in gt_rows) / max(1, n_gt),
        "image_all_gt_exact": mean(image_rows, "all_gt_exact"),
        "strict_image_exact": mean(image_rows, "strict_image_exact"),
        "union_iou": mean(image_rows, "union_iou"),
        "mean_missed_01": mean(image_rows, "missed_01"),
        "mean_spurious_01": mean(image_rows, "spurious_01"),
        "mean_merge_count_01": mean(image_rows, "merge_count_01"),
        "mean_split_count_01": mean(image_rows, "split_count_01"),
        "spurious_crc_valid_rate": (
            sum(int(row["crc_valid"]) for row in pred_rows if int(row["is_spurious_01"]))
            / max(1, sum(int(row["is_spurious_01"]) for row in pred_rows))
        ),
    }
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(summary, flush=True)


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_attack(
    args: argparse.Namespace,
    attack: str,
    comp: dict,
    pipe: StableDiffusionPipeline,
    inv: DDIMInverseScheduler,
    model: MaskHead,
    carriers: torch.Tensor,
    codebook: torch.Tensor,
    cfg: CRC.Config,
    wins: list[int],
    stride: int,
    device: torch.device,
) -> None:
    image_path = args.outdir / f"per_image_{attack}.csv"
    gt_path = args.outdir / f"per_gt_{attack}.csv"
    pred_path = args.outdir / f"per_pred_{attack}.csv"
    summary_path = args.outdir / f"summary_{attack}.csv"
    for path in (image_path, gt_path, pred_path, summary_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {path}")

    image_rows: list[dict] = []
    gt_rows: list[dict] = []
    pred_rows: list[dict] = []
    count = min(args.num_samples, int(comp["images"].shape[0]))
    meta = comp["meta"]
    for idx in range(count):
        image = comp["images"][idx : idx + 1].float().to(device) / 255.0
        torch.manual_seed(args.attack_seed + idx)
        attacked = E.apply_attack(
            pipe,
            attack,
            image,
            meta["prompt"],
            meta["steps"],
            meta["guidance_scale"],
        ).to(torch.float16)
        recovered = P.invert_image(
            pipe,
            inv,
            attacked,
            meta["prompt"],
            meta["steps"],
            meta["guidance_scale"],
        ).float()
        margins = B.margin_maps_fast(
            recovered, carriers, codebook, meta["H"], meta["W"], wins, stride
        )[None]
        soft_prob = torch.sigmoid(model(normalize_input(torch.cat([recovered, margins], dim=1))))
        binary_union = (soft_prob > args.mask_threshold)[0, 0]

        # Decode each predicted component.
        pred_masks = connected_components(binary_union, args.min_component_pixels)
        predictions = [
            decode_component(
                recovered,
                soft_prob,
                mask,
                carriers,
                codebook,
                cfg,
                meta["chunk"],
                args.topk,
                device,
            )
            for mask in pred_masks
        ]

        # Compare with ground truth.
        gt_masks = comp["region_masks"][idx].bool().cpu()
        gt_ids = [bits_to_string(bits) for bits in comp["data_payloads"][idx]]
        ious = pairwise_iou(pred_masks, gt_masks)
        pred_to_gt, gt_to_pred = frozen_hungarian(ious)
        matched01 = sum(value >= args.match_iou_min for _, value in gt_to_pred.values())
        matched05 = sum(value >= 0.5 for _, value in gt_to_pred.values())
        merge_count = sum(int((ious[pred_idx] >= args.match_iou_min).sum()) >= 2 for pred_idx in range(len(pred_masks)))
        split_count = sum(int((ious[:, gt_idx] >= args.match_iou_min).sum()) >= 2 for gt_idx in range(len(gt_ids)))

        gt_exact_flags: list[int] = []
        matched_iou_values: list[float] = []
        for gt_idx, true_id in enumerate(gt_ids):
            pred_idx, match_iou = gt_to_pred.get(gt_idx, (-1, 0.0))
            matched_01 = int(pred_idx >= 0 and match_iou >= args.match_iou_min)
            matched_05 = int(pred_idx >= 0 and match_iou >= 0.5)
            if matched_01:
                prediction = predictions[pred_idx]
                crc_valid = int(prediction["crc_valid"])
                decoded_id = str(prediction["decoded_id"])
                id_exact = int(crc_valid and decoded_id == true_id)
                matched_iou_values.append(match_iou)
            else:
                crc_valid = 0
                decoded_id = ""
                id_exact = 0
            e2e = matched_01 * id_exact
            gt_exact_flags.append(e2e)
            gt_rows.append(
                {
                    "attack": attack,
                    "idx": idx,
                    "gt_region": gt_idx,
                    "matched_pred": pred_idx,
                    "matched_iou": match_iou,
                    "matched_01": matched_01,
                    "matched_05": matched_05,
                    "pred_crc_valid": crc_valid,
                    "id_exact": id_exact,
                    "e2e_exact": e2e,
                    "true_id": true_id,
                    "decoded_id": decoded_id,
                }
            )

        spurious_crc_valid = 0
        for pred_idx, prediction in enumerate(predictions):
            gt_idx, match_iou = pred_to_gt.get(pred_idx, (-1, 0.0))
            matched_01 = int(gt_idx >= 0 and match_iou >= args.match_iou_min)
            matched_05 = int(gt_idx >= 0 and match_iou >= 0.5)
            is_spurious = 1 - matched_01
            spurious_crc_valid += is_spurious * int(prediction["crc_valid"])
            pred_rows.append(
                {
                    "attack": attack,
                    "idx": idx,
                    "pred_component": pred_idx,
                    "area": prediction["area"],
                    "crc_valid": prediction["crc_valid"],
                    "crc_candidate_score": prediction["score"] if prediction["score"] is not None else "",
                    "decoded_id": prediction["decoded_id"],
                    "matched_gt": gt_idx,
                    "matched_iou": match_iou,
                    "matched_01": matched_01,
                    "matched_05": matched_05,
                    "is_spurious_01": is_spurious,
                }
            )

        all_gt_exact = int(bool(gt_exact_flags) and all(gt_exact_flags))
        strict_exact = int(all_gt_exact and len(predictions) == len(gt_ids))
        image_rows.append(
            {
                "attack": attack,
                "idx": idx,
                "n_gt": len(gt_ids),
                "n_pred": len(predictions),
                "matched_01": matched01,
                "matched_05": matched05,
                "missed_01": len(gt_ids) - matched01,
                "spurious_01": len(predictions) - matched01,
                "component_precision_01": matched01 / max(1, len(predictions)),
                "component_recall_01": matched01 / max(1, len(gt_ids)),
                "merge_count_01": merge_count,
                "split_count_01": split_count,
                "union_iou": union_iou(binary_union, gt_masks),
                "mean_matched_iou": sum(matched_iou_values) / max(1, len(matched_iou_values)),
                "per_gt_e2e_exact": sum(gt_exact_flags) / max(1, len(gt_exact_flags)),
                "all_gt_exact": all_gt_exact,
                "strict_image_exact": strict_exact,
                "spurious_crc_valid": spurious_crc_valid,
            }
        )
        if (idx + 1) % 10 == 0:
            print(f"[{attack}] {idx + 1}/{count}", flush=True)

    write_rows(image_path, IMAGE_FIELDS, image_rows)
    write_rows(gt_path, GT_FIELDS, gt_rows)
    write_rows(pred_path, PRED_FIELDS, pred_rows)
    aggregate(image_rows, gt_rows, pred_rows, summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=PATHS.runs / "maskhead_ckpt" / "train2017_a128_classical_FINAL.pt",
    )
    parser.add_argument("--composites", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attacks", default="identity,diffusion_regen_s0.25")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument("--match-iou-min", type=float, default=0.1)
    parser.add_argument("--attack-seed", type=int, default=890001)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    comp = torch.load(args.composites, map_location="cpu", mmap=True)
    meta = comp["meta"]

    manifest_path = args.outdir / "config.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    manifest = {
        "composites": str(args.composites),
        "checkpoint": str(args.ckpt),
        "meta": meta,
        "attacks": [item.strip() for item in args.attacks.split(",") if item.strip()],
        "num_samples": args.num_samples,
        "mask_threshold": args.mask_threshold,
        "min_component_pixels": args.min_component_pixels,
        "match_iou_min": args.match_iou_min,
        "topk": args.topk,
        "attack_seed": args.attack_seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cfg = CRC.Config(meta["data_bits"], meta["crc_bits"], (args.topk,))
    carriers, codebook = CRC.make_key(meta, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = MaskHead(in_ch=ckpt["meta"]["in_ch"], base=ckpt["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pipe = StableDiffusionPipeline.from_pretrained(
        str(PATHS.sd21_model),
        local_files_only=True,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    inv = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    for attack in [item.strip() for item in args.attacks.split(",") if item.strip()]:
        evaluate_attack(
            args,
            attack,
            comp,
            pipe,
            inv,
            model,
            carriers,
            codebook,
            cfg,
            ckpt["meta"]["margin_wins"],
            ckpt["meta"]["margin_stride"],
            device,
        )


if __name__ == "__main__":
    main()
