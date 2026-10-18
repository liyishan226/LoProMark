"""Merge sharded per-attack evaluation rows and check index coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NUMERIC = [
    "raw_payload_bit", "raw_payload_exact", "raw_data_bit", "raw_data_exact",
    "oracle_topk", "crc_valid", "crc_bit", "crc_exact", "false_accept",
    "crc_or_raw_bit", "crc_or_raw_exact", "obit", "oexact", "iou", "det", "area",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--attacks", default="identity,diffusion_regen_s0.25")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--attack-seed", type=int, default=1320001)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    sources = []
    summary = []

    for attack_index, attack in enumerate(attacks):
        files = sorted(args.root.glob(f"shard*/rows_{attack}.csv"))
        if not files:
            raise FileNotFoundError(f"no shard rows for {attack}")
        rows = []
        for path in files:
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row["kind"] != "pos":
                        raise ValueError(f"unexpected non-positive row in {path}")
                    rows.append(row)
            sources.append(str(path))
        counts = {}
        for row in rows:
            idx = int(row["idx"])
            counts[idx] = counts.get(idx, 0) + 1
            expected_seed = args.attack_seed + attack_index * 1_000_003 + idx
            if int(row["attack_seed"]) != expected_seed:
                raise ValueError(f"seed mismatch {attack}/{idx}: {row['attack_seed']} != {expected_seed}")
        expected = set(range(args.n))
        got = set(counts)
        if got != expected or any(value != 1 for value in counts.values()):
            raise ValueError(
                f"coverage failure {attack}: missing={len(expected-got)} extra={len(got-expected)} "
                f"duplicates={sum(v-1 for v in counts.values() if v>1)}"
            )
        rows.sort(key=lambda row: int(row["idx"]))
        combined = args.outdir / f"rows_{attack}.csv"
        with combined.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        item = {"attack": attack, "n": len(rows)}
        for key in NUMERIC:
            item[key] = sum(float(row[key]) for row in rows) / len(rows)
        summary.append(item)

    with (args.outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    manifest = {
        "n": args.n, "attacks": attacks, "attack_seed": args.attack_seed,
        "seed_formula": "attack_seed + attack_index*1000003 + global_positive_index",
        "source_files": sources, "summary": summary,
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
