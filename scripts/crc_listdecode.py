"""CRC-guided soft list decoding for region provenance payloads.

Scores all codewords per symbol, searches candidate tuples in score
order, and returns the best CRC-valid tuple.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
M = _load("scripts/blended_inpaint.py", "blended_inpaint")
C = _load("scripts/mask_gate.py", "mask_gate")
E = _load("scripts/eval_per_attack.py", "per_attack_eval")


CRC_POLYS = {
    8: 0x07,       # CRC-8
    16: 0x1021,   # CRC-16/CCITT style polynomial
    24: 0x864CFB, # CRC-24/OpenPGP polynomial
}


@dataclass(frozen=True)
class Config:
    data_bits: int
    crc_bits: int
    topks: tuple[int, ...]

    @property
    def total_bits(self) -> int:
        return self.data_bits + self.crc_bits

    @property
    def name(self) -> str:
        ks = "-".join(str(k) for k in self.topks)
        return f"data{self.data_bits}_crc{self.crc_bits}_k{ks}"


def parse_configs(value: str) -> list[Config]:
    out = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad config '{item}', expected data_bits:crc_bits:topk[,topk].")
        topks = tuple(int(x) for x in parts[2].split(",") if x.strip())
        out.append(Config(int(parts[0]), int(parts[1]), topks))
    if not out:
        raise ValueError("No configs were provided.")
    return out


def crc_remainder(bits: list[int], width: int) -> int:
    if width not in CRC_POLYS:
        raise ValueError(f"Unsupported CRC width {width}; supported: {sorted(CRC_POLYS)}")
    poly = CRC_POLYS[width]
    reg = 0
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    for bit in bits:
        reg ^= (int(bit) & 1) << (width - 1)
        reg = ((reg << 1) ^ poly) & mask if (reg & top) else (reg << 1) & mask
    return reg


def crc_bits_for(data_bits: list[int], width: int) -> list[int]:
    value = crc_remainder(data_bits, width)
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def make_crc_payload(data_bits: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
    data = [int(x) for x in data_bits.detach().cpu().view(-1).tolist()]
    payload = data + crc_bits_for(data, width)
    return torch.tensor(payload, dtype=torch.float32, device=device)


def valid_crc(payload_bits: list[int], data_bits: int, crc_width: int) -> bool:
    data = payload_bits[:data_bits]
    got = payload_bits[data_bits : data_bits + crc_width]
    return got == crc_bits_for(data, crc_width)


def codebook_scores(code_corr: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    code_bits = codebook.shape[-1]
    blocks = code_corr.view(*code_corr.shape[:-1], -1, code_bits)
    signs = codebook.mul(2.0).sub(1.0)
    scores = torch.einsum("brnm,cm->brnc", blocks, signs)
    return scores[0, 0]


def chunk_indices(payload_bits: torch.Tensor, chunk_bits: int) -> torch.Tensor:
    chunks = payload_bits.long().view(-1, chunk_bits)
    weights = (2 ** torch.arange(chunk_bits - 1, -1, -1, device=payload_bits.device)).long()
    return (chunks * weights).sum(-1)


def payload_from_indices(indices: torch.Tensor, chunk_bits: int, device: torch.device) -> torch.Tensor:
    data = P.binary_data_book(chunk_bits, device)
    return data[indices.long()].flatten().float()


_CHOICE_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_BITBOOK_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def device_key(device: torch.device) -> str:
    return f"{device.type}:{device.index if device.index is not None else -1}"


def choice_matrix(n_chunks: int, topk: int, device: torch.device) -> torch.Tensor:
    key = (n_chunks, topk, device_key(device))
    cached = _CHOICE_CACHE.get(key)
    if cached is not None:
        return cached
    grids = [torch.arange(topk, device=device) for _ in range(n_chunks)]
    choices = torch.cartesian_prod(*grids).long()
    _CHOICE_CACHE[key] = choices
    return choices


def bit_book_tensor(chunk_bits: int, device: torch.device) -> torch.Tensor:
    key = (chunk_bits, device_key(device))
    cached = _BITBOOK_CACHE.get(key)
    if cached is not None:
        return cached
    book = P.binary_data_book(chunk_bits, device).long()
    _BITBOOK_CACHE[key] = book
    return book


def crc_valid_batch(payload_bits: torch.Tensor, data_bits: int, crc_width: int) -> torch.Tensor:
    """Vectorized CRC check for payload_bits shaped [N, total_bits]."""
    if crc_width not in CRC_POLYS:
        raise ValueError(f"Unsupported CRC width {crc_width}; supported: {sorted(CRC_POLYS)}")
    bits = payload_bits.long()
    poly = CRC_POLYS[crc_width]
    mask = (1 << crc_width) - 1
    top = 1 << (crc_width - 1)
    reg = torch.zeros(bits.shape[0], dtype=torch.long, device=bits.device)
    for pos in range(data_bits):
        reg = reg ^ (bits[:, pos] << (crc_width - 1))
        do_xor = (reg & top) != 0
        shifted = (reg << 1) & mask
        reg = torch.where(do_xor, (shifted ^ poly) & mask, shifted)

    got = torch.zeros(bits.shape[0], dtype=torch.long, device=bits.device)
    for pos in range(crc_width):
        got = got | (bits[:, data_bits + pos] << (crc_width - 1 - pos))
    return reg == got


def make_key(meta: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    outer = meta["payload_bits"] // meta["chunk"] * meta["code_bits"]
    cb = P.make_binary_codebook(meta["chunk"], meta["code_bits"], meta["seed"] + 571, device)
    ph = P.make_carriers(outer, 4, meta["H"], meta["W"], meta["seed"] + 17, device)
    return ph, cb


@torch.no_grad()
def build_composites(
    pipe: StableDiffusionPipeline,
    cfg: Config,
    args: argparse.Namespace,
    device: torch.device,
    out_path: Path,
) -> dict:
    if out_path.exists() and not args.force_build:
        print(f"reuse {out_path}", flush=True)
        return torch.load(out_path, map_location="cpu", mmap=True)

    if cfg.total_bits % args.codebook_chunk_bits:
        raise ValueError(f"{cfg.name}: total bits must be divisible by chunk bits.")

    random.seed(args.seed + args.cover_offset)
    torch.manual_seed(args.seed + args.cover_offset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H = args.size // pipe.vae_scale_factor
    W = args.size // pipe.vae_scale_factor
    chunk = args.codebook_chunk_bits
    outer_bits = cfg.total_bits // chunk * args.codebook_code_bits
    codebook = P.make_binary_codebook(chunk, args.codebook_code_bits, args.seed + 571, device)
    carriers = P.make_carriers(outer_bits, pipe.unet.config.in_channels, H, W, args.seed + 17, device)
    covers = sorted(args.cover_dir.glob("*.jpg"))
    if not covers:
        raise FileNotFoundError(f"No jpg covers found under {args.cover_dir}")

    images, masks, payloads, data_payloads, manifest_rows = [], [], [], [], []
    t0 = time.time()
    for i in range(args.num_samples):
        cover_index = (args.cover_offset + i) % len(covers)
        cover_path = covers[cover_index]
        # Independent sample seeding makes every row reproducible without replaying
        # the RNG history of the preceding samples.
        sample_seed = args.seed + 10_000_019 + args.cover_offset * 1_000_003 + i
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sample_seed)
        cover = M.load_cover(cover_path, args.size, device)
        z_real = P.encode_image_to_latent(pipe, cover.to(torch.float16)).float()
        frac = random.uniform(args.frac_min, args.frac_max)
        edit_mask = C.irregular_mask(frac, H, W, device)

        data_gen = torch.Generator(device="cpu")
        data_gen.manual_seed(args.seed + 1000003 + args.cover_offset * 1000 + i)
        data_cpu = torch.randint(0, 2, (cfg.data_bits,), generator=data_gen, dtype=torch.float32)
        payload_1d = make_crc_payload(data_cpu, cfg.crc_bits, device)
        payload = payload_1d.view(1, 1, -1)

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

        x_t = edit_mask * (wm * init_sigma) + (1.0 - edit_mask) * pipe.scheduler.add_noise(
            z_real, eps, pipe.scheduler.timesteps[0].unsqueeze(0)
        )
        composite = P.tensor_to_bchw(
            pipe(
                prompt=[args.prompt],
                height=args.size,
                width=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                latents=x_t.to(torch.float16),
                output_type="pt",
                callback_on_step_end=make_cb(edit_mask, z_real, eps),
                callback_on_step_end_tensor_inputs=["latents"],
            ).images
        )

        image_cpu = (composite[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        mask_cpu = edit_mask[0].bool().cpu()
        payload_cpu = payload_1d.bool().cpu()
        data_cpu_bool = data_cpu.bool().cpu()
        images.append(image_cpu)
        masks.append(mask_cpu)
        payloads.append(payload_cpu)
        data_payloads.append(data_cpu_bool)
        manifest_rows.append({
            "index": i,
            "cover_index": cover_index,
            "cover_file": cover_path.name,
            "sample_seed": sample_seed,
            "data_seed": args.seed + 1_000_003 + args.cover_offset * 1000 + i,
            "support_fraction": float(mask_cpu.float().mean()),
        })
        if (i + 1) % args.progress_every == 0 or (i + 1) == args.num_samples:
            dt = (time.time() - t0) / (i + 1)
            print(f"[build {cfg.name}] {i + 1}/{args.num_samples} ({dt:.2f}s/sample)", flush=True)

    comp = dict(
        images=torch.stack(images),
        masks=torch.stack(masks),
        payloads=torch.stack(payloads),
        data_payloads=torch.stack(data_payloads),
        meta=dict(
            seed=args.seed,
            payload_bits=cfg.total_bits,
            data_bits=cfg.data_bits,
            crc_bits=cfg.crc_bits,
            chunk=chunk,
            code_bits=args.codebook_code_bits,
            alpha=args.alpha,
            size=args.size,
            H=H,
            W=W,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            prompt=args.prompt,
            cover_offset=args.cover_offset,
            sample_seed_formula="seed + 10000019 + cover_offset*1000003 + sample_index",
            data_seed_formula="seed + 1000003 + cover_offset*1000 + sample_index",
        ),
    )
    torch.save(comp, out_path)
    manifest_path = out_path.with_name("build_manifest.csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    config = {
        "command": sys.argv,
        "torch": torch.__version__,
        "device": str(device),
        "config_name": cfg.name,
        "num_samples": args.num_samples,
        "cover_count_available": len(covers),
        "sample_seed_formula": "seed + 10000019 + cover_offset*1000003 + sample_index",
        "data_seed_formula": "seed + 1000003 + cover_offset*1000 + sample_index",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    config_path = out_path.with_name("build_config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out_path} images={tuple(comp['images'].shape)}", flush=True)
    print(f"logged manifest={manifest_path} config={config_path}", flush=True)
    return comp


def select_crc_candidate(
    scores: torch.Tensor,
    topk: int,
    cfg: Config,
    chunk_bits: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, float | None]:
    n_chunks = cfg.total_bits // chunk_bits
    top = scores.topk(topk, dim=-1)
    choices = choice_matrix(n_chunks, topk, device)
    chunk_ids = torch.arange(n_chunks, device=device)[None, :].expand_as(choices)
    candidate_indices = top.indices[chunk_ids, choices]
    candidate_scores = top.values[chunk_ids, choices].sum(dim=-1)
    candidate_bits = bit_book_tensor(chunk_bits, device)[candidate_indices].reshape(choices.shape[0], cfg.total_bits)
    valid = crc_valid_batch(candidate_bits, cfg.data_bits, cfg.crc_bits)
    if not bool(valid.any()):
        return None, None
    masked_scores = candidate_scores.masked_fill(~valid, -float("inf"))
    best = int(masked_scores.argmax().detach().cpu())
    return candidate_bits[best].float(), float(candidate_scores[best].detach().cpu())


@torch.no_grad()
def evaluate_config(
    pipe: StableDiffusionPipeline,
    inv: DDIMInverseScheduler,
    model: MaskHead,
    comp: dict,
    cfg: Config,
    args: argparse.Namespace,
    device: torch.device,
    wins: list[int],
    stride: int,
) -> list[dict]:
    meta = comp["meta"]
    chunk = meta["chunk"]
    ph, cb = make_key(meta, device)
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    rows = []

    for attack in attacks:
        stats = {
            k: dict(
                oracle=0,
                crc_valid=0,
                crc_exact=0,
                crc_bit=0.0,
                false_accept=0,
                fallback_exact=0,
                fallback_bit=0.0,
            )
            for k in cfg.topks
        }
        raw_total_exact = 0
        raw_data_exact = 0
        raw_data_bit = 0.0
        iou_sum = 0.0
        t0 = time.time()

        for i in range(comp["images"].shape[0]):
            img = comp["images"][i : i + 1].float().to(device) / 255.0
            attacked = E.apply_attack(pipe, attack, img, meta["prompt"], meta["steps"], meta["guidance_scale"]).to(torch.float16)
            rec = P.invert_image(pipe, inv, attacked, meta["prompt"], meta["steps"], meta["guidance_scale"]).float()
            mm = B.margin_maps_fast(rec, ph, cb, meta["H"], meta["W"], wins, stride)[None]
            prob = torch.sigmoid(model(normalize_input(torch.cat([rec, mm], dim=1))))
            corr = E._corr(rec, prob, ph)
            scores = codebook_scores(corr, cb)

            raw = P.codebook_decode_soft(corr, cb, chunk)[0, 0]
            true_full = comp["payloads"][i].float().to(device)
            true_data = comp["data_payloads"][i].float().to(device)
            raw_data = raw[: cfg.data_bits]
            raw_total_exact += int(bool((raw == true_full).all()))
            raw_data_exact += int(bool((raw_data == true_data).all()))
            raw_data_bit += float((raw_data == true_data).float().mean())

            pred_mask = (prob > 0.5).float()
            true_mask = comp["masks"][i : i + 1].float().to(device)
            inter = (pred_mask * true_mask).sum()
            union = ((pred_mask + true_mask) > 0).float().sum().clamp_min(1)
            iou_sum += float((inter / union).detach().cpu())

            true_idx = chunk_indices(true_full, chunk)
            max_k = max(cfg.topks)
            top_idx = scores.topk(max_k, dim=-1).indices

            for k in cfg.topks:
                in_topk = (true_idx[:, None] == top_idx[:, :k]).any(-1)
                if bool(in_topk.all()):
                    stats[k]["oracle"] += 1

                selected, _ = select_crc_candidate(scores, k, cfg, chunk, device)
                if selected is None:
                    fallback = raw_data
                else:
                    stats[k]["crc_valid"] += 1
                    selected_data = selected[: cfg.data_bits]
                    bit = float((selected_data == true_data).float().mean())
                    stats[k]["crc_bit"] += bit
                    if bool((selected_data == true_data).all()):
                        stats[k]["crc_exact"] += 1
                    else:
                        stats[k]["false_accept"] += 1
                    fallback = selected_data

                stats[k]["fallback_bit"] += float((fallback == true_data).float().mean())
                stats[k]["fallback_exact"] += int(bool((fallback == true_data).all()))

            if (i + 1) % args.progress_every == 0 or (i + 1) == comp["images"].shape[0]:
                dt = (time.time() - t0) / (i + 1)
                print(f"[eval {cfg.name} {attack}] {i + 1}/{comp['images'].shape[0]} ({dt:.2f}s/sample)", flush=True)

        n = comp["images"].shape[0]
        for k in cfg.topks:
            valid = max(1, stats[k]["crc_valid"])
            rows.append(
                dict(
                    config=cfg.name,
                    data_bits=cfg.data_bits,
                    crc_bits=cfg.crc_bits,
                    total_bits=cfg.total_bits,
                    topk=k,
                    attack=attack,
                    n=n,
                    raw_total_exact=raw_total_exact / n,
                    raw_data_exact=raw_data_exact / n,
                    raw_data_bit=raw_data_bit / n,
                    oracle_topk=stats[k]["oracle"] / n,
                    crc_valid_rate=stats[k]["crc_valid"] / n,
                    crc_exact=stats[k]["crc_exact"] / n,
                    crc_bit_valid=stats[k]["crc_bit"] / valid,
                    false_accept_rate=stats[k]["false_accept"] / n,
                    crc_or_raw_exact=stats[k]["fallback_exact"] / n,
                    crc_or_raw_bit=stats[k]["fallback_bit"] / n,
                    iou=iou_sum / n,
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=PATHS.runs / "maskhead_ckpt" / "train2017_a128_classical_FINAL.pt")
    ap.add_argument("--cover-dir", type=Path, default=Path("data/val2017"))
    ap.add_argument("--out-root", type=Path, default=PATHS.runs / "crc_listdecode_smallset")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--configs", default="56:8:2;48:16:3;32:16:3,5;32:24:5")
    ap.add_argument("--attacks", default="identity,diffusion_regen_s0.25")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--cover-offset", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--prompt", default="a natural photo, realistic, detailed")
    ap.add_argument("--codebook-chunk-bits", type=int, default=8)
    ap.add_argument("--codebook-code-bits", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--frac-min", type=float, default=0.05)
    ap.add_argument("--frac-max", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--progress-every", type=int, default=8)
    args = ap.parse_args()

    configs = parse_configs(args.configs)
    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=device)
    model = MaskHead(in_ch=ck["meta"]["in_ch"], base=ck["base"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    wins = ck["meta"]["margin_wins"]
    stride = ck["meta"]["margin_stride"]

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

    all_rows: list[dict] = []
    for cfg in configs:
        comp_dir = args.out_root / f"{cfg.name}_n{args.num_samples}_a{int(args.alpha)}"
        comp_path = comp_dir / "pos.pt"
        comp = build_composites(pipe, cfg, args, device, comp_path)
        if args.build_only:
            continue
        rows = evaluate_config(pipe, inv, model, comp, cfg, args, device, wins, stride)
        all_rows.extend(rows)
        torch.cuda.empty_cache()

    if args.build_only:
        print("build-only complete")
        return

    out_csv = args.out_root / "summary.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nsaved {out_csv}")
    print(
        "config,attack,K,raw_total,raw_data,oracle,crc_valid,crc_exact,false_accept,crc_or_raw,iou",
        flush=True,
    )
    for row in all_rows:
        print(
            f"{row['config']},{row['attack']},K={row['topk']},"
            f"{row['raw_total_exact']:.3f},{row['raw_data_exact']:.3f},"
            f"{row['oracle_topk']:.3f},{row['crc_valid_rate']:.3f},"
            f"{row['crc_exact']:.3f},{row['false_accept_rate']:.3f},"
            f"{row['crc_or_raw_exact']:.3f},{row['iou']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
