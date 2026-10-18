# LoProMark Code

Core implementation of region-bound latent watermarking for partial diffusion
editing. The diffusion backbone is frozen Stable Diffusion 2.1-Base at 512x512.

## Layout

- `lopromark/` library package
  - `config.py` paths and model locations, edit `sd21_model` for your setup
  - `messages.py` identifier sampling
  - `mask_head.py` compact two-scale U-Net mask head, loss, input normalization
  - `attacks.py`, `attack_protocol.py` the 14 non-regenerative transforms and
    the attack protocol
  - `metrics.py`, `losses.py`, `data.py`, `models.py`, `vae_feature.py` shared
    utilities
- `scripts/` pipeline entry points
  - `carrier_core.py` keyed carrier bank, codebook construction, carrier
    injection, and DDIM inversion helpers, loaded by later scripts
  - `decode_helpers.py`, `blended_inpaint.py`, `mask_gate.py` shared helpers
    for decoding, blended-latent partial generation, and mask gating
  - `build_composites.py` build marked partial composites, embedding side
  - `build_maskhead_dataset.py` build the classical-transform training cache
    and the four codebook-margin maps, windows 6, 10, 16, 22 at stride 2
  - `train_maskhead.py` train the mask head, 80/20 split, seed 2026
  - `eval_per_attack.py` per-attack evaluation core
  - `crc_listdecode.py` CRC-guided soft list decoding, top-3 symbol candidates,
    at most 729 tuples
  - `run_crc_per_attack.py` main verification benchmark, resumable per-attack
    evaluation with the CRC-or-raw decode policy
  - `eval_blind_multiregion.py` blind multi-identifier evaluation, K = 2, 3
  - `object_mask_benchmark.py` paired object-mask protocol
  - `merge_eval_shards.py` merge the five disjoint evaluation shards
- `configs/sd21_local.json` model configuration

## Pipeline

1. Place Stable Diffusion 2.1-Base weights at the path in `lopromark/config.py`
   and COCO val2017 images under `data/val2017`.
2. Build marked composites with `scripts/build_composites.py`.
3. Build the mask-head training cache with `scripts/build_maskhead_dataset.py`.
4. Train the mask head with `scripts/train_maskhead.py`.
5. Run the per-attack benchmark with `scripts/run_crc_per_attack.py`.
   Shards merge with `scripts/merge_eval_shards.py`.
6. Multi-identifier and object-mask protocols run through
   `scripts/eval_blind_multiregion.py` and `scripts/object_mask_benchmark.py`.

Each script exposes `--help` with its full argument list. Seeds and shard
indexing follow the protocol in the supplementary material.
