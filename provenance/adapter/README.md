# ARIS V10.1 — LoRA Adapter

The adapter weights (`adapter_model.safetensors`, 295,488,936 bytes / ~295 MB)
are committed **to this repository** at:

    provenance/adapter/adapter_model.safetensors

They are stored using **Git LFS**, tracked via the repo-root `.gitattributes`
rule:

    provenance/adapter/adapter_model.safetensors filter=lfs diff=lfs merge=lfs -text

A pinned Hugging Face mirror is also published for convenience but is **not**
the authoritative artifact.

## Identity

- File: `adapter_model.safetensors`
- Size: 295,488,936 bytes (~295 MB)
- Format: safetensors (fp32)
- SHA-256: `b5500b5603e79d055eb1baafc9d1cf4c234c2378cba9d8478ea4de974a677832`

## How to obtain the adapter

A normal clone with Git LFS installed retrieves the weights automatically:

    git clone https://github.com/Vicgrace01/ARIS.git
    cd ARIS
    git lfs pull
    ls -l provenance/adapter/adapter_model.safetensors   # ~295 MB

If Git LFS is **not** installed, the file appears only as a small pointer
(~130 bytes) and must be retrieved with `git lfs pull`.

## Verify

    sha256sum provenance/adapter/adapter_model.safetensors
    # expect:
    # b5500b5603e79d055eb1baafc9d1cf4c234c2378cba9d8478ea4de974a677832

The committed `adapter_manifest.json` records the same SHA-256 and the
storage path.

## Mirror (non-authoritative)

- Repository: https://huggingface.co/Vicgrace/ARIS-V10.1
- Revision:   `894c24fb6c4143acda0b3e54e26ef44b962e67a5`
- Direct URL: https://huggingface.co/Vicgrace/ARIS-V10.1/resolve/894c24fb6c4143acda0b3e54e26ef44b962e67a5/adapter_model.safetensors

The mirror is pinned to a specific revision. The authoritative artifact for
provenance and judging is the in-repo Git LFS object above.

## What is committed here

- `adapter_model.safetensors` — the LoRA weights (Git LFS, ~295 MB)
- `adapter_config.json` — the LoRA structure (r=64, alpha=128, target modules)
- `adapter_manifest.json` — SHA-256, size, storage path, verification command
- `README.md` — this file

## What proves the training run happened

The adapter is one of several §3.1 proof-of-training artifacts. In this
repository you will also find, all directly committed:

- `training_loss_log.csv`, `val_loss_log.csv` — per-step loss
- `loss_curves.png` — visualisation
- `training_args.json`, `training_summary.json` — hyperparameters
- `merge_and_quantize.py` — the merge + GGUF conversion script
- `dataset_card.md`, `dataset_sample.jsonl` — corpus description
- `dataset/` — canonical claims, blacklist, conditional register
- `checksums.txt` — SHA-256 of base model files, adapter, final GGUF
