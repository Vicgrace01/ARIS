# ARIS V10.1 — LoRA Adapter

The adapter weights (`adapter_model.safetensors`, 295.5 MB)
are hosted on HuggingFace Hub and are **not** committed to this repository
because they exceed GitHub's 100 MB per-file limit.

## Where to find it

- Repository: https://huggingface.co/Vicgrace/ARIS-V10.1
- Revision:   `894c24fb6c4143acda0b3e54e26ef44b962e67a5`
- Direct URL: https://huggingface.co/Vicgrace/ARIS-V10.1/resolve/894c24fb6c4143acda0b3e54e26ef44b962e67a5/adapter_model.safetensors

## Verify

The committed `adapter_manifest.json` contains the SHA-256 of the hosted
file. To verify:

```bash
curl -L -o adapter_model.safetensors "https://huggingface.co/Vicgrace/ARIS-V10.1/resolve/894c24fb6c4143acda0b3e54e26ef44b962e67a5/adapter_model.safetensors"
sha256sum adapter_model.safetensors
# expect: b5500b5603e79d055eb1baafc9d1cf4c234c2378cba9d8478ea4de974a677832
```

## What is committed here

- `adapter_config.json` — the LoRA structure (r=64, alpha=128, target modules)
- `adapter_manifest.json` — SHA-256, size, HF URL, verification command

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
