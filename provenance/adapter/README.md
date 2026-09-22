# ARIS Adapter — Provenance

The LoRA adapter weights for ARIS are committed **in this repository** at:

    provenance/adapter/adapter_model.safetensors

They are stored using **Git LFS** (tracked via the repo-root `.gitattributes`).
This removes any dependency on an external mirror for the authoritative artifact.

## Identity

- File: `adapter_model.safetensors`
- Size: 295,488,936 bytes (~295 MB)
- SHA-256: `b5500b5603e79d055eb1baafc9d1cf4c234c2378cba9d8478ea4de974a677832`

## How to obtain the adapter

A normal clone with Git LFS installed retrieves the weights automatically:

    git clone https://github.com/Vicgrace01/ARIS.git
    cd ARIS
    git lfs pull
    ls -l provenance/adapter/adapter_model.safetensors   # ~295 MB

If Git LFS is **not** installed, the file appears only as a small pointer
(~130 bytes) and must be pulled with `git lfs pull`.

## Mirror (non-authoritative)

A Hugging Face mirror is also published for convenience:

- Repository: https://huggingface.co/Vicgrace/ARIS-V10.1
- Direct URL: https://huggingface.co/Vicgrace/ARIS-V10.1/resolve/894c24fb6c4143acda0b3e54e26ef44b962e67a5/adapter_model.safetensors

The mirror is pinned to a specific revision. The authoritative artifact for
provenance and judging is the in-repo Git LFS object above.

## Verification

    sha256sum provenance/adapter/adapter_model.safetensors
    # expected:
    # b5500b5603e79d055eb1baafc9d1cf4c234c2378cba9d8478ea4de974a677832
