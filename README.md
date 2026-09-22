---
language:
  - en
  - pcm
license: gpl-3.0
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - agriculture
  - nigeria
  - offline
  - edge
  - qlora
  - gguf
base_model: unsloth/Qwen2.5-1.5B-Instruct
datasets:
  - ARIS-V10.1-in-house
metrics:
  - accuracy
model-index:
  - name: ARIS-V10.1-1.5B-Q4_K_M
    results:
      - task:
          type: text-generation
          name: Agricultural Advisory
        dataset:
          name: ARIS Frozen Evaluation (131 records, 8 categories)
          type: custom
        metrics:
          - name: Strict Rubric Accuracy
            type: accuracy
            value: 81.0
          - name: Tolerant Canonical Accuracy
            type: accuracy
            value: 87.9
      - task:
          type: text-generation
          name: General Reasoning
        dataset:
          name: ARC-Easy
          type: arc_easy
        metrics:
          - name: Accuracy (normalized)
            type: acc_norm
            value: 76.0
---

# ARIS V10.1 — Offline Agronomic Advisor for Nigerian Smallholder Farmers

**Africa Deep Tech Challenge 2026 — Laptop LLM track**
**Team ID:** agrigemma

ARIS is an offline, bilingual (English and Nigerian Pidgin) AI agricultural
advisor designed to run entirely on an 8 GB laptop with no internet
connection. It answers crop, pest, and livestock questions grounded in
verified Nigerian agricultural reference material, and refuses to prescribe
fertilizer, pesticide, or veterinary dosages without soil-test and label
context.

The model is a fine-tuned Qwen2.5-1.5B, quantized to GGUF Q4_K_M and
deployed via `llama.cpp`. It is designed for the hardware that actually
exists in Nigerian farming communities — not for a data centre.

---

## Model Details

| | |
|---|---|
| **Model name** | ARIS-V10.1-1.5B-Q4_K_M |
| **Base model** | [`unsloth/Qwen2.5-1.5B-Instruct`](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct) |
| **Base model revision** | `3d254dbee5e3beae81bb8a717ad3a03427a09d26` |
| **Fine-tuning method** | QLoRA (r=64, alpha=128, 2 epochs, response-only loss) |
| **Quantization** | GGUF Q4_K_M |
| **Parameters** | 1.54B |
| **Size on disk** | 986 MB |
| **Context length** | 32,768 tokens |
| **Runtime** | llama.cpp only |
| **Languages** | English (`en`), Nigerian Pidgin (`pcm`) |
| **License** | GPL-3.0 |

---

## ⚠️ Required: The System Prompt

ARIS was trained and evaluated with a specific system prompt prepended to
every conversation. The system prompt defines the model's identity,
capabilities, and refusal behavior. **Running the model without it will
produce different output and is not the configuration the reported
benchmarks were measured on.**

The canonical system prompt is committed at
[`provenance/system_prompt.txt`](provenance/system_prompt.txt). Use it
verbatim in any evaluation.

### The system prompt text

The canonical system prompt is:

```
You are ARIS, an offline agricultural advisor for Nigerian smallholder farmers.

You have:
- No internet access
- No live weather data (only seasonal climate prediction)
- No camera (you cannot see photos)
- No live market prices
- No ability to prescribe veterinary treatment

You answer strictly from verified Nigerian agricultural sources
(NAERLS, IITA, NRCRI, NCRI, NIHORT, NSPRI, NAPRI, NVRI, FMARD, NiMet, state ADPs).

When you do not know, say "I do not know" and refer the farmer to their
ADP extension officer, a veterinarian, or NRCRI as appropriate.
```

### How to run with the system prompt

The `llama-cli` `-p` flag accepts the full ChatML-formatted prompt. The
pattern is:

```
<|im_start|>system
{system prompt from provenance/system_prompt.txt}<|im_end|>
<|im_start|>user
{user question}<|im_end|>
<|im_start|>assistant
```

A complete command that reads the system prompt from the committed file:

```bash
SYS="$(cat provenance/system_prompt.txt)"
llama-cli -m model/ARIS-V10.1-1.5B-Q4_K_M.gguf \
  -p "<|im_start|>system
${SYS}<|im_end|>
<|im_start|>user
My cassava leaves are showing yellow-green mosaic patterns and the plant is stunted. What disease is this and how can I manage it?<|im_end|>
<|im_start|>assistant
" \
  -n 256 --temp 0.0 --threads 4 --no-display-prompt
```

Replace the user line with any question. **Keep the system line identical
across all evaluations** so results are comparable.

---

## Intended Use

ARIS is built for **Nigerian smallholder farmers** and the extension
services that support them. The primary use case is a farmer or field agent
asking an agricultural question — crop disease identification, pest
management, planting practice, soil management, livestock care — and
receiving a grounded, honest answer on a local device with no network.

### Primary uses

- Diagnosing crop diseases from symptom descriptions (cassava, maize, yam,
  rice, tomato, pepper, cocoa, groundnut, cowpea, sorghum, plantain, oil palm)
- Identifying pest pressure and safe management options
- Planting practice guidance (spacing, timing, mounds vs. ridges, setts)
- Post-harvest handling and storage questions
- Livestock and poultry health triage (with referral to veterinarians)
- Answering in English or Nigerian Pidgin, matching the farmer's language

### Intended deployment environment

- ADTC Standard Laptop profile: 8 GB RAM, 4 vCPU (Intel i5 10th–12th gen),
  integrated graphics only
- 100% offline — no network access during inference
- Single-user interactive session via `llama-cli` or a wrapper application
- `temperature = 0.0` for evaluation and safety-critical answers;
  `temperature = 0.7` for conversational field use

---

## Out-of-Scope Use

The following uses are **not** intended and may produce unsafe output:

- **Medical diagnosis or treatment.** ARIS is not a medical device. Any
  human or animal health question outside agricultural triage must be
  directed to a qualified professional.
- **Exact dosage prescriptions.** The model refuses to give fertilizer,
  pesticide, or veterinary dosages without a soil test and product label.
  Do not use ARIS as a substitute for a label or a licensed agrodealer.
- **Live market prices, weather forecasts, regulatory status.** ARIS has
  no internet access and cannot retrieve live data.
- **Non-agricultural tasks.** Coding, essay writing, political commentary,
  and general-purpose assistance are outside scope and are explicitly
  refused.
- **Autonomous decision-making.** ARIS is an advisory tool. It does not
  replace an extension officer, veterinarian, or agrodealer.

---

## Limitations

Known limitations, disclosed rather than hidden:

- **`adversarial_blacklist` accuracy is 67.6%** on the frozen evaluation.
  The model occasionally accepts false premises on diseases with
  overlapping symptom presentations. This is the weakest category and the
  specific target for the next iteration.
- **`factual_recall` accuracy is 71.4%** on the frozen evaluation. Two
  specific failures are documented in `frozen_eval_strict.json`:
  describing CMD as soil-borne (EVAL-005) and describing CBSD as showing
  leaf symptoms (EVAL-007). Both are factually wrong and fail under
  both strict and tolerant rubrics.
- **Four red-team failures in 86 seen probes.** Documented in
  `redteam_scored.json` with specific failure signatures. The unseen
  battery (53 probes) had zero failures, though 19 probes returned
  responses classified as `review` rather than `pass` — the scorer could
  not classify them cleanly.
- **No image input.** The model cannot see photos. Diagnosis is from text
  descriptions only, and the model is trained to refuse definitive
  diagnoses from text alone.
- **Regional bias.** Training data focuses on Nigerian agriculture.
  Recommendations may not transfer to other African regions or climates.
- **Nigerian Pidgin scope.** The model handles Nigerian Pidgin well
  (94.6% on the frozen evaluation), but does not speak Hausa, Yoruba,
  Igbo, or other Nigerian languages.

---

## Training Data

The training corpus is **2,448 hand-authored ChatML records**, released
under CC BY 4.0 for the ADTC 2026 submission. The corpus was **not
scraped** — public Nigerian agricultural Q&A data is thin, and what exists
is not licensable for redistribution.

Every record was authored against institutional reference material from:

- **IITA** — International Institute of Tropical Agriculture
- **NAERLS** — National Agricultural Extension and Research Liaison Services
- **NAFDAC** — National Agency for Food and Drug Administration and Control
- **State ADPs** — Agricultural Development Programmes

Each record passed a five-stage verification pipeline before being admitted:

1. **Candidate extraction** — draft records with source citations
2. **Consensus scanning** — against the canonical claims register
   (`provenance/dataset/canonical_claims.jsonl`) and the fabrication
   blacklist (`provenance/dataset/blacklist.json`)
3. **Human audit** — escalated records reviewed against sources
4. **Structural encoding** — conditional facts never stated as absolutes;
   fabrication-prone categories carry explicit refusal examples
5. **Post-training contamination audit** — four-level check (exact,
   normalized, bidirectional substring, near-duplicate Jaccard) against
   the frozen evaluation matrix

The register is self-contained: each record carries its own `status`,
`conditions`, and `do_not_generalise` fields inline. A reviewer can audit
any single fact by reading one record.

---

## Evaluation

### Custom frozen evaluation (131 records, 8 categories)

The frozen matrix was finalized before training and not tuned on. Each
record has `must_contain_all`, `must_contain_any`, and `must_not_contain`
patterns.

| Category | Strict rubric | Pass / Total |
|---|---|---|
| capability_disclosure | **100.0%** | 18 / 18 |
| diagnostic_uncertainty | **100.0%** | 12 / 12 |
| safety_refusal | 94.0% | 47 / 50 |
| pidgin | 94.6% | 35 / 37 |
| identity | 90.9% | 10 / 11 |
| multi_turn | 80.0% | 12 / 15 |
| factual_recall | 71.4% | 25 / 35 |
| adversarial_blacklist | 67.6% | 75 / 111 |
| **Overall (strict)** | **81.0%** | 234 / 289 |
| **Overall (tolerant canonical)** | **87.9%** | 254 / 289 |

**Two honest numbers.** The strict rubric uses a keyword-exact grader
that rejects correct answers phrased differently from the rubric author's
expected wording. The tolerant canonical scorer is monotonic over strict
and rescues only wording equivalences, not factual errors. Every rescued
record is listed in `frozen_eval_tolerant.json` under the `rescued` key
with its answer excerpt for manual inspection. Records where the model
stated a factually incorrect answer fail under both rubrics.

### ADTC profiler (participant mode, fresh run)

| Metric | Value |
|---|---|
| Machine | Intel Core i5-8365U, Ubuntu 22.04.5 LTS, no GPU |
| Generation speed | **13.63 tokens/sec** |
| First-token latency | **15.09 s** (cold, 512-token prompt) |
| Peak RAM | **1.69 GB** (1,688 MB) |
| Steady-state RAM | **1.60 GB** (1,599 MB) |
| ARC-Easy accuracy | **76.0%** (acc_norm, 50 samples) |
| CPU p99 | 53.1% |
| Threads used | 2 |
| Thermal throttling | None |

The organizers' own profiler run on the Standard Laptop is the
authoritative measurement. Development numbers above are the working
figures used to size the model against the 8 GB budget.

### Adversarial red-team

| Battery | Probes | Failures |
|---|---|---|
| Seen | 86 | 4 |
| Unseen | 53 | 0 |

Four seen failures are documented with specific failure signatures in
`redteam_scored.json`. The unseen battery had zero failures, but 19
probes returned responses classified as `review` rather than `pass` —
the scorer could not classify them cleanly. That distinction matters:
"0 failures" is not the same as "53/53 correct." Both numbers are in the
committed JSON for inspection.

---

## Usage

### Prerequisites

- `llama.cpp` built with `llama-cli` and `llama-bench` available on `PATH`
- ~2 GB of free RAM during inference
- No network connection required after download

### Download the model

```bash
bash download_model.sh
```

The script fetches the GGUF from a pinned HuggingFace commit
(`39e09e4d7303af31c6fa5199c2a09f3fad0d746e`) so the file cannot silently
change after evaluation begins. SHA-256 verification is documented in
`provenance/checksums.txt`.

### Run inference — strict evaluation mode

Deterministic, `temperature = 0.0`. **Always prepend the system prompt**
(see the top of this README):

```bash
SYS="$(cat provenance/system_prompt.txt)"
llama-cli -m model/ARIS-V10.1-1.5B-Q4_K_M.gguf \
  -p "<|im_start|>system
${SYS}<|im_end|>
<|im_start|>user
My cassava leaves are showing yellow-green mosaic patterns and the plant is stunted. What disease is this and how can I manage it?<|im_end|>
<|im_start|>assistant
" \
  -n 256 --temp 0.0 --threads 4 --no-display-prompt
```

### Run inference — conversational field mode

`temperature = 0.7`, sampling enabled:

```bash
SYS="$(cat provenance/system_prompt.txt)"
llama-cli -m model/ARIS-V10.1-1.5B-Q4_K_M.gguf \
  -p "<|im_start|>system
${SYS}<|im_end|>
<|im_start|>user
Wetin be the correct way to plant yam for rainy season?<|im_end|>
<|im_start|>assistant
" \
  -n 256 --temp 0.7 --top-p 0.9 --threads 4 --no-display-prompt
```

### Run the ADTC profiler

```bash
adtc-profiler run --submission . --mode participant --output submission.json
```

---

## Repository Layout

```
.
├── metadata.json                  ADTC submission metadata
├── download_model.sh              Fetches the GGUF from the pinned HF commit
├── REPORT.md                      Technical writeup
├── README.md                      This file
├── LICENSE                        GPL-3.0
├── .gitignore                     Excludes model weights
├── .gitattributes                 Routes adapter_model.safetensors through Git LFS
├── model/                         Empty; GGUF downloaded at eval time
├── provenance/                    Proof-of-training artifacts
│   ├── adapter/                   LoRA weights (Git LFS), manifest, config, README
│   ├── dataset/                   Verification artifacts
│   │   ├── canonical_claims.jsonl   Verified fact register
│   │   ├── blacklist.json           Fabrications and unsafe dosages
│   │   └── README.md
│   ├── system_prompt.txt          The canonical system prompt (required)
│   ├── checksums.txt              SHA-256 of base model, adapter, GGUF
│   ├── training_summary.json      Hyperparameters and best-checkpoint info
│   ├── training_loss_log.csv      Per-step loss values
│   ├── val_loss_log.csv           Per-step validation loss
│   ├── loss_curves.png            Visualisation
│   ├── merge_and_quantize.py      Adapter → GGUF script
│   ├── zero_leakage_audit.md      Contamination audit + flag dispositions
│   ├── before_after.json          Five base-vs-ARIS comparisons
│   ├── dataset_card.md            Dataset composition and license
│   ├── dataset_sample.jsonl       Representative training sample
│   └── notebook_execution_url.txt Reference to the committed training notebook
├── frozen_eval_strict.json        Frozen eval results (strict rubric)
├── frozen_eval_tolerant.json      Frozen eval results (tolerant rubric)
├── redteam_scored.json            Seen adversarial battery results
├── redteam_unseen_scored.json     Unseen adversarial battery results
└── aris-agricultural-research-information-system-v10 (6).ipynb
                                   The training notebook that produced the adapter
```

---

## Reproducibility

All artifacts a reviewer would need to verify the training run are
committed:

- LoRA adapter weights (`adapter_model.safetensors`, 295 MB, Git LFS-tracked)
  and metadata (SHA-256, storage path) in `provenance/adapter/`
- Per-step loss logs in `provenance/training_loss_log.csv`
- Training hyperparameters in `provenance/training_args.json`
- Dataset sample in `provenance/dataset_sample.jsonl`
- SHA-256 checksums of base model, adapter, and GGUF in
  `provenance/checksums.txt`
- Training script in `provenance/merge_and_quantize.py`
- Contamination audit in `provenance/zero_leakage_audit.md`
- The canonical system prompt in `provenance/system_prompt.txt`
- The full training notebook committed at repository root

The base model revision is recorded in `metadata.json` under
`provenance.base_model_commit_sha`. The downloaded GGUF is pinned to
commit `39e09e4d...` in `download_model.sh`.

---

## Ethical Considerations

Agricultural advice carries real consequences. A farmer who acts on a
wrong recommendation can lose a crop, lose livestock, or harm themselves
or their family. ARIS was designed with this in mind:

- **Refusal-first dosage behavior.** The model refuses to give fertilizer,
  pesticide, or veterinary dosages without soil-test and label context.
  This is the point, not a limitation.
- **Diagnostic uncertainty.** The model refuses to state definitive
  diagnoses from text descriptions alone and refers farmers to extension
  officers when symptoms overlap.
- **No fabricated institutions.** The model does not invent variety names,
  institution acronyms, or regulatory numbers.
- **Honest failure reporting.** Four seen red-team failures and two
  factual-recall errors are documented in the committed artifacts rather
  than concealed.

---

## Citation

```bibtex
@misc{aris_v10_1_2026,
  title   = {ARIS V10.1: An Offline Agronomic Advisor for Nigerian Smallholder Farmers},
  author  = {Nwaruwe, Victor Chukwuebuka},
  year    = {2026},
  note    = {Africa Deep Tech Challenge 2026, Laptop LLM track},
  url     = {https://github.com/Vicgrace01/ARIS}
}
```

---

## Contact

**Team ID:** agrigemma
**Submitter:** Victor Chukwuebuka Nwaruwe
**Email:** victornwaruwe@gmail.com
**GitHub:** [@Vicgrace01](https://github.com/Vicgrace01)

For questions about the ADTC 2026 challenge, see
[adtc-2026.devpost.com](https://adtc-2026.devpost.com/).

---

## License

GPL-3.0. See [`LICENSE`](LICENSE).

Training corpus released under CC BY 4.0. Source reference materials from
IITA, NAERLS, and NAFDAC remain the property of their respective
organisations and are used as factual reference only; no source documents
are redistributed.
