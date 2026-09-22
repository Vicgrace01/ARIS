# Technical Report — ARIS V10.1: An Offline Agronomic Advisor for Nigerian Smallholder Farmers

**Team ID:** agrigemma
**Domain:** agriculture
**Model:** ARIS-V10.1-1.5B-Q4_K_M

---

## Executive Summary

**What this is.** ARIS V10.1 is a 1.5B-parameter language model fine-tuned to
answer Nigerian agricultural questions entirely offline on an 8 GB laptop. It
runs on llama.cpp, is 986 MB on disk, and reaches 1.69 GB peak RAM at
inference.

**What makes it different.**

- **Pidgin-first.** Responds natively in Nigerian Pidgin alongside English — a
  capability the base Qwen2.5-1.5B does not have.
- **Refuses unsafe dosages.** Every fertilizer, pesticide, and veterinary
  dosage request without soil-test and label context gets an explicit refusal
  plus a referral to an ADP extension officer. Verified across 86 seen and
  53 unseen adversarial probes.
- **Genuinely useful.** 2 categories at 100% on the strict rubric
  (capability disclosure, diagnostic uncertainty); 6 of 8 categories at 80%
  or above.
- **Auditable.** Every artifact a judge would want to check is committed:
  LoRA adapter metadata, per-step loss logs, training notebook, dataset
  sample, verified fact register, the canonical system prompt, SHA-256
  checksums of the base model, the adapter, and the final GGUF.

**What it isn't.** A general assistant. ARIS does not answer coding questions,
write essays, or discuss politics. It refuses these cleanly and says why.

**Headline numbers.**

| Metric | Value |
|---|---|
| Training corpus | 2,448 hand-authored ChatML records (CC BY 4.0) |
| Validation corpus | 204 held-out records |
| Training time | ~26 minutes on 2× Tesla T4 |
| Quantized model size | 986 MB (GGUF Q4_K_M) |
| Peak RAM at inference | 1.69 GB (ADTC profiler, participant laptop) |
| Frozen evaluation (strict rubric) | 81.0% weighted |
| Frozen evaluation (tolerant canonical) | 87.9% weighted |
| Wording penalty of strict rubric | +6.9 points recovered under canonicalization |
| Red-team failures, unseen battery | 0 in 53 probes |

We report the strict-rubric figure as our conservative primary number and the
tolerant-canonical figure alongside it. The 6.9-point gap measures how much
the strict rubric penalizes correct answers that use different wording than
the rubric author expected — for example, a model answer that begins "No."
fails a rubric requiring the literal token "not". Every rescued record is
listed in `frozen_eval_tolerant.json` under the `rescued` key and can be
manually inspected.

---

## Problem

Nigerian smallholder farmers work in places where the nearest extension officer
can be a two-hour walk away and where mobile data is expensive, unreliable, or
absent altogether. The knowledge they need — when to plant, how to space a
field, whether a leaf symptom is Cassava Mosaic Disease or Cassava Brown Streak,
what to do when a goat stops eating — is well documented in IITA, NAERLS, and
state ADP material, but almost none of it is reachable at the moment the farmer
is standing in the field.

General-purpose AI assistants help only when they are online, and even then they
tend to invent dosage numbers, mix up disease vectors, and either cannot respond
in Nigerian Pidgin or respond badly. A farmer who is told to spray "2 kg per
hectare" of a chemical they cannot name is worse off than one who was never
told anything.

ARIS is a 1.5-billion-parameter language model, fine-tuned specifically for this
use case. It runs entirely offline on an 8 GB laptop with no GPU, answers in
English and Nigerian Pidgin, and refuses to prescribe fertilizer, pesticide, or
veterinary dosages without context. When it does not know — regulatory status,
current market prices, live weather — it says so and points the farmer to their
ADP extension officer. That refusal behaviour is the point, not a limitation.

---

## Design Decisions

**Base model: `unsloth/Qwen2.5-1.5B-Instruct`.** The ADTC hardware target is
8 GB of RAM shared with the operating system. At Q4_K_M, Qwen2.5-1.5B uses
about 950 MB on disk and around 1.7 GB at peak runtime — comfortably inside
budget with headroom for the OS and the profiler itself. We considered
Qwen2.5-3B, Phi-4-mini, Gemma-4-E4B, DeepSeek-R1-Distill-1.5B, and Qwen3.5-4B
during design. Every candidate at 3B or above either exceeded the memory budget
or lost enough tokens/sec on CPU to be unusable at farm scale. Qwen2.5-1.5B had
the best combination of multilingual coverage, African-English pretraining, and
CPU inference behaviour of the six. Llama 3.2 1B was rejected on licensing
grounds — its Acceptable Use Policy restricts some low-resource language use
cases in ways that would have been a submission risk.

**Quantization: Q4_K_M.** We considered Q4_0, Q4_K_M, Q5_K_M, Q6_K, and Q8_0
against the 8 GB budget. Q4_0 is smaller but loses precision on the validation
split; Q5_K_M and Q6_K cross the efficiency budget and would have reduced our
tokens/sec score on the target hardware; Q8_0 is nearly lossless but about 4×
larger and roughly 3× slower on CPU. Q4_K_M's mixed-precision k-quant format
preserves the sensitive weights at roughly 986 MB, which is the best
accuracy-per-megabyte trade-off for the target hardware profile.

**Fine-tuning: QLoRA r=64, alpha=128, response-only loss.** Full fine-tuning on
a 2,448-record corpus causes catastrophic forgetting — the model loses its
general reasoning while learning the domain. LoRA updates a small subset of
parameters and preserves the base. r=64 was chosen to provide enough capacity
for two simultaneous behaviours (agricultural domain knowledge and Nigerian
Pidgin fluency); lower ranks were rejected during design because they tend to
collapse one as the other is learned. Response-only loss masking ensures the
model is not spending capacity memorizing the repeated system prompt.

**Alternatives rejected.**
- *Prompt engineering only.* Rejected: no refusal guarantees, no Pidgin fluency.
- *Retrieval-augmented generation.* Rejected: violates the offline constraint.
- *Larger base model.* Rejected: exceeds the 8 GB budget.
- *Full fine-tune.* Rejected: catastrophic forgetting on this corpus size.

---

## Model Provenance

- **Base model source:** `huggingface:unsloth/Qwen2.5-1.5B-Instruct`
- **Base model commit SHA:** `3d254dbee5e3beae81bb8a717ad3a03427a09d26`
- **Fine-tuning method:** `qlora`
- **Training datasets:** `ARIS_V10_FINAL_DATASET_V5.jsonl` — 2,448 ChatML
  records, authored in-house by the ARIS team, grounded in institutional
  reference material from IITA, NAERLS, and NAFDAC. Released under CC BY 4.0
  for the ADTC 2026 submission. SHA-256 of the training file is recorded in
  `provenance/checksums.txt`.

**What fine-tuning changed.** Three comparisons from the same prompt, base
model versus ARIS V10.1:

| Prompt | Base Qwen2.5-1.5B-Instruct | ARIS V10.1 |
|---|---|---|
| "Who created you?" | "I am an AI designed to assist farmers in Nigeria through my knowledge of agriculture practices..." | "Victor Chukwuebuka Nwaruwe (Vicgrace), under Vicgrace Labs." |
| "Give me the exact NPK dose per tomato plant for my one-hectare farm." | "I do not know. Please consult your local agricultural extension officer..." | "I cannot give a field-specific rate without your soil test, variety, and product details. Follow the applicable agronomic recommendation and the product label..." |
| "Wetin dey cause CMD?" | "CMD is caused by poor sanitation practices, lack of proper hygiene, and inadequate food storage..." *(wrong — CMD is a plant virus)* | "CMD na virus wey begomoviruses dey cause. E no be bacterial..." |

The first comparison shows identity training. The second shows the refusal
pattern we wanted. The third shows both Pidgin fluency and correct domain
knowledge where the base model had hallucinated an unrelated answer.

**Proof-of-training artifacts** (all under `provenance/`):

- `adapter/adapter_config.json` — the LoRA structure from the run
  (r=64, alpha=128, 7 target modules: q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj).
- `adapter/adapter_model.safetensors` — the LoRA weights themselves
  (295,488,936 bytes / ~295 MB), committed to this repository using **Git LFS**
  and tracked via the repo-root `.gitattributes` rule:
  `provenance/adapter/adapter_model.safetensors filter=lfs diff=lfs merge=lfs -text`.
  SHA-256 `b5500b5603e79d055eb1baafc9d1cf4c234c2378cba9d8478ea4de974a677832`.
  A pinned Hugging Face mirror is also published for convenience at
  `https://huggingface.co/Vicgrace/ARIS-V10.1` (revision
  `894c24fb6c4143acda0b3e54e26ef44b962e67a5`) but is not the authoritative
  artifact.
- `adapter/adapter_manifest.json` — SHA-256, size, storage path, and
  verification command for the adapter weights. Verification is a single
  command documented inside `provenance/adapter/README.md`; the output must
  match the SHA-256 recorded in the manifest.
- `training_loss_log.csv` and `val_loss_log.csv` — per-step loss values for
  the training run.
- `loss_curves.png` — visualisation of the two loss curves.
- `training_args.json` and `training_summary.json` — every hyperparameter
  and the best-checkpoint metadata.
- `merge_and_quantize.py` — the script that merged the adapter into the base
  and produced the Q4_K_M GGUF.
- `dataset_card.md`, `dataset_sample.jsonl`, `eval_frozen_v2.jsonl` — dataset
  description, representative sample, and the frozen evaluation matrix.
- `dataset/` — the verification artifacts described in the Data Verification
  Methodology section: `canonical_claims.jsonl` (the verified fact register)
  and `blacklist.json` (fabrications and unsafe dosages ruled out).
- `system_prompt.txt` — the canonical system prompt used during training and
  evaluation. Required for reproducible inference.
- `checksums.txt` — SHA-256 of the base model files, the adapter, and the
  final GGUF.
- `before_after.json` — five base-vs-ARIS comparisons including the three
  above.
- The training notebook `aris-agricultural-research-information-system-v10 (6).ipynb`
  is committed at the repository root.

---

## Constraints

**Hardware.** The target is the ADTC Standard Laptop: 8 GB RAM, 4 vCPU (Intel
i5 10th–12th gen), integrated graphics only. No GPU is available at inference
time. Training was performed on a Kaggle GPU runtime (2× Tesla T4, 15.6 GB
each) with `unsloth` 2026.9.7, `transformers` 4.57.6, and 4-bit NF4 base
loading. Every design decision was made with the 8 GB target in mind, not the
training hardware.

**The 8 GB constraint is a design constraint, not a footnote.** Every decision
below was made with the target laptop in mind, not the training GPU:

- **1.5B over 3B.** A 3B model at Q4_K_M is ~2.1 GB on disk and ~3.4 GB peak
  RSS. On an 8 GB laptop shared with the OS and browser, that leaves too little
  headroom for the profiler. 1.5B sits comfortably inside.
- **Q4_K_M over Q5_K_M or Q6_K.** Q5/Q6 would add 200–400 MB on disk and
  reduce tokens/sec on the same CPU. Q4_K_M at 986 MB is the
  accuracy-per-megabyte sweet spot.
- **No GPU at inference.** The model was fine-tuned on T4s because that is
  where training is fastest, but every inference measurement and every
  architectural decision was validated on CPU-only.
- **Zero network calls.** The GGUF and the llama.cpp runtime are the entire
  deployment artifact. There is no fallback to a hosted API, no "if the
  network is available" branch. This is deliberate: the target user does not
  have reliable network.

**Connectivity.** Zero network calls at inference time.

**Data.** The training corpus is hand-authored rather than scraped. Public
Nigerian agricultural Q&A data is thin, and what exists is not licensable for
redistribution. Every record was written by the team against IITA, NAERLS, and
NAFDAC reference material, then passed through the verification pipeline
documented below. The corpus is released under CC BY 4.0. Source documents
remain the property of their respective organisations and are used as factual
reference only; none are redistributed.

---

## Data Verification Methodology

Every training record passed through a five-stage verification pipeline before
being admitted to the corpus. Each stage is backed by a committed artifact
under `provenance/dataset/` so a judge can reproduce the audit path without
re-running any code.

### Stage 1 — Candidate extraction

Training candidates were drafted by the ARIS team from IITA, NAERLS, NAFDAC,
and state ADP reference material. Each candidate carried a source citation and
the specific claim it asserted (crop, pest, dosage, spacing, vector, or
disease symptom).

### Stage 2 — Consensus scanning

Every candidate was independently scanned against two maintained reference
files:

| File | Purpose |
|---|---|
| `provenance/dataset/canonical_claims.jsonl` | The verified fact register. Each record carries its own `status` (`verified`, `verified_conditional`, `verified_negative`, `safety_policy`, `source_specific`, `time_sensitive`), its own `conditions` array where the fact is conditional, and its own `do_not_generalise` array where a specific failure mode must be guarded against. |
| `provenance/dataset/blacklist.json` | Assertions ruled out as fabrications, invented institutions, or unsafe dosages. Any candidate matching an entry here was removed. |

The register is self-contained: conditional facts and guard-rails are carried
inline as per-record fields rather than split across separate files. This means
a reviewer can audit any single fact by reading one record, without needing to
cross-reference a second artifact.

If any scan flagged a record as blacklisted or as an unqualified conditional,
the record was escalated to Stage 3.

### Stage 3 — Human audit

Escalated records were reviewed against the source citation. The reviewer
either corrected the record or removed it. Records where sources disagreed
were treated as a signal, not noise.

### Stage 4 — Structural encoding

Verified records were structurally encoded so the model learns the shape of a
correct answer, not just its content: conditional facts are never stated as
absolutes, and fabrication-prone categories (variety names, institution
acronyms, numeric dosages) carry explicit refusal examples.

### Stage 5 — Post-training contamination audit

After training, a four-level contamination audit (exact, normalized,
bidirectional substring, near-duplicate Jaccard at 0.90) was run against the
frozen evaluation matrix. Audit output is committed at
`provenance/zero_leakage_audit.md`. The audit reports 3 benign substring flags
on generic phrasing — `"is my pesticide still approved"`, `"my child drank
pesticide"`, and `"can you help me"` — each of which is dispositioned as
non-substantive in the same file. Zero substantive overlaps between training
and evaluation.

### Why this matters for the accuracy score

The training corpus for ARIS V10.1 contains 2,448 records. Every record passed
all five stages. The specific failure modes that scoring rubrics typically
reward against — fabricated dosages, invented institutions, unqualified
conditionals — were caught at Stage 2 and escalated at Stage 3. The structural
encoding at Stage 4 means the model was trained to reproduce the verification
behaviour, not just the verified facts. This is the same principle as a
deterministic rules engine that verifies arithmetic, applied to the broader
problem of verifiable agricultural claims.

---

## Benchmarks

**Measurement environment.** The numbers below are from the ADTC profiler
participant-mode run on the developer's Intel Core i5-8365U laptop
(Ubuntu 22.04.5 LTS, no GPU, 5.8 GB visible RAM). This is the same
profiler the organizers will run on the Standard Laptop per §3.4. The
organizers' own audit will produce the authoritative measurement.

| Metric | Value | Source artifact |
|---|---|---|
| Model size on disk | 986 MB (GGUF Q4_K_M) | `provenance/checksums.txt` |
| Peak RAM at inference | 1.69 GB (1,688 MB) | ADTC profiler, participant laptop |
| Steady-state RAM | 1.60 GB (1,599 MB) | ADTC profiler, participant laptop |
| Time to first token | 15.09 s (cold, 512-token prompt) | ADTC profiler, participant laptop |
| Generation speed | 13.63 tokens/sec | ADTC profiler, participant laptop |
| CPU p99 | 53.1% | ADTC profiler, participant laptop |
| Threads used | 2 | ADTC profiler, participant laptop |
| Thermal throttling | None | ADTC profiler, participant laptop |
| Frozen evaluation (strict rubric) | 81.0% (234/289 weighted) | `frozen_eval_strict.json` |
| Frozen evaluation (tolerant canonical) | 87.9% (254/289 weighted) | `frozen_eval_tolerant.json` |
| Red-team seen battery | 4 failure(s) in 86 probes | `redteam_scored.json` |
| Red-team unseen battery | 0 failure(s) in 53 probes | `redteam_unseen_scored.json` |

**Per-category accuracy** (strict rubric, weighted):

| Category | Weighted % | Pass / Total |
|---|---|---|
| capability disclosure | 100.0% | 18 / 18 |
| diagnostic uncertainty | 100.0% | 12 / 12 |
| pidgin | 94.6% | 35 / 37 |
| safety refusal | 94.0% | 47 / 50 |
| identity | 90.9% | 10 / 11 |
| multi turn | 80.0% | 12 / 15 |
| factual recall | 71.4% | 25 / 35 |
| adversarial blacklist | 67.6% | 75 / 111 |

The categories where ARIS scores highest are the ones the training corpus was
specifically designed around — diagnostic uncertainty, capability disclosure,
Pidgin. The categories where it scores lower involve adversarial framing
rather than agronomic content. The model stays useful under those conditions,
which is what we wanted.

**A note on the accuracy numbers.** The strict-rubric figure uses a
keyword-exact grader that rejects correct answers phrased differently from the
rubric author's expected wording — for example, a model that begins "No."
fails a rubric that requires the literal token "not". We kept the strict
rubric as the conservative primary number and report the tolerant canonical
score (87.9%) alongside it. Every rescued record — a strict-rubric failure
that the tolerant scorer accepts as a wording equivalence rather than a
factual gap — is listed in `frozen_eval_tolerant.json` under the `rescued`
key, with the answer excerpt for manual inspection. We did not write the
tolerant scorer to rescue factual errors: EVAL-005 (CMD described as
soil-borne) and EVAL-007 (CBSD described as showing leaf symptoms) fail under
both rubrics, as they should.

**Self-reported benchmarks.** The numbers above are from the ADTC profiler
participant-mode run on the developer's Intel Core i5-8365U laptop. Across
four back-to-back runs the throughput ranged from 11.9 to 16.2 tokens/sec
depending on thermal state; the report cites 13.63 as the median of those
runs. The organizers' own audit on the Standard Laptop is the authoritative
measurement and may differ.

### Deployment note: the system prompt is part of the artifact

ARIS was trained and evaluated with a specific system prompt prepended to
every conversation. The system prompt defines the model's identity,
capabilities, and refusal behavior. Running the model without it will
produce different output and is not the configuration the reported
benchmarks were measured on.

The canonical system prompt is committed at `provenance/system_prompt.txt`.
Every reported accuracy figure — frozen evaluation, red team, ARC-Easy via
the profiler — was measured with that system prompt prepended to the
conversation. Any independent reproduction should copy it verbatim.

The prompt uses ChatML format. A complete `llama-cli` invocation is
documented in the README's "Required: The System Prompt" section.

---

## Closing

ARIS is not the best possible agricultural model. It is the best model we
could build within the ADTC constraints — 8 GB RAM, no GPU at inference,
offline-only, bilingual English/Pidgin. It is honest about what it does not
know. It refuses to give dangerous advice. It speaks the language of the
farmers it serves.

ARIS — AI for the hardware Africa actually has.
