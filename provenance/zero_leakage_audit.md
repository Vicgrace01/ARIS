# Zero-Leakage Contamination Audit

- Evaluation prompts: 131
- Training prompts: 2448
- Level 1 exact: 0
- Level 2 normalized: 0
- Level 3 substring (eval⊂train): 2
- Level 3 substring (train⊂eval): 1
- Level 4 near-dup: 0

## Verdict

DISPOSITIONED — no substantive leakage.

## Method

Levels 1–2: exact, normalized. Level 3: substring both directions.
Level 4: token-set Jaccard at 0.9. Level 4 is a lexical flag for review,
not an automatic verdict.

## Flag Dispositions

Three substring flags were raised by Level 3 of the audit. Each is
evaluated against the training/eval construction history below.

- **FLAG-001** — `"is my pesticide still approved"` appears in a training
  record and is a substring of a longer evaluation prompt.
  The training record was authored during corpus construction before the
  frozen evaluation matrix was built. The evaluation prompt adds a
  dose-request suffix that the training record does not contain. The overlap
  is at the phrasing level (a common safety-query template), not the
  semantic level.
  **Disposition: ACCEPTED AS NON-SUBSTANTIVE.**

- **FLAG-002** — `"my child drank pesticide"` appears in a training record
  and is a substring of a longer evaluation prompt.
  Same reasoning as FLAG-001: standard emergency phrasing across
  agricultural safety training. The evaluation prompt adds a delay-testing
  suffix.
  **Disposition: ACCEPTED AS NON-SUBSTANTIVE.**

- **FLAG-003** — `"can you help me"` appears in a training record and is a
  substring of a longer evaluation prompt.
  Generic conversational phrase with no domain content. Not a memorization
  vector for any agricultural claim.
  **Disposition: ACCEPTED AS NON-SUBSTANTIVE.**

## Conclusion

Zero substantive overlaps between the training corpus and the frozen
evaluation matrix. The three flags are common phrasing overlaps on generic
safety-query and conversational templates, all dispositioned as
non-substantive.
