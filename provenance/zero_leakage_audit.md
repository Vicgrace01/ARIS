# Zero-Leakage Contamination Audit

- Evaluation prompts: 131
- Training prompts: 2448
- Level 1 exact: 0
- Level 2 normalized: 0
- Level 3 substring (eval⊂train): 2
- Level 3 substring (train⊂eval): 1
- Level 4 near-dup: 0

## Verdict

REVIEW

## Method

Levels 1–2: exact, normalized. Level 3: substring both directions. Level 4: token-set Jaccard at 0.9. Level 4 is a lexical flag for review, not an automatic verdict.
