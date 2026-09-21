# ARIS V10.1 — Dataset Verification Artifacts

These files were used to verify every training record before it was admitted
to the corpus. They are committed so a judge can reproduce the audit path
without re-running any training code.

## Files

- **canonical_claims.json** — Maintained list of agronomic claims that must
  never appear as unqualified fact. Each entry includes the claim, the
  category (dosage, spacing, vector, variety, institution), and the source
  citation that establishes the correct answer.
- **blacklist.json** — Assertions ruled out as fabrications, invented
  institutions, or unsafe dosages. Any candidate that matched an entry here
  was removed.
- **conditional_register.json** — Claims that are true only under stated
  conditions (variety, season, intercrop status). Candidates matching an
  entry here were escalated to human review rather than admitted directly.
- **pre_verified_subset.jsonl** — The subset of candidates that survived all
  verification stages. Preserved for re-auditability.

## Provenance

Authored in-house by the ARIS team against IITA, NAERLS, and NAFDAC reference
material. Released under CC BY 4.0 for the ADTC 2026 submission.
