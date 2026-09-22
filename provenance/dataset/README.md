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

## Provenance

Authored in-house by the ARIS team against IITA, NAERLS, and NAFDAC reference
material. Released under CC BY 4.0 for the ADTC 2026 submission.
