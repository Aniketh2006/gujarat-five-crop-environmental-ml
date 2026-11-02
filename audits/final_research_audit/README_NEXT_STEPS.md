# Read first — Stage 1–4 research audit

**Verdict:** Course project results available; not publication certified.

- `01_ZERO_LABEL_VERIFICATION_QUEUE.csv`: 335 source cells still require independent original-table verification. Do not mark them VERIFIED without checking.
- `02_ZERO_LABEL_AUDIT_SUMMARY.csv`: counts by crop.
- `03_PRIOR_SPOT_CHECKS_FOR_REVIEW.csv`: only three prior positive-value spot checks.
- `04_BAYESIAN_SOURCE_SENSITIVITY.csv`: paired preferred/strict posterior summary; not a posterior predictive check.
- `05_SPATIAL_BLOCK_IDW_BASELINE_OOF.csv`: block-held-out **IDW**, not Kriging.
- `06_METRICS.json`: calculated diagnostics.
- `Stage1_4_Research_Audit_Report.docx`: submission-oriented audit draft.

**Next action:** Review source cells for the 335 zero-area entries in the original Gujarat final-forecast APY PDF/Excel files, especially ambiguous blank/dash/zero entries. The V9 AUC is provisional until labels are verified. Do not push to Git until reviewing licenses and excluding `.venv`, backups, and secrets.
