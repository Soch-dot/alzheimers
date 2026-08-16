# Phase 5 - Screening Threshold + Uncertainty (Research Only)

- Model: experimental binary screening candidate (LogisticRegression).
- Threshold selected on dev subject-level OOF ONLY (sweep 0.10-0.90 step 0.01).
- Selected rule: sens_ge_0.80 at threshold 0.40.
- Outer 38-subject test applied exactly once.
- No calibration added; no production changes.
