# Phase 13 - Production Probability Sanity Audit (AUDIT ONLY)

- Audits what the production screening_probability means in practice for the frozen
  experimental binary screening candidate (binary_lr_latest_visit_v1).
- n=38 outer-test subjects (latest available visit by MR Delay), subject-level.
- No production changes, no recalibration, no probability transformation, no commit.
- Artifacts: outer_test_predictions.csv, probability_bins.csv, probability_distribution.json,
  extreme_probability_cases.csv, calibration_metrics.json, reliability.png, fixed_case_result.json,
  coefficient_analysis.json.
- The reliability diagram and calibration metrics are EXPLORATORY (n=38); not statistically definitive.
- The probability is a model-estimated screening probability, NOT clinically validated.
