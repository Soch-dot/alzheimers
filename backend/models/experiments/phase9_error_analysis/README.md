# Phase 9 - Missed-Positive Error Analysis (Research Only)

- Diagnostic analysis of missed positives OAS2_0145, OAS2_0176 (Converted), OAS2_0175 (Demented).
- Same Phase 5 model (LR C=10 balanced; age, sex, education_years, mmse, ses); threshold 0.40; latest-visit policy.
- ONLY prediction-time-available features used; CDR/future/MRI/label-derived forbidden.
- Counterfactual ranges restricted to observed dataset values.
- Aggregation alternatives reported DESCRIPTIVELY only (no rule selected on outer test).
- No production changes; no rules; no commits.
