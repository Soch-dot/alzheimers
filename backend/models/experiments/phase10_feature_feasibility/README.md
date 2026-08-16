# Phase 10 - Feature-Feasibility Experiment (Research Only)

- Experimental feature-feasibility analysis: can prediction-time-available features beat the five-feature ceiling?
- Experiment A (current-product info): Sets 1-4 (control, visit_count, time_since_baseline, prev_mmse+mmse_delta).
- Experiment B (OASIS MRI, research-only): Set 5 = Set 1 + eTIV, nWBV, ASF.
- CDR, Group, future/post-outcome information PROHIBITED; no MRI added to product; /predict unchanged.
- Grouped 5-fold CV on 112 train subjects; outer 38 subjects evaluated once per set; threshold 0.40; subject-level latest visit.
- LIMITATION: target is the final OASIS outcome label; gains classify the outcome label, not future conversion.
- No production changes; no commits.
