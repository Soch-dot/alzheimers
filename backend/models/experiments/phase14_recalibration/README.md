# Phase 14 - Experimental Probability Recalibration Feasibility (Research Only)

- Compares raw vs sigmoid(Platt) vs isotonic calibration on the frozen production LR
  probability outputs. Underlying model never retrained.
- Canonical 112/38 split; OOF raw probs (5-fold grouped CV, latest visit); second-level
  grouped calibration CV for honest method selection; outer 38 used exactly once.
- Threshold discipline: Option A preserves raw >= 0.40 screening decision; calibrated
  display threshold = f(0.40). No recalibration integrated; production unchanged.
- Isotonic flagged as unstable (Converted n=14). No commits.
- Model version: binary_lr_latest_visit_v1. Features: age, sex, education_years, mmse, ses.
