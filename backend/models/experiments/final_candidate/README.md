# Phase 11 - Experimental Model Lock + Reproducible Validation Package (Research Only)

- Freezes the experimental binary screening candidate: LR C=10 balanced, features age/sex/education_years/mmse/ses.
- Target: 0=Nondemented, 1=Converted|Demented. Threshold 0.40. Visit policy: latest available by MR Delay; single visit used as-is.
- Probability semantics: model-estimated probability of a dementia-related outcome (NOT diagnosis/certainty).
- Latest-visit classification is NOT future-conversion prediction (OASIS Group is a final subject-level label).
- Rebuilt from scratch on the 112 training subjects' latest visits; outer 38 untouched until final evaluation.
- Research-only SHAP (LinearExplainer) for the frozen pipeline; production SHAP and /predict unchanged.
- NOT a production integration; no production changes; no commits.
- LIMITATIONS: n=150 subjects, 14 Converted, 4 Converted in outer test, wide bootstrap CIs, no clinical validation,
  no prospective conversion prediction; Converted cases with normal MMSE can be indistinguishable from Nondemented.
