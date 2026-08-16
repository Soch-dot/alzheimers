# Phase 7 - Latest-Available-Visit Screening + Eligibility (Research Only)

- Experimental latest-available-visit screening analysis.
- Same Phase 4/5 binary candidate (LR C=10 balanced; age, sex, education_years, mmse, ses).
- Threshold 0.40 frozen; MR Delay authoritative ordering; 112/38 split reused.
- Baseline vs latest-available-visit comparison, paired subject analysis, probability changes.
- No CDR; no longitudinal delta features; no production changes; no commits.
- LIMITATION: OASIS Group label is constant per subject (final classification).
  Latest-visit classification does NOT mean the model predicted future conversion;
  it classifies the subject using information available at the latest observed visit.
