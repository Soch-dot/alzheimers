# Phase 8 - Single-Visit Coverage + Visit-Count Stratified Performance (Research Only)

- Experimental single-visit coverage analysis.
- No true one-visit subjects exist in OASIS (all 150 subjects have >=2 visits);
  one-visit performance is SIMULATED by restricting each subject to their earliest observed visit.
- Scenario A: simulated one-visit (trained on earliest visits only).
- Scenario B/C: latest-available-visit screening (Phase 5 winner; B and C identical).
- Threshold 0.40 frozen; MR Delay ordering; 112/38 split reused; no CDR; no longitudinal delta features.
- LIMITATION: OASIS Group label is constant per subject (final classification);
  latest-visit classification does NOT mean future-conversion prediction.
- No production changes; no commits.
