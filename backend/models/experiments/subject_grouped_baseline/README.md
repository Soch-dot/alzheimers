# Subject-Grouped Baseline (ML Phase 1) - Research Only

Leakage-free subject-level evaluation of the production Random Forest configuration.  Do NOT confuse this score with final accuracy.

## Split
- Unit: Subject ID (subject-level stratified split; visits kept together)
- Train subjects: 112, Test subjects: 38, overlap: 0
- Train rows: 281, Test rows: 92

## Preprocessing (leakage-safe)
- Pipeline fitted on training subjects ONLY: SimpleImputer(strategy='median')
- No StandardScaler (not needed for Random Forest)
- No df.fillna(df.median()) before the split

## Model (unchanged production config)
- RandomForestClassifier(n_estimators=200, random_state=42, class_weight={0:1,1:4,2:2})

## Results
See metrics.json.  Classification is multiclass; per-class metrics for Converted (class 1) are unstable because of tiny test subject count.
