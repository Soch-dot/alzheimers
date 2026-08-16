# 3-Class vs Binary Screening (ML Phase 4) - Research Only

- Binary target: 1=Converted|Demented, 0=Nondemented (experimental).
- Same 112/38 subject split as Phase 1/2/3, zero overlap.
- Subject is the independent unit; visit probabilities aggregated by mean.
- Winner (experimental): LogisticRegression with raw calibration.
- Outer test threshold fixed at 0.5 (no threshold tuning).
- No production / /predict / SHAP / MMSE / Q11 changes.
