# Next Steps: Reducing Rule Dependency

Based on ChatGPT's feedback and confusion matrix analysis, here's what we found and what to do next.

## 🔍 Current Status

### ✅ What's Working Well
- **MMSE dominance**: 20.51% (SHAP) - **BELOW 25% TARGET!** ✅
- **CDR is primary feature**: 18.07% (SHAP) - Perfect!
- **Accuracy**: 90.67% - Excellent
- **ROC-AUC**: 96.02% - Excellent
- **Nondemented & Demented**: Both have 100% recall - Perfect!

### ⚠️ Critical Issue Found

**Converted Class Recall: 12.5%** - **VERY LOW!**

**Confusion Matrix Shows:**
- 75% of Converted cases are being predicted as **Nondemented**
- 12.5% are correctly predicted as Converted
- 12.5% are predicted as Demented

**Main Problem:** Model struggles with **Nondemented ↔ Converted** distinction (6 cases confused) vs Converted ↔ Demented (1 case).

## 🎯 Goal: Reduce Rule Dependency from 20% → <5%

Currently, the model relies on:
- **ML Model**: ~80% of predictions
- **Post-processing Rules**: ~20% of predictions

**Target:** Make the model learn rule logic internally so rules are only needed for <5% of edge cases.

## 📋 Action Plan

### 1. ✅ Generate Synthetic Adversarial Cases (DONE)

**File:** `src/generate_synthetic_data.py`

**Generated 105 synthetic cases covering:**
- Young + CDR=0.5 + High MMSE → Converted (20 cases)
- Low Education + Low MMSE + CDR=0 → Nondemented (20 cases)
- Very Old + CDR=0.5 + Moderate MMSE → Demented (15 cases)
- CDR=0 + MMSE≥20 → Nondemented (15 cases)
- CDR≥1.0 → Demented (15 cases)
- Borderline Converted cases (20 cases)

**Next Step:** Merge with original dataset and retrain.

### 2. ⏳ Merge Synthetic Data with Training Data

**Action Required:**
1. Modify `train_models.py` to load synthetic data
2. Combine with original dataset
3. Retrain model
4. Verify Converted recall improves

**Expected Result:** Converted recall should improve from 12.5% → 70-85%

### 3. ⏳ Increase Weight of CDR=0 and CDR=1 Cases

**Current class weights:** `{0: 1.2, 1: 10.0, 2: 1.0}`

**Proposed change:**
- Increase CDR=0 (Nondemented) weight when CDR=0.0
- Increase CDR≥1.0 (Demented) weight when CDR≥1.0
- Keep Converted weight high (already 10x)

**Implementation:** Add sample weights based on CDR values in training.

### 4. ⏳ Add More CDR-Based Features

**Current CDR features:**
- `cdr_severe`, `cdr_mild`, `cdr_normal`
- `cdr_zero_override`, `cdr_high_override`
- `cdr_converted_indicator`
- `cdr_risk_multiplier`

**New features to add:**
- `cdr_confidence_score` - How confident is the CDR assessment?
- `cdr_age_interaction` - CDR adjusted for age
- `cdr_education_interaction` - CDR adjusted for education

### 5. ⏳ Monitor Rule Usage

**Action:** Add logging to track when rules override model predictions.

**Goal:** Measure current rule usage (should be ~20%) and track reduction over time.

**Target:** <5% rule usage after improvements.

## 📊 Metrics to Track

### Primary Metrics
1. **Converted Recall**: Current 12.5% → Target 80-85%
2. **Rule Usage**: Current ~20% → Target <5%
3. **Overall Accuracy**: Maintain 90%+
4. **ROC-AUC**: Maintain 96%+

### Secondary Metrics
1. **Nondemented ↔ Converted confusion**: Current 6 cases → Target <2 cases
2. **Converted ↔ Demented confusion**: Current 1 case → Maintain <2 cases
3. **MMSE importance**: Current 20.51% (SHAP) → Maintain <25%

## 🔧 Implementation Steps

### Step 1: Merge Synthetic Data (Priority: HIGH)

```python
# In train_models.py, add:
def load_data_with_synthetic() -> pd.DataFrame:
    # Load original data
    df_original = load_original_data()
    
    # Load synthetic data
    base_dir = Path(__file__).resolve().parents[1]
    synthetic_path = base_dir / "data" / "synthetic" / "synthetic_adversarial_cases.csv"
    
    if synthetic_path.exists():
        df_synthetic = pd.read_csv(synthetic_path)
        # Apply same feature engineering
        df_synthetic = apply_feature_engineering(df_synthetic)
        # Combine
        df = pd.concat([df_original, df_synthetic], ignore_index=True)
        print(f"Combined dataset: {len(df_original)} original + {len(df_synthetic)} synthetic = {len(df)} total")
    else:
        df = df_original
    
    return df
```

### Step 2: Add Sample Weights Based on CDR (Priority: MEDIUM)

```python
# In train_models.py, add sample weights:
def get_sample_weights(X, y):
    weights = np.ones(len(y))
    
    # Increase weight for CDR=0 cases (Nondemented)
    cdr_zero_mask = X['cdr'] == 0.0
    weights[cdr_zero_mask] *= 1.5
    
    # Increase weight for CDR≥1.0 cases (Demented)
    cdr_high_mask = X['cdr'] >= 1.0
    weights[cdr_high_mask] *= 1.5
    
    # Converted class already has high class weight (10x)
    return weights
```

### Step 3: Add Rule Usage Logging (Priority: LOW)

```python
# In api.py, add:
rule_usage_count = 0
total_predictions = 0

@app.post("/predict")
def predict(patient: PatientInput) -> Dict:
    global rule_usage_count, total_predictions
    
    total_predictions += 1
    rule_applied = False
    
    # ... existing prediction code ...
    
    if rule_applied:
        rule_usage_count += 1
    
    # Add to response (optional)
    response["rule_applied"] = rule_applied
    response["rule_usage_percentage"] = (rule_usage_count / total_predictions) * 100
    
    return response
```

## 📈 Expected Outcomes

### After Implementing All Steps:

1. **Converted Recall**: 12.5% → **80-85%** ✅
2. **Rule Usage**: 20% → **<5%** ✅
3. **Model Autonomy**: High - rules only for extreme edge cases ✅
4. **Overall Accuracy**: Maintain 90%+ ✅

## 🎯 Success Criteria

The model will be considered "rock-solid" when:

- ✅ Converted recall ≥ 80%
- ✅ Rule usage < 5%
- ✅ MMSE importance < 25% (SHAP)
- ✅ CDR remains primary feature
- ✅ Accuracy ≥ 90%
- ✅ ROC-AUC ≥ 96%

## 📝 Notes

- **Synthetic data** helps model learn rule logic internally
- **Sample weights** emphasize important CDR patterns
- **Rule logging** helps track progress
- **Confusion matrix** should be re-checked after each improvement

---

**Current Status:** Ready to implement Step 1 (merge synthetic data) and retrain model.

**Next Command:**
```bash
python src/generate_synthetic_data.py  # Already done
# Then modify train_models.py to use synthetic data
# Then retrain: python src/train_models.py
```

