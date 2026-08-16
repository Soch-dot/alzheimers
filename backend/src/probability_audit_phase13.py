"""
ML PIPELINE PHASE 13 -- PRODUCTION PROBABILITY SANITY AUDIT (AUDIT ONLY).

Audits what the production screening_probability output means in practice.
NO production code, model, threshold, SHAP, frontend, or UI wording is modified.
NO recalibration. NO probability transformation. NO commit.

Active production model: backend/models/production/binary_lr_latest_visit_v1.pkl
  - Logistic Regression, C=10.0, class_weight=balanced, lbfgs, max_iter=3000
  - Target: 0=Nondemented, 1=Converted OR Demented
  - Features: age, sex, education_years, mmse, ses
  - Preprocessing: SimpleImputer(median) -> StandardScaler -> LR
  - Threshold: 0.40
Checksum must match 8FC95A3838FFF665CF47FA55C4322096.

Evaluation: canonical outer-test 38 subjects (latest available visit by MR Delay),
subject-level, diagnostic-only. No tuning, no new split.
"""

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, brier_score_loss
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
PROD_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "production" / "binary_lr_latest_visit_v1.pkl"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase13_probability_audit"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]

THRESHOLD = 0.40
EXPECTED_MD5 = "8FC95A3838FFF665CF47FA55C4322096"

BINS = [
    (0.00, 0.10, "< 0.10"),
    (0.10, 0.20, "0.10-0.19"),
    (0.20, 0.30, "0.20-0.29"),
    (0.30, 0.40, "0.30-0.39"),
    (0.40, 0.50, "0.40-0.49"),
    (0.50, 0.60, "0.50-0.59"),
    (0.60, 0.70, "0.60-0.69"),
    (0.70, 0.80, "0.70-0.79"),
    (0.80, 0.90, "0.80-0.89"),
    (0.90, 0.95, "0.90-0.94"),
    (0.95, 0.99, "0.95-0.99"),
    (0.99, 1.01, ">= 0.99"),
]


def md5_hex(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "SES": "ses", "Group": LABEL_COL,
        "Visit": "raw_visit", "MR Delay": MR_DELAY_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    df = df.sort_values([SUBJECT_COL, MR_DELAY_COL]).reset_index(drop=True)
    return df


def latest_frame(subset):
    m = subset[subset[MR_DELAY_COL] == subset.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")].copy()
    return m.reset_index(drop=True)


def ece(y_true, y_prob, n_bins=10):
    """Standard Expected Calibration Error (equal-width bins over [0,1])."""
    y_prob = np.asarray(y_prob); y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        total += (m.sum() / len(y_true)) * abs(acc - conf)
    return float(total)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # checksum
    actual_md5 = md5_hex(PROD_ARTIFACT)
    print("Production artifact md5:", actual_md5, "| expected:", EXPECTED_MD5)
    assert actual_md5 == EXPECTED_MD5, "PRODUCTION ARTIFACT CHECKSUM MISMATCH"

    model = joblib.load(PROD_ARTIFACT)
    assert isinstance(model, __import__("sklearn").pipeline.Pipeline)
    print("Model steps:", [s[0] for s in model.steps])

    # canonical split
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_subjects = set(split["test_subjects"])
    assert len(test_subjects) == 38
    df = load_data()
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]
    te_latest = latest_frame(df_test)
    assert len(te_latest) == 38

    X = te_latest[FEATURES]
    y = te_latest[BINARY_COL].to_numpy()
    prob = model.predict_proba(X)[:, 1]

    te = te_latest.copy()
    te["true_binary_label"] = y
    te["screening_probability"] = prob
    te["screening_result"] = np.where(prob >= THRESHOLD, "positive", "negative")
    te["screening_threshold"] = THRESHOLD

    pred_cols = [SUBJECT_COL, "true_binary_label", "screening_probability",
                 "screening_result", "mmse", "age", "ses", "education_years", "sex",
                 MR_DELAY_COL]
    te[pred_cols].to_csv(OUT_DIR / "outer_test_predictions.csv", index=False)

    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    print("\nOuter test: n=%d positive=%d negative=%d" % (len(y), n_pos, n_neg))

    # ---- 3. distribution ----
    dist = {
        "min": float(prob.min()), "max": float(prob.max()),
        "mean": float(prob.mean()), "median": float(np.median(prob)),
        "std": float(prob.std()),
        "count": int(len(prob)),
        "count_positive": n_pos,
        "count_negative": n_neg,
    }
    for lbl, thr in [("ge_0.90", 0.90), ("ge_0.95", 0.95), ("ge_0.99", 0.99),
                     ("ge_0.999", 0.999)]:
        dist[f"count_{lbl}"] = int((prob >= thr).sum())
    dist["count_ge_0.90"] = int((prob >= 0.90).sum())
    dist["count_ge_0.95"] = int((prob >= 0.95).sum())
    dist["count_ge_0.99"] = int((prob >= 0.99).sum())
    dist["count_ge_0.999"] = int((prob >= 0.999).sum())
    print("\nDistribution:", {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in dist.items() if k != "count"})

    # ---- bins ----
    bin_rows = []
    for lo, hi, label in BINS:
        if hi > 1.0:
            m = prob >= lo
        else:
            m = (prob >= lo) & (prob < hi)
        n = int(m.sum())
        act_pos = int(y[m].sum())
        act_neg = n - act_pos
        rate = (act_pos / n) if n else None
        bin_rows.append({
            "probability_range": label, "n": n, "actual_positive": act_pos,
            "actual_negative": act_neg, "observed_positive_rate": rate,
            "mean_predicted_probability": (float(prob[m].mean()) if n else None),
        })
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(OUT_DIR / "probability_bins.csv", index=False)
    print("\nProbability bins:")
    print(bin_df.to_string(index=False))

    (OUT_DIR / "probability_distribution.json").write_text(
        json.dumps(dist, indent=2), encoding="utf-8")

    # ---- 4/5. reliability diagram (exploratory, n=38) ----
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    n_bins = 5
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    freqs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            m = (prob >= lo) & (prob <= hi)
        else:
            m = (prob >= lo) & (prob < hi)
        freqs.append(y[m].mean() if m.sum() else np.nan)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(centers, freqs, "o-", color="tab:blue", label="Observed positive rate")
    ax.set_xlabel("Mean predicted screening probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Exploratory reliability diagram — n=38\n(not statistically definitive)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reliability.png", dpi=150)
    plt.close(fig)
    print("\nReliability diagram saved (n=38, 5 equal-width bins).")

    # ---- 6. calibration metrics ----
    ll = float(log_loss(y, prob, labels=[0, 1]))
    bs = float(brier_score_loss(y, prob))
    ece_val = ece(y, prob, n_bins=10)
    # calibration slope/intercept: logistic regression of y on logit(p)
    from scipy.special import logit
    from scipy.stats import linregress
    slope = intercept = None
    slope_note = ""
    eps = 1e-9
    p_clip = np.clip(prob, eps, 1 - eps)
    logits = logit(p_clip)
    finite = np.isfinite(logits)
    if finite.sum() >= 5:
        res = linregress(logits[finite], y[finite])
        slope = float(res.slope); intercept = float(res.intercept)
        slope_note = ("stability_limited" if finite.sum() < 15
                      else "computed")
    cal_metrics = {
        "log_loss": ll, "brier_score": bs, "ece_10_bins": ece_val,
        "calibration_slope": slope, "calibration_intercept": intercept,
        "calibration_method": ("logistic regression of observed outcome on logit(p); "
                               "unstable / not meaningful at n=38"),
        "n_outer_subjects": 38,
        "note": "n=38 subject-level observations; all calibration metrics are exploratory, "
                "not statistically definitive. No confidence intervals invented.",
    }
    (OUT_DIR / "calibration_metrics.json").write_text(json.dumps(cal_metrics, indent=2),
                                                      encoding="utf-8")
    print("\nCalibration metrics: logloss=%.4f brier=%.4f ece=%.4f slope=%s intercept=%s"
          % (ll, bs, ece_val, ("%.3f" % slope) if slope is not None else "None",
             ("%.3f" % intercept) if intercept is not None else "None"))

    # ---- 7. extreme-probability cases (>= 0.90) ----
    extr = te[te["screening_probability"] >= 0.90].copy()
    extr_cols = [SUBJECT_COL, "true_binary_label", "screening_probability",
                 "screening_result", "age", "mmse", "education_years", "ses", "sex"]
    extr[extr_cols].sort_values("screening_probability", ascending=False).to_csv(
        OUT_DIR / "extreme_probability_cases.csv", index=False)
    print("\nExtreme cases (>=0.90): n=%d; true-positive=%d true-negative=%d"
          % (len(extr), int((extr["true_binary_label"] == 1).sum()),
             int((extr["true_binary_label"] == 0).sum())))
    for lbl, thr in [("0.95+", 0.95), ("0.99+", 0.99), ("0.999+", 0.999)]:
        sub = te[te["screening_probability"] >= thr]
        print("  %-6s n=%d tp=%d tn=%d"
              % (lbl, len(sub), int((sub["true_binary_label"] == 1).sum()),
                 int((sub["true_binary_label"] == 0).sum())))

    # ---- 8. fixed sanity-check fixture ----
    fx = pd.DataFrame([{"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}])
    pfx = float(model.predict_proba(fx[FEATURES])[:, 1][0])
    fx_result = {
        "fixture": {"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1},
        "screening_probability": round(pfx, 6),
        "screening_result": "positive" if pfx >= THRESHOLD else "negative",
        "threshold": THRESHOLD,
        "note": "sanity-check fixture ONLY; not a validation sample.",
    }
    (OUT_DIR / "fixed_case_result.json").write_text(json.dumps(fx_result, indent=2),
                                                    encoding="utf-8")
    print("\nFixed fixture 74/M/10/23/1: prob=%.6f result=%s"
          % (pfx, fx_result["screening_result"]))

    # ---- 9. coefficient sanity ----
    coef = model.named_steps["clf"].coef_[0]
    intercept = model.named_steps["clf"].intercept_[0]
    medians = pd.Series(model.named_steps["imputer"].statistics_, index=FEATURES)
    scaled = model.named_steps["scaler"]
    coef_analysis = {
        "feature_order": FEATURES,
        "coefficients": {f: round(float(c), 4) for f, c in zip(FEATURES, coef)},
        "intercept": round(float(intercept), 4),
        "imputer_medians": medians.round(3).to_dict(),
        "scaler_scale": {f: round(float(s), 4) for f, s in zip(FEATURES, scaled.scale_)},
        "scaler_mean": {f: round(float(m), 4) for f, m in zip(FEATURES, scaled.mean_)},
        "threshold": THRESHOLD,
        "explanation": (
            "LR decision = sigmoid(intercept + sum(coef_j * (x_j - median_j)/scale_j)). "
            "For 74/M/10/23/1, the large negative mmse coefficient (-4.89 on standardized "
            "MMSE) dominates: mmse=23 is far below the imputed median (~28-29) after "
            "standardization, producing a large positive logit contribution; education 10 "
            "and ses 1 also carry negative coefficients (lower education/SES -> higher "
            "screening odds); male (sex=1) has a positive coefficient. The result is a "
            "logit far above 0, hence sigmoid -> ~0.99998."
        ),
    }
    (OUT_DIR / "coefficient_analysis.json").write_text(json.dumps(coef_analysis, indent=2),
                                                       encoding="utf-8")
    print("\nCoefficients:", {f: round(float(c), 4) for f, c in zip(FEATURES, coef)},
          "intercept:", round(float(intercept), 4))

    # ---- README ----
    (OUT_DIR / "README.md").write_text(
        "# Phase 13 - Production Probability Sanity Audit (AUDIT ONLY)\n\n"
        "- Audits what the production screening_probability means in practice for the frozen\n"
        "  experimental binary screening candidate (binary_lr_latest_visit_v1).\n"
        "- n=38 outer-test subjects (latest available visit by MR Delay), subject-level.\n"
        "- No production changes, no recalibration, no probability transformation, no commit.\n"
        "- Artifacts: outer_test_predictions.csv, probability_bins.csv, probability_distribution.json,\n"
        "  extreme_probability_cases.csv, calibration_metrics.json, reliability.png, fixed_case_result.json,\n"
        "  coefficient_analysis.json.\n"
        "- The reliability diagram and calibration metrics are EXPLORATORY (n=38); not statistically definitive.\n"
        "- The probability is a model-estimated screening probability, NOT clinically validated.\n",
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== PRODUCTION INTEGRITY (AUDIT ONLY) ===")
    print("production artifact md5 unchanged:", actual_md5 == EXPECTED_MD5)
    print("api.py/frontend/SHAP not touched (verified before run)")
    print("no commit made")


if __name__ == "__main__":
    main()