"""
ML PIPELINE PHASE 11 -- EXPERIMENTAL MODEL LOCK + REPRODUCIBLE VALIDATION PACKAGE.

Freezes the research configuration established in Phases 1-10 into a
reproducible, research-only package. This is STILL NOT a production
integration.

FROZEN CONFIGURATION:
  Target       : 0 = Nondemented, 1 = Converted OR Demented
  Model        : LogisticRegression (C=10.0, class_weight=balanced,
                 solver=lbfgs, max_iter=3000)
  Features     : age, sex, education_years, mmse, ses  (exact order)
  NO           : CDR, MRI features, visit_count, prev_mmse, mmse_delta,
                 future information
  Visit policy : single visit -> use it; multiple visits -> latest by MR Delay.
  Threshold    : 0.40  (experimental screening threshold; frozen)
  Probability  : raw binary LR probability = "model-estimated probability of a
                 dementia-related outcome" (NOT Alzheimer's/diagnosis/certainty)

Leakage-safe preprocessing: SimpleImputer(median) -> StandardScaler ->
LogisticRegression, fit on training subjects ONLY. Model rebuilt from scratch;
the production RF artifact (best_model.pkl) is NOT loaded or reused.
Outer 38 subjects untouched until final evaluation.

SHAP: research-only LinearExplainer for the exact frozen LR pipeline using the
exact same preprocessing (imputer+scaler fitted on training latest visits) and
exact same feature order. Production SHAP artifacts are NOT modified.
"""

from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
)
import joblib

import sklearn
import shap

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "final_candidate"
SHAP_DIR = OUT_DIR / "shap"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"

FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
FEATURE_DISPLAY = ["age", "sex", "education_years", "mmse", "ses"]

THRESHOLD = 0.40
N_BOOT = 2000
BOOT_ALPHA = 0.05
NP_SEED = 42
CV_RANDOM_STATE = 42  # documented; CV not rerun in this phase

MODEL_CONFIG = {
    "model_type": "LogisticRegression",
    "C": 10.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 3000,
    "random_state": 42,
}


def subject_id_hash(ids, label):
    canonical = "\n".join(sorted(ids))
    return {f"{label}_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            f"{label}_count": len(ids)}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "SES": "ses", "Group": LABEL_COL,
        "Visit": "raw_visit", "MR Delay": MR_DELAY_COL,
        "eTIV": "etiv", "nWBV": "nwbv", "ASF": "asf",
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    df = df.sort_values([SUBJECT_COL, MR_DELAY_COL]).reset_index(drop=True)
    df["temporal_rank"] = df.groupby(SUBJECT_COL).cumcount() + 1
    df["visit_count"] = df.groupby(SUBJECT_COL)[SUBJECT_COL].transform("size")
    return df


def latest_frame(subset):
    m = subset[subset[MR_DELAY_COL] == subset.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")].copy()
    return m.reset_index(drop=True)


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs",
                                   random_state=42)),
    ]).set_params(clf__C=10.0, clf__class_weight="balanced")


def metrics_at_threshold(y, p, t=THRESHOLD):
    y = np.asarray(y); p = np.asarray(p)
    y_pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    uniq = len(np.unique(y))
    return {
        "threshold": float(t), "n": int(len(y)),
        "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
        "sensitivity": float(sens), "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
        "ppv": float(ppv), "npv": float(npv),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, p)) if uniq == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if uniq == 2 else None,
        "brier": float(brier_score_loss(y, p)) if uniq == 2 else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if uniq == 2 else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y, p, n_boot=N_BOOT, t=THRESHOLD, seed=NP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    y = np.asarray(y); p = np.asarray(p)
    stats = {k: [] for k in ["sensitivity", "specificity", "balanced_accuracy",
                             "ppv", "npv"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]; pb = p[idx]
        y_pred = (pb >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        stats["sensitivity"].append(sens)
        stats["specificity"].append(spec)
        stats["balanced_accuracy"].append((sens + spec) / 2)
        stats["ppv"].append(ppv)
        stats["npv"].append(npv)
    out = {}
    for k, vals in stats.items():
        vals = np.asarray(vals)
        lo, hi = np.percentile(vals, [100 * BOOT_ALPHA / 2, 100 * (1 - BOOT_ALPHA / 2)])
        out[k] = {
            "point_estimate": round(float(np.mean(vals)), 4),
            "ci_lower_95": round(float(lo), 4),
            "ci_upper_95": round(float(hi), 4),
            "n_bootstraps": int(n_boot),
        }
    return out


def group_split_stats(frame):
    return {
        "n": int(len(frame)),
        "Nondemented": int((frame[LABEL_COL] == 0).sum()),
        "Converted": int((frame[LABEL_COL] == 1).sum()),
        "Demented": int((frame[LABEL_COL] == 2).sum()),
    }


def missing_report(frame, tag):
    return {
        tag: {
            "ses_nan": int(frame["ses"].isna().sum()),
            "mmse_nan": int(frame["mmse"].isna().sum()),
            "mmse_nan_subjects": sorted(frame.loc[frame["mmse"].isna(), SUBJECT_COL].unique().tolist()),
            "handling": "SimpleImputer(strategy=median) fitted on TRAINING subjects' latest-visit rows only",
        }
    }


def main():
    np.random.seed(NP_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = sorted(split["train_subjects"])
    test_subjects = sorted(split["test_subjects"])
    assert len(set(train_subjects) & set(test_subjects)) == 0
    assert len(train_subjects) == 112 and len(test_subjects) == 38

    df = load_data()
    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]

    tr_latest = latest_frame(df_train)   # 112 subjects, 1 latest-visit row each
    te_latest = latest_frame(df_test)    # 38 subjects, 1 latest-visit row each
    assert len(tr_latest) == 112 and len(te_latest) == 38

    # ---- rebuild model from scratch (training subjects, latest visits only) ----
    X_train = tr_latest[FEATURES]
    y_train = tr_latest[BINARY_COL]
    model = make_pipeline()
    model.fit(X_train, y_train)

    # ---- final evaluation on outer 38 (never touched during development) ----
    X_test = te_latest[FEATURES]
    y_test = te_latest[BINARY_COL]
    prob_test = model.predict_proba(X_test)[:, 1]
    te_out = te_latest.copy()
    te_out["probability"] = prob_test
    te_out["predicted_binary"] = (prob_test >= THRESHOLD).astype(int)

    metrics = metrics_at_threshold(y_test, prob_test)
    print("=== OUTER TEST (38 subjects, latest visit, threshold 0.40) ===")
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
              "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]:
        print("  %-18s %s" % (k, metrics[k]))
    print("  TP=%d TN=%d FP=%d FN=%d" % (metrics["tp"], metrics["tn"],
                                          metrics["fp"], metrics["fn"]))

    # ---- converted / demented / nondemented analysis ----
    conv = te_out[te_out[LABEL_COL] == 1]
    dem = te_out[te_out[LABEL_COL] == 2]
    non = te_out[te_out[LABEL_COL] == 0]
    conv_det = int((conv["predicted_binary"] == 1).sum())
    dem_det = int((dem["predicted_binary"] == 1).sum())
    non_fp = int((non["predicted_binary"] == 1).sum())
    print("\nConverted detected=%d/%d missed=%d | Demented detected=%d/%d missed=%d | Nondemented FP=%d"
          % (conv_det, len(conv), len(conv) - conv_det, dem_det, len(dem),
             len(dem) - dem_det, non_fp))

    # ---- bootstrap CIs ----
    ci = bootstrap_ci(y_test, prob_test)
    print("\n=== BOOTSTRAP CIs (n=38 outer subjects, 2000 subject-level resamples) ===")
    for k, v in ci.items():
        print("  %-18s %.4f [%.4f, %.4f]" % (k, v["point_estimate"],
                                              v["ci_lower_95"], v["ci_upper_95"]))

    # ---- artifacts ----
    # model artifact
    artifact_path = OUT_DIR / "binary_lr_latest_visit.pkl"
    joblib.dump(model, artifact_path)

    # outer predictions csv
    pred_cols = [SUBJECT_COL, LABEL_COL, BINARY_COL, "age", "sex", "education_years",
                 "mmse", "ses", "visit_count", MR_DELAY_COL, "probability", "predicted_binary"]
    te_out[pred_cols].to_csv(OUT_DIR / "outer_test_predictions.csv", index=False)

    # confusion matrix csv
    cm = confusion_matrix(y_test, te_out["predicted_binary"], labels=[0, 1])
    pd.DataFrame(cm, index=["True Nondemented", "True Converted/Demented"],
                 columns=["Pred Nondemented", "Pred Converted/Demented"]).to_csv(
        OUT_DIR / "confusion_matrix.csv")

    # outer metrics json
    (OUT_DIR / "outer_test_metrics.json").write_text(json.dumps({
        **metrics,
        "converted": {"total": int(len(conv)), "detected": conv_det, "missed": len(conv) - conv_det},
        "demented": {"total": int(len(dem)), "detected": dem_det, "missed": len(dem) - dem_det},
        "nondemented_false_positives": non_fp,
    }, indent=2), encoding="utf-8")

    # bootstrap json
    (OUT_DIR / "bootstrap_confidence_intervals.json").write_text(json.dumps({
        "method": "subject-level bootstrap resampling (resample the 38 outer subjects with replacement)",
        "n_outer_subjects": 38,
        "n_bootstraps": N_BOOT,
        "confidence_level": "95% (percentile)",
        "seed": NP_SEED,
        "interval_estimates": ci,
        "caveat": "n=38 outer subjects; intervals are wide; no population-level statistical certainty claimed.",
    }, indent=2), encoding="utf-8")

    # ---- model metadata ----
    model_metadata = {
        "model_type": "LogisticRegression (experimental binary screening candidate)",
        "hyperparameters": MODEL_CONFIG,
        "feature_order": FEATURE_DISPLAY,
        "target_mapping": {"0": "Nondemented", "1": "Converted OR Demented"},
        "threshold": THRESHOLD,
        "visit_policy": {
            "single_visit": "If a patient has only one available assessment, that assessment is used.",
            "multiple_visits": ("If multiple assessments are available, the latest assessment by "
                                "MR Delay is used."),
            "temporal_ordering": "MR Delay (NOT raw Visit)",
        },
        "scientific_note": ("Latest-visit classification is not future-conversion prediction because "
                            "the OASIS Group label is a final subject-level classification."),
        "probability_semantics": ("model-estimated probability of a dementia-related outcome "
                                  "(Nondemented vs Converted/Demented); raw binary LR probability."),
        "preprocessing": "SimpleImputer(median) -> StandardScaler -> LogisticRegression (fit on training subjects only)",
        "train_subjects": len(train_subjects),
        "outer_test_subjects": len(test_subjects),
        "dataset_rows": int(len(df)),
        "dataset_subjects": int(df[SUBJECT_COL].nunique()),
        "training_data": "latest available visit per training subject (112 subjects)",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_seeds": {"numpy": NP_SEED, "cv": CV_RANDOM_STATE},
        "validation_methodology": ("Canonical 112/38 subject split (Phase 2); outer 38 untouched until "
                                   "final evaluation; no tuning on outer test."),
        "missing_data": {**missing_report(tr_latest, "train_latest"),
                         **missing_report(te_latest, "outer_test_latest")},
        "excluded_features": ["CDR (target/post-diagnostic leakage)", "eTIV/nWBV/ASF (research-only, not collected by product)",
                              "visit_count", "prev_mmse", "mmse_delta", "future information"],
        "training_subject_counts": group_split_stats(tr_latest),
        "outer_test_subject_counts": group_split_stats(te_latest),
    }
    (OUT_DIR / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2),
                                                 encoding="utf-8")

    # ---- reproducibility manifest ----
    reproducibility = {
        "dataset_filename": "Dataset.csv",
        "dataset_row_count": int(len(df)),
        "dataset_subject_count": int(df[SUBJECT_COL].nunique()),
        "canonical_split_path": str(SPLIT_PATH),
        "training_subject_id_hash": subject_id_hash(train_subjects, "training_subjects"),
        "test_subject_id_hash": subject_id_hash(test_subjects, "test_subjects"),
        "model_config": MODEL_CONFIG,
        "feature_order": FEATURE_DISPLAY,
        "threshold": THRESHOLD,
        "visit_policy": "latest available visit by MR Delay; single visit used as-is",
        "evaluation_metrics": {k: metrics[k] for k in
                               ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
                                "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss",
                                "tp", "tn", "fp", "fn"]},
        "code_path": str(Path(__file__).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "shap_version": shap.__version__,
        "note": "Research-only package; no sensitive patient data beyond subject IDs and model outputs stored.",
    }
    (OUT_DIR / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2),
                                                  encoding="utf-8")

    # ---- research-only SHAP (LinearExplainer, exact frozen pipeline) ----
    # Apply the same imputer+scaler (fitted on training latest visits) so the
    # explanation uses the exact same preprocessing and feature order.
    X_train_scaled = model.named_steps["imputer"].transform(X_train)
    X_train_scaled = model.named_steps["scaler"].transform(X_train_scaled)
    X_test_scaled = model.named_steps["imputer"].transform(X_test)
    X_test_scaled = model.named_steps["scaler"].transform(X_test_scaled)

    explainer = shap.LinearExplainer(model.named_steps["clf"], X_train_scaled,
                                     feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_scaled)

    global_importance = np.abs(shap_values).mean(axis=0)
    importance_table = pd.DataFrame({
        "feature": FEATURE_DISPLAY,
        "mean_abs_shap": global_importance,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_table.to_csv(SHAP_DIR / "global_importance.csv", index=False)

    contributions = pd.DataFrame(shap_values, columns=FEATURE_DISPLAY)
    contributions[SUBJECT_COL] = te_out[SUBJECT_COL].values
    contributions["probability"] = prob_test
    contributions.to_csv(SHAP_DIR / "feature_contributions.csv", index=False)

    (SHAP_DIR / "feature_names.json").write_text(json.dumps({
        "feature_names": FEATURE_DISPLAY,
        "explainer": "shap.LinearExplainer on frozen LogisticRegression (raw log-odds contributions)",
        "preprocessing": "same SimpleImputer(median)->StandardScaler fitted on training latest visits",
        "background_data": "112 training subjects' scaled latest-visit rows",
        "explained_rows": "38 outer-test subjects' scaled latest-visit rows",
    }, indent=2), encoding="utf-8")

    # ---- README ----
    (OUT_DIR / "README.md").write_text(
        "# Phase 11 - Experimental Model Lock + Reproducible Validation Package (Research Only)\n\n"
        "- Freezes the experimental binary screening candidate: LR C=10 balanced, features age/sex/education_years/mmse/ses.\n"
        "- Target: 0=Nondemented, 1=Converted|Demented. Threshold 0.40. Visit policy: latest available by MR Delay; single visit used as-is.\n"
        "- Probability semantics: model-estimated probability of a dementia-related outcome (NOT diagnosis/certainty).\n"
        "- Latest-visit classification is NOT future-conversion prediction (OASIS Group is a final subject-level label).\n"
        "- Rebuilt from scratch on the 112 training subjects' latest visits; outer 38 untouched until final evaluation.\n"
        "- Research-only SHAP (LinearExplainer) for the frozen pipeline; production SHAP and /predict unchanged.\n"
        "- NOT a production integration; no production changes; no commits.\n"
        "- LIMITATIONS: n=150 subjects, 14 Converted, 4 Converted in outer test, wide bootstrap CIs, no clinical validation,\n"
        "  no prospective conversion prediction; Converted cases with normal MMSE can be indistinguishable from Nondemented.\n",
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            print("  ", f)

    print("\n=== SANITY CHECKS ===")
    print("split overlap == 0:", len(set(train_subjects) & set(test_subjects)) == 0)
    print("outer test subjects == 38:", len(te_latest) == 38)
    print("model is fresh pipeline (not best_model.pkl):", isinstance(model, Pipeline))
    print("cdr not in features:", "cdr" not in FEATURES)
    print("threshold == 0.40:", THRESHOLD == 0.40)
    print("imputation fit on train only: True (inside pipeline fit on tr_latest)")

    # report summary dict for the final report
    summary = {
        "metrics": metrics,
        "converted": {"total": len(conv), "detected": conv_det, "missed": len(conv) - conv_det,
                      "subjects": sorted(conv[SUBJECT_COL].tolist())},
        "demented": {"total": len(dem), "detected": dem_det, "missed": len(dem) - dem_det},
        "nondemented_fp": non_fp,
        "bootstrap": {k: v["point_estimate"] for k, v in ci.items()},
        "bootstrap_ci": {k: [v["ci_lower_95"], v["ci_upper_95"]] for k, v in ci.items()},
        "importance": importance_table.to_dict("records"),
    }
    (OUT_DIR / "_summary_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary report written to _summary_report.json")


if __name__ == "__main__":
    main()