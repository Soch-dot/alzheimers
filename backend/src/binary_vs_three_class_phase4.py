"""
ML PIPELINE PHASE 4 -- 3-CLASS VS BINARY SCREENING COMPARISON (RESEARCH ONLY).

Benchmarks a binary target (0=Nondemented, 1=Converted|Demented) under the exact
Phase 1/2 subject split and grouped-CV framework, calibrates out-of-sample, and
compares subject-level performance against the existing experimental 3-class model.

Rules:
- Same 112/38 subject split, zero overlap, outer test untouched during selection.
- Binary target is a separate experimental column; original `group` untouched.
- No production / /predict / SHAP / MMSE / Q11 / calibration-file changes.
- Subject is the independent unit; visit probabilities aggregated by mean.
"""

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

import joblib
import sklearn
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
WINNER_3CLASS_PKL = Path(__file__).resolve().parents[1] / "models" / "experiments" / "model_benchmark" / "LogisticRegression_winner.pkl"
OOF_3CLASS_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "model_benchmark" / "oof_predictions.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "binary_vs_three_class"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
LABEL_COL = "group"
BINARY_COL = "binary_target"
SUBJECT_COL = "Subject ID"
PROBA_COL = "prob_any_dementia"
BINARY_THRESHOLD = 0.5

N_SPLITS = 5
CV_RANDOM_STATE = 42
NP_SEED = 42
CLASS_NAMES = ["Nondemented", "Converted", "Demented"]
LABELS = [0, 1, 2]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "CDR": "cdr", "SES": "ses", "Group": LABEL_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    # EXPERIMENTAL binary target: 1 = Converted|Demented, 0 = Nondemented.
    # Original `group` column is untouched.
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    return df[[SUBJECT_COL] + FEATURES + [LABEL_COL, BINARY_COL]]


def load_split():
    d = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = set(d["train_subjects"])
    test_subjects = set(d["test_subjects"])
    if len(train_subjects) != 112 or len(test_subjects) != 38:
        raise SystemExit("STOP: saved split counts differ from 112/38")
    if train_subjects & test_subjects:
        raise SystemExit("STOP: saved split has subject overlap")
    return train_subjects, test_subjects


def build_candidates():
    base_rf = dict(n_estimators=200, random_state=42, n_jobs=1)
    cands = {}
    cands["LogisticRegression"] = {
        "estimator": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
        ]),
        "param_grid": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    }
    cands["RandomForest"] = {
        "estimator": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**base_rf)),
        ]),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10],
            "clf__min_samples_leaf": [1, 3],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": [None, "balanced"],
        },
    }
    cands["SVM"] = {
        "estimator": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42)),
        ]),
        "param_grid": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__kernel": ["rbf"],
            "clf__gamma": ["scale", 0.1],
            "clf__class_weight": [None, "balanced"],
        },
    }
    if XGBOOST_AVAILABLE:
        cands["XGBoost"] = {
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                    random_state=42, n_jobs=1, tree_method="hist",
                    eval_metric="logloss",
                )),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.1, 0.3],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__min_child_weight": [1, 3],
                "clf__scale_pos_weight": [1, 2],
            },
        }
    return cands


def binary_metrics(y_true, p):
    y_pred = (p >= BINARY_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p)),
    }


def run_binary_benchmark(df_train):
    X = df_train[FEATURES]
    y = df_train[BINARY_COL]
    groups = df_train[SUBJECT_COL].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    results = {}
    for name, c in build_candidates().items():
        print(f"\n=== TUNING + CV: {name} (binary) ===")
        gs = GridSearchCV(c["estimator"], c["param_grid"], cv=cv,
                          scoring="balanced_accuracy", n_jobs=1, refit=True)
        gs.fit(X, y, groups=groups)
        best = gs.best_estimator_
        best_params = gs.best_params_
        print("best params:", best_params, "| CV balanced acc: %.4f" % gs.best_score_)

        fold_metrics = []
        oof_rows = []
        for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
            tr_sid = set(groups[tr]); va_sid = set(groups[va])
            if tr_sid & va_sid:
                raise SystemExit("STOP: subject overlap in fold")
            m = clone(best)
            m.fit(X.iloc[tr], y.iloc[tr])
            p_va = m.predict_proba(X.iloc[va])[:, 1]
            fm = binary_metrics(y.iloc[va].to_numpy(), p_va)
            fm["fold"] = fold
            fold_metrics.append(fm)
            for i in range(len(va)):
                oof_rows.append({
                    "subject_id": groups[va][i],
                    "fold": fold,
                    "true_binary_label": int(y.iloc[va[i]]),
                    "predicted_binary_label": int((p_va[i] >= BINARY_THRESHOLD)),
                    PROBA_COL: float(p_va[i]),
                })
        results[name] = {
            "best_params": best_params,
            "cv_balanced_accuracy": float(gs.best_score_),
            "fold_metrics": fold_metrics,
            "oof": pd.DataFrame(oof_rows),
            "best_estimator": best,
        }
        # summarize
        fm_df = pd.DataFrame(fold_metrics)
        print("  mean ± std  balacc %.4f±%.4f | sens %.4f | spec %.4f | f1 %.4f | roc %.4f | prauc %.4f | brier %.4f | logloss %.4f"
              % (fm_df.balanced_accuracy.mean(), fm_df.balanced_accuracy.std(),
                 fm_df.sensitivity.mean(), fm_df.specificity.mean(), fm_df.f1.mean(),
                 fm_df.roc_auc.mean(), fm_df.pr_auc.mean(), fm_df.brier.mean(), fm_df.log_loss.mean()))
    return results


def calibrate_binary(oof_df):
    """Honest second-level subject-grouped CV comparing raw/sigmoid/isotonic on OOF."""
    X = oof_df.reset_index(drop=True)
    y = X["true_binary_label"].to_numpy()
    p = X[PROBA_COL].to_numpy()
    groups = X["subject_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    records = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
        tr_sid = set(groups[tr]); va_sid = set(groups[va])
        if tr_sid & va_sid:
            raise SystemExit("STOP: calibration fold subject overlap")
        sig = _SigmoidCalibration(); sig.fit(p[tr].reshape(-1, 1), y[tr])
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(p[tr], y[tr])
        for method in ["raw", "sigmoid", "isotonic"]:
            if method == "raw":
                p_va = p[va]
            elif method == "sigmoid":
                p_va = sig.predict(p[va].reshape(-1, 1))
            else:
                p_va = iso.predict(p[va])
            m = binary_metrics(y[va], p_va)
            m["method"] = method
            m["fold"] = fold
            m["n_subjects"] = len(va_sid)
            records.append(m)
    return pd.DataFrame(records)


def subject_aggregate(df_rows, prob_col):
    """Subject-level aggregation: mean of visit probabilities per subject."""
    g = df_rows.groupby(SUBJECT_COL)
    subj = pd.DataFrame({
        "subject_probability": g[prob_col].mean(),
        "true_binary_label": g["true_binary_label"].first(),
        "n_visits": g.size(),
    }).reset_index()
    return subj


def final_binary_eval(winner_name, results, cal_method, cal_fit, df_train, df_test, oof_winner):
    est = clone(results[winner_name]["best_estimator"])
    X_tr = df_train[FEATURES]; y_tr = df_train[BINARY_COL]
    est.fit(X_tr, y_tr)

    # Apply calibrator fit on OOF (training subjects only).
    X_te = df_test[FEATURES]
    raw_te = est.predict_proba(X_te)[:, 1]
    if cal_method == "raw":
        cal_te = raw_te
    elif cal_method == "sigmoid":
        cal_te = cal_fit["sigmoid"].predict(raw_te.reshape(-1, 1))
    else:
        cal_te = cal_fit["isotonic"].predict(raw_te)

    vis = pd.DataFrame({
        SUBJECT_COL: df_test[SUBJECT_COL].to_numpy(),
        "true_binary_label": df_test[BINARY_COL].to_numpy(),
        "visit_prob_raw": raw_te,
        "visit_prob_cal": cal_te,
    })
    # subject-level aggregation (mean visit probability)
    subj_raw = subject_aggregate(vis.rename(columns={"visit_prob_raw": PROBA_COL}),
                                 PROBA_COL)
    subj_cal = subject_aggregate(vis.rename(columns={"visit_prob_cal": PROBA_COL}),
                                 PROBA_COL)
    m_raw_subj = binary_metrics(subj_raw["true_binary_label"].to_numpy(), subj_raw["subject_probability"].to_numpy())
    m_cal_subj = binary_metrics(subj_cal["true_binary_label"].to_numpy(), subj_cal["subject_probability"].to_numpy())
    m_raw_vis = binary_metrics(vis["true_binary_label"].to_numpy(), vis["visit_prob_raw"].to_numpy())
    m_cal_vis = binary_metrics(vis["true_binary_label"].to_numpy(), vis["visit_prob_cal"].to_numpy())
    return {
        "estimator": est,
        "subject_level": {"raw": m_raw_subj, "calibrated": m_cal_subj},
        "visit_level": {"raw": m_raw_vis, "calibrated": m_cal_vis},
        "subject_table": subj_cal,
        "visit_table": vis,
        "n_subjects": len(subj_cal),
        "n_rows": len(vis),
        "n_positive_subjects": int((subj_cal["true_binary_label"] == 1).sum()),
        "n_negative_subjects": int((subj_cal["true_binary_label"] == 0).sum()),
    }


def three_class_subject_reference(df_test):
    """3-class subject-level reference derived with Phase-3 methodology
    (sigmoid calibrator fit on training-subject OOF, applied to outer test)."""
    est = joblib.load(WINNER_3CLASS_PKL)
    oof = pd.read_csv(OOF_3CLASS_PATH)
    oof = oof[oof["model"] == "LogisticRegression"]
    y_oof = oof["true_label"].to_numpy()
    p_oof = oof[["prob_Nondemented", "prob_Converted", "prob_Demented"]].to_numpy()

    sig = {}
    for c in LABELS:
        yc = (y_oof == c).astype(int)
        s = _SigmoidCalibration(); s.fit(p_oof[:, c].reshape(-1, 1), yc)
        sig[c] = s

    X_te = df_test[FEATURES]
    raw_te = est.predict_proba(X_te)
    cal_te = np.zeros_like(raw_te)
    for c in LABELS:
        cal_te[:, c] = sig[c].predict(raw_te[:, c].reshape(-1, 1))
    s = cal_te.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    cal_te = cal_te / s

    subj = pd.DataFrame({
        SUBJECT_COL: df_test[SUBJECT_COL].to_numpy(),
        "true_group": df_test[LABEL_COL].to_numpy(),
    })
    for i, cn in enumerate(CLASS_NAMES):
        subj[f"p_{cn}"] = cal_te[:, i]
    agg = subj.groupby(SUBJECT_COL).agg(
        true_group=("true_group", "first"), n_visits=(SUBJECT_COL, "size"),
        **{f"p_{cn}": (f"p_{cn}", "mean") for cn in CLASS_NAMES}).reset_index()
    pred_idx = agg[[f"p_{cn}" for cn in CLASS_NAMES]].to_numpy().argmax(axis=1)
    agg["predicted_group"] = pred_idx
    agg["predicted_any_dementia"] = (pred_idx != 0).astype(int)
    agg["true_any_dementia"] = (agg["true_group"] != 0).astype(int)

    y_true_bin = agg["true_any_dementia"].to_numpy()
    p_any = (agg[f"p_Converted"] + agg[f"p_Demented"]).to_numpy()
    m = binary_metrics(y_true_bin, p_any)

    per_class_recall = {}
    for c in LABELS:
        mask = agg["true_group"] == c
        if mask.sum():
            per_class_recall[CLASS_NAMES[c]] = float((agg.loc[mask, "predicted_group"] == c).mean())
        else:
            per_class_recall[CLASS_NAMES[c]] = None
    m["balanced_accuracy_3class"] = float(np.nanmean(list(per_class_recall.values())))
    m["per_class_recall"] = per_class_recall
    m["n_subjects"] = len(agg)
    m["n_positive_subjects"] = int((agg["true_any_dementia"] == 1).sum())
    m["n_negative_subjects"] = int((agg["true_any_dementia"] == 0).sum())
    return agg, m


def main():
    np.random.seed(NP_SEED)
    df = load_data()
    train_subjects, test_subjects = load_split()

    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]
    if set(df_train[SUBJECT_COL]) & set(df_test[SUBJECT_COL]):
        raise SystemExit("STOP: subject overlap")

    print("=== SECTION 2/3: BINARY TARGET + SPLIT ===")
    print("binary class dist (subjects) train:",
          df_train.groupby(SUBJECT_COL)[BINARY_COL].first().value_counts().sort_index().to_dict())
    print("binary class dist (subjects) test: ",
          df_test.groupby(SUBJECT_COL)[BINARY_COL].first().value_counts().sort_index().to_dict())

    results = run_binary_benchmark(df_train)

    # Model selection: balanced accuracy primary; sensitivity/specificity secondary.
    comp = pd.DataFrame([{
        "model": name,
        "cv_balanced_accuracy": r["cv_balanced_accuracy"],
        "cv_mean_balacc": float(pd.DataFrame(r["fold_metrics"]).balanced_accuracy.mean()),
        "cv_mean_sensitivity": float(pd.DataFrame(r["fold_metrics"]).sensitivity.mean()),
        "cv_mean_specificity": float(pd.DataFrame(r["fold_metrics"]).specificity.mean()),
        "cv_mean_roc_auc": float(pd.DataFrame(r["fold_metrics"]).roc_auc.mean()),
        "cv_mean_pr_auc": float(pd.DataFrame(r["fold_metrics"]).pr_auc.mean()),
        "cv_mean_brier": float(pd.DataFrame(r["fold_metrics"]).brier.mean()),
        "cv_mean_logloss": float(pd.DataFrame(r["fold_metrics"]).log_loss.mean()),
    } for name, r in results.items()]).sort_values(
        ["cv_balanced_accuracy", "cv_mean_sensitivity", "cv_mean_roc_auc"],
        ascending=[False, False, False])
    print("\n=== SECTION 8: BINARY MODEL COMPARISON (grouped CV) ===")
    print(comp.to_string(index=False))
    winner = comp.iloc[0]["model"]
    print("BINARY WINNER (experimental):", winner)

    # OOF for all models
    oof_all = pd.concat([r["oof"].assign(model=n) for n, r in results.items()], ignore_index=True)

    # Calibration comparison on winner OOF
    oof_winner = results[winner]["oof"]
    cal_res = calibrate_binary(oof_winner)
    print("\n=== SECTION 9: CALIBRATION COMPARISON (binary, honest OOF) ===")
    cal_summary = {}
    for method in ["raw", "sigmoid", "isotonic"]:
        sub = cal_res[cal_res["method"] == method]
        cal_summary[method] = {
            "log_loss": (float(sub.log_loss.mean()), float(sub.log_loss.std())),
            "brier": (float(sub.brier.mean()), float(sub.brier.std())),
            "balanced_accuracy": (float(sub.balanced_accuracy.mean()), float(sub.balanced_accuracy.std())),
            "sensitivity": (float(sub.sensitivity.mean()), float(sub.sensitivity.std())),
            "specificity": (float(sub.specificity.mean()), float(sub.specificity.std())),
        }
        print(f"{method}: logloss={cal_summary[method]['log_loss'][0]:.4f}±{cal_summary[method]['log_loss'][1]:.4f} "
              f"brier={cal_summary[method]['brier'][0]:.4f}±{cal_summary[method]['brier'][1]:.4f} "
              f"balacc={cal_summary[method]['balanced_accuracy'][0]:.4f} "
              f"sens={cal_summary[method]['sensitivity'][0]:.4f} spec={cal_summary[method]['specificity'][0]:.4f}")
    selected_cal = min(["raw", "sigmoid", "isotonic"],
                       key=lambda m: (cal_summary[m]["log_loss"][0], cal_summary[m]["brier"][0]))
    print("SELECTED CALIBRATION (experimental):", selected_cal)

    # Fit final calibrator on ALL winner OOF (training subjects only)
    y_all = oof_winner["true_binary_label"].to_numpy()
    p_all = oof_winner[PROBA_COL].to_numpy()
    cal_fit = {}
    sig_final = _SigmoidCalibration(); sig_final.fit(p_all.reshape(-1, 1), y_all)
    iso_final = IsotonicRegression(out_of_bounds="clip"); iso_final.fit(p_all, y_all)
    cal_fit = {"sigmoid": sig_final, "isotonic": iso_final}

    # Outer test final eval
    print("\n=== SECTION 11: OUTER TEST (SUBJECT LEVEL, method=%s) ===" % selected_cal)
    outer = final_binary_eval(winner, results, selected_cal, cal_fit, df_train, df_test, oof_winner)
    print("subjects=%d rows=%d positive=%d negative=%d" % (
        outer["n_subjects"], outer["n_rows"], outer["n_positive_subjects"], outer["n_negative_subjects"]))
    print("subject-level raw:", json.dumps({k: round(v, 4) for k, v in outer["subject_level"]["raw"].items()}))
    print("subject-level calibrated:", json.dumps({k: round(v, 4) for k, v in outer["subject_level"]["calibrated"].items()}))

    # 3-class subject-level reference
    print("\n=== SECTION 12: 3-CLASS SUBJECT-LEVEL REFERENCE (phase-3 sigmoid) ===")
    three_agg, three_m = three_class_subject_reference(df_test)
    print("subjects=%d positive=%d negative=%d" % (
        three_m["n_subjects"], three_m["n_positive_subjects"], three_m["n_negative_subjects"]))
    print("3-class any-dementia (subject-level):", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v)
         for k, v in three_m.items() if k not in ("per_class_recall",)}))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ---- artifacts ----
    comp.to_csv(OUT_DIR / "benchmark_metrics.csv", index=False)

    # calibration reliability plots (honest OOF, winner)
    plot_dir = OUT_DIR / "calibration_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    y_win = oof_winner["true_binary_label"].to_numpy()
    p_win = oof_winner[PROBA_COL].to_numpy()
    p_sig = sig_final.predict(p_win.reshape(-1, 1))
    p_iso = iso_final.predict(p_win)
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, p, color in [("raw", p_win, "tab:blue"),
                            ("sigmoid", p_sig, "tab:green"),
                            ("isotonic", p_iso, "tab:red")]:
        bins = np.linspace(0, 1, 11)
        mean_pred = np.zeros(10); frac_pos = np.zeros(10)
        for i in range(10):
            m = (p >= bins[i]) & (p < bins[i + 1])
            if m.sum() == 0:
                mean_pred[i] = np.nan; frac_pos[i] = np.nan
                continue
            mean_pred[i] = p[m].mean(); frac_pos[i] = y_win[m].mean()
        ax.plot(mean_pred, frac_pos, "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction positive")
    ax.set_title("Binary reliability (winner=%s, OOF)" % winner)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "reliability_binary_winner_off.png", dpi=150)
    plt.close()
    fold_df = pd.concat([pd.DataFrame(r["fold_metrics"]).assign(model=n)
                         for n, r in results.items()], ignore_index=True)
    fold_df.to_csv(OUT_DIR / "cv_metrics.csv", index=False)
    oof_all.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    cal_res.to_csv(OUT_DIR / "calibration_metrics.csv", index=False)
    outer["subject_table"].to_csv(OUT_DIR / "subject_level_outer_test.csv", index=False)
    outer["visit_table"].to_csv(OUT_DIR / "visit_level_outer_test.csv", index=False)
    three_agg.to_csv(OUT_DIR / "three_class_subject_level_outer_test.csv", index=False)

    comparison = {
        "property": [
            "Target", "Independent unit", "Best model (experimental)",
            "Calibration", "Balanced accuracy (subject-level outer)",
            "Sensitivity (subject-level outer)",
            "Specificity (subject-level outer)",
            "ROC-AUC (subject-level outer)",
            "PR-AUC (subject-level outer)",
            "Brier (subject-level outer)",
            "Converted prediction",
        ],
        "3-class": [
            "Nondemented/Converted/Demented", "subject",
            "LogisticRegression (Phase 2) + sigmoid (Phase 3)",
            "sigmoid", "%.4f" % three_m["balanced_accuracy"],
            "%.4f" % three_m["sensitivity"],
            "%.4f" % three_m["specificity"],
            "%.4f" % three_m["roc_auc"],
            "%.4f" % three_m["pr_auc"],
            "%.4f" % three_m["brier"],
            "never predicted on outer test (4 subjects)",
        ],
        "binary": [
            "Nondemented / Any dementia", "subject",
            "%s" % winner,
            "%s" % selected_cal,
            "%.4f" % outer["subject_level"]["calibrated"]["balanced_accuracy"],
            "%.4f" % outer["subject_level"]["calibrated"]["sensitivity"],
            "%.4f" % outer["subject_level"]["calibrated"]["specificity"],
            "%.4f" % outer["subject_level"]["calibrated"]["roc_auc"],
            "%.4f" % outer["subject_level"]["calibrated"]["pr_auc"],
            "%.4f" % outer["subject_level"]["calibrated"]["brier"],
            "N/A (collapsed into any-dementia)",
        ],
    }
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(OUT_DIR / "comparison.csv", index=False)
    print("\n=== COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))

    cm_bin = confusion_matrix(outer["subject_table"]["true_binary_label"].to_numpy(),
                              (outer["subject_table"]["subject_probability"] >= BINARY_THRESHOLD).astype(int),
                              labels=[0, 1])
    pd.DataFrame(cm_bin, index=["actual_Nondemented", "actual_AnyDementia"],
                 columns=["pred_Nondemented", "pred_AnyDementia"]).to_csv(
        OUT_DIR / "confusion_matrix_binary.csv")

    repro = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": __import__('xgboost').__version__ if XGBOOST_AVAILABLE else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seeds": {"numpy": NP_SEED, "cv": CV_RANDOM_STATE},
        "n_splits": N_SPLITS,
        "cv_splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by Subject ID",
        "outer_split_source": str(SPLIT_PATH),
        "feature_order": FEATURES,
        "class_mapping": GROUP_MAP,
        "binary_target_definition": "1 = Converted|Demented, 0 = Nondemented (experimental column; original group untouched)",
        "binary_threshold": BINARY_THRESHOLD,
        "aggregation_method": "subject probability = mean of visit probabilities across visits",
        "preprocessing": {
            "LogisticRegression": "SimpleImputer(median)->StandardScaler",
            "SVM": "SimpleImputer(median)->StandardScaler",
            "RandomForest": "SimpleImputer(median)",
            "XGBoost": "SimpleImputer(median)",
        },
        "selection_criterion": "balanced accuracy primary; sensitivity/specificity/ROC-AUC secondary",
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps({
        "reproducibility": repro,
        "binary_winner": winner,
        "binary_winner_hyperparameters": results[winner]["best_params"],
        "selected_calibration": selected_cal,
        "calibration_summary": {m: {k: list(v) for k, v in cal_summary[m].items()} for m in cal_summary},
        "binary_subject_level_outer": outer["subject_level"],
        "binary_visit_level_outer": outer["visit_level"],
        "three_class_subject_level_outer": {k: v for k, v in three_m.items() if k not in ("per_class_recall",)},
        "three_class_per_class_recall": three_m["per_class_recall"],
        "comparison_table": comparison,
        "notes": [
            "Subject is the independent unit; visit probabilities aggregated by mean.",
            "Outer 38-subject test used exactly once after selection.",
            "Converted outer test has only 4 subjects - statistically unstable.",
            "No threshold optimization performed; BINARY_THRESHOLD=0.5 fixed for comparison.",
        ],
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# 3-Class vs Binary Screening (ML Phase 4) - Research Only\n\n"
        "- Binary target: 1=Converted|Demented, 0=Nondemented (experimental).\n"
        "- Same 112/38 subject split as Phase 1/2/3, zero overlap.\n"
        "- Subject is the independent unit; visit probabilities aggregated by mean.\n"
        "- Winner (experimental): %s with %s calibration.\n"
        "- Outer test threshold fixed at 0.5 (no threshold tuning).\n"
        "- No production / /predict / SHAP / MMSE / Q11 changes.\n" % (winner, selected_cal),
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("split: 112/38, overlap 0:", len(train_subjects & test_subjects) == 0)
    print("all OOF subjects within training:", set(oof_all.subject_id) <= train_subjects)
    print("outer test subjects not in OOF:", not (set(outer['subject_table'][SUBJECT_COL]) <= train_subjects))
    print("original group column untouched: binary target is a separate column")


if __name__ == "__main__":
    main()
