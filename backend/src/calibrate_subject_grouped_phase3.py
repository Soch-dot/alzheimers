"""
ML PIPELINE PHASE 3 -- PROBABILITY CALIBRATION (RESEARCH ONLY).

Calibrates the Phase 2 experimental winner (LogisticRegression) using the saved
subject-grouped OOF predictions, comparing raw / sigmoid / isotonic under an
HONEST subject-grouped second-level CV (calibrator never evaluated on its own
fit subjects), then applies the selected method once to the untouched outer test.

Rules:
- Calibration selection uses OOF (112 training subjects) ONLY.
- Outer 38-subject test is touched exactly once for the final evaluation.
- No production changes, no /predict changes, no SHAP changes.
"""

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd

from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)

import joblib
import sklearn

OOF_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "model_benchmark" / "oof_predictions.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
WINNER_PKL = Path(__file__).resolve().parents[1] / "models" / "experiments" / "model_benchmark" / "LogisticRegression_winner.pkl"
DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "calibration_phase3"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
LABEL_COL = "group"
SUBJECT_COL = "Subject ID"
CLASS_NAMES = ["Nondemented", "Converted", "Demented"]
LABELS = [0, 1, 2]
PROB_COLS = ["prob_Nondemented", "prob_Converted", "prob_Demented"]

N_SPLITS = 5
CV_RANDOM_STATE = 42
NP_SEED = 42
WINNER = "LogisticRegression"
METHODS = ["raw", "sigmoid", "isotonic"]


def ece_binary(y_true, p, bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def multiclass_brier(y_true, proba):
    y_onehot = np.zeros((len(y_true), 3))
    for i, c in enumerate(LABELS):
        y_onehot[:, i] = (y_true == c).astype(float)
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def multiclass_ece(y_true, proba, bins=10):
    vals = []
    for i, c in enumerate(LABELS):
        vals.append(ece_binary((y_true == c).astype(int), proba[:, i], bins))
    return float(np.mean(vals))


def metric_block(y_true, proba):
    y_pred = np.argmax(proba, axis=1)
    return {
        "log_loss": float(log_loss(y_true, proba, labels=LABELS)),
        "brier_mean_1vr": multiclass_brier(y_true, proba),
        "ece_mean_1vr": multiclass_ece(y_true, proba),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)),
    }


def fit_calibrators(proba_tr, y_tr):
    """Fit per-class calibrators (class c vs rest) on OOF training rows."""
    sig = {}
    iso = {}
    for c in LABELS:
        yc = (np.asarray(y_tr) == c).astype(int)
        pc = np.asarray(proba_tr)[:, LABELS.index(c)]
        s = _SigmoidCalibration()
        s.fit(pc.reshape(-1, 1), yc)
        sig[c] = s
        i = IsotonicRegression(out_of_bounds="clip")
        i.fit(pc, yc)
        iso[c] = i
    return sig, iso


def apply_calibrators(sig, iso, proba_va, method):
    proba_va = np.asarray(proba_va)
    out = np.zeros_like(proba_va)
    for c in LABELS:
        col = proba_va[:, LABELS.index(c)].reshape(-1, 1)
        if method == "sigmoid":
            out[:, LABELS.index(c)] = sig[c].predict(col)
        else:  # isotonic
            out[:, LABELS.index(c)] = iso[c].predict(col.ravel())
    # normalize rows to sum to 1
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def honest_calibration_cv(oof_df):
    """Second-level subject-grouped CV to compare raw/sigmoid/isotonic on OOF."""
    X = oof_df.reset_index(drop=True)
    y = X["true_label"].to_numpy()
    groups = X["subject_id"].to_numpy()
    proba = X[PROB_COLS].to_numpy()

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    records = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups)):
        tr_sid = set(groups[tr_idx])
        va_sid = set(groups[va_idx])
        if tr_sid & va_sid:
            raise SystemExit("STOP: subject overlap between calibration folds")
        sig, iso = fit_calibrators(proba[tr_idx], y[tr_idx])
        for method in METHODS:
            if method == "raw":
                p_va = proba[va_idx]
            else:
                p_va = apply_calibrators(sig, iso, proba[va_idx], method)
            m = metric_block(y[va_idx], p_va)
            m["method"] = method
            m["fold"] = fold
            m["n_rows"] = int(len(va_idx))
            m["n_subjects"] = int(len(va_sid))
            records.append(m)
    return pd.DataFrame(records)


def main():
    np.random.seed(NP_SEED)

    oof_all = pd.read_csv(OOF_PATH)
    oof = oof_all[oof_all["model"] == WINNER].copy()
    if len(oof) != 281 or oof["subject_id"].nunique() != 112:
        raise SystemExit("STOP: unexpected OOF shape for winner")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_subjects = set(split["test_subjects"])
    if len(set(oof["subject_id"]) & test_subjects) != 0:
        raise SystemExit("STOP: OOF rows contain outer-test subjects")

    print("=== SECTION 1: HONEST OOF CALIBRATION COMPARISON (%s) ===" % WINNER)
    res = honest_calibration_cv(oof)

    summary = {}
    for method in METHODS:
        sub = res[res["method"] == method]
        summary[method] = {k: {
            "mean": float(sub[k].mean()), "std": float(sub[k].std())
        } for k in ["log_loss", "brier_mean_1vr", "ece_mean_1vr",
                    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}
        print(f"\n{method}:")
        for k, v in summary[method].items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Selection: primary log loss, tie-breakers Brier then ECE (OOF, honest).
    ranking = sorted(
        summary,
        key=lambda m: (summary[m]["log_loss"]["mean"],
                       summary[m]["brier_mean_1vr"]["mean"],
                       summary[m]["ece_mean_1vr"]["mean"]),
    )
    selected = ranking[0]
    print("\n=== SECTION 2: SELECTED CALIBRATION METHOD ===")
    print(f"Ranking by OOF log loss/Brier/ECE: {ranking}")
    print(f"Selected: {selected}")

    # Fit final calibrator on ALL OOF rows of winner (112 training subjects).
    y_all = oof["true_label"].to_numpy()
    proba_all = oof[PROB_COLS].to_numpy()
    sig_final, iso_final = fit_calibrators(proba_all, y_all)
    final_cal = {"selected": selected, "sigmoid": sig_final, "isotonic": iso_final}

    # ---- Outer test: load winner model, apply calibrator, evaluate once ----
    winner_est = joblib.load(WINNER_PKL)
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "CDR": "cdr", "SES": "ses", "Group": LABEL_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]
    X_te = df_test[FEATURES]
    y_te = df_test[LABEL_COL].to_numpy()

    raw_te = winner_est.predict_proba(X_te)
    if selected == "raw":
        cal_te = raw_te
    else:
        cal_te = apply_calibrators(sig_final, iso_final, raw_te, selected)

    eval_raw = metric_block(y_te, raw_te)
    eval_cal = metric_block(y_te, cal_te)
    eval_raw.update({"n_subjects": int(df_test[SUBJECT_COL].nunique()),
                     "n_rows": int(len(df_test))})
    eval_cal.update({"n_subjects": int(df_test[SUBJECT_COL].nunique()),
                     "n_rows": int(len(df_test))})

    cm_raw = confusion_matrix(y_te, np.argmax(raw_te, axis=1), labels=LABELS).tolist()
    cm_cal = confusion_matrix(y_te, np.argmax(cal_te, axis=1), labels=LABELS).tolist()

    print("\n=== SECTION 3: OUTER TEST (ONE-SHOT, method=%s) ===" % selected)
    print("test subjects=%d test rows=%d converted test subjects=%d"
          % (eval_cal["n_subjects"], eval_cal["n_rows"],
             int(df_test[df_test[LABEL_COL] == 1][SUBJECT_COL].nunique())))
    print("raw:       log_loss=%.4f brier=%.4f ece=%.4f acc=%.4f balacc=%.4f macroF1=%.4f wF1=%.4f"
          % (eval_raw["log_loss"], eval_raw["brier_mean_1vr"], eval_raw["ece_mean_1vr"],
             eval_raw["accuracy"], eval_raw["balanced_accuracy"], eval_raw["macro_f1"], eval_raw["weighted_f1"]))
    print("calibrated log_loss=%.4f brier=%.4f ece=%.4f acc=%.4f balacc=%.4f macroF1=%.4f wF1=%.4f"
          % (eval_cal["log_loss"], eval_cal["brier_mean_1vr"], eval_cal["ece_mean_1vr"],
             eval_cal["accuracy"], eval_cal["balanced_accuracy"], eval_cal["macro_f1"], eval_cal["weighted_f1"]))
    print("raw CM:")
    print(np.array(cm_raw))
    print("calibrated CM:")
    print(np.array(cm_cal))
    print("\nRaw probability ranges on outer test:")
    for i, cn in enumerate(CLASS_NAMES):
        print("  %s: min=%.4f mean=%.4f max=%.4f" % (cn, raw_te[:, i].min(), raw_te[:, i].mean(), raw_te[:, i].max()))
    print("Calibrated probability ranges on outer test:")
    for i, cn in enumerate(CLASS_NAMES):
        print("  %s: min=%.4f mean=%.4f max=%.4f" % (cn, cal_te[:, i].min(), cal_te[:, i].mean(), cal_te[:, i].max()))

    # OOF calibrated predictions artifact
    oof_out = oof.copy()
    cal_oof = np.zeros_like(proba_all)
    if selected == "raw":
        cal_oof = proba_all
    else:
        cal_oof = apply_calibrators(sig_final, iso_final, proba_all, selected)
    oof_out["cal_Nondemented"] = cal_oof[:, 0]
    oof_out["cal_Converted"] = cal_oof[:, 1]
    oof_out["cal_Demented"] = cal_oof[:, 2]
    oof_out["calibration_method"] = selected

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "calibration_cv_folds.csv", index=False)
    oof_out.to_csv(OUT_DIR / "calibrated_oof_predictions.csv", index=False)

    comparison = {}
    for method in METHODS:
        comparison[method] = {
            "log_loss": summary[method]["log_loss"],
            "brier_mean_1vr": summary[method]["brier_mean_1vr"],
            "ece_mean_1vr": summary[method]["ece_mean_1vr"],
            "balanced_accuracy": summary[method]["balanced_accuracy"],
            "macro_f1": summary[method]["macro_f1"],
        }

    repro = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "random_seeds": {"numpy": NP_SEED, "cv": CV_RANDOM_STATE},
        "n_splits": N_SPLITS,
        "cv_splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by Subject ID",
        "model": WINNER,
        "oof_source": str(OOF_PATH),
        "methods": METHODS,
        "calibration_note": "per-class one-vs-rest _SigmoidCalibration (Platt) and IsotonicRegression(out_of_bounds='clip'); rows renormalized to sum 1",
        "selection_criterion": "OOF log loss primary, Brier then ECE tie-breakers; honest second-level subject-grouped CV",
    }

    (OUT_DIR / "calibration_comparison.json").write_text(json.dumps({
        "reproducibility": repro,
        "selected_method": selected,
        "comparison": comparison,
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "outer_test_calibrated_evaluation.json").write_text(json.dumps({
        "method": selected,
        "raw": eval_raw, "calibrated": eval_cal,
        "raw_confusion_matrix": cm_raw,
        "calibrated_confusion_matrix": cm_cal,
        "n_test_subjects": eval_cal["n_subjects"],
        "n_test_rows": eval_cal["n_rows"],
        "converted_test_subjects": int(df_test[df_test[LABEL_COL] == 1][SUBJECT_COL].nunique()),
        "note": "Converted outer-test metrics statistically unstable (n=4 subjects)",
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# Calibration Phase 3 - Research Only\n\n"
        "- Model: %s (experimental Phase 2 winner; NOT production)\n"
        "- OOF calibration comparison via honest subject-grouped 5-fold CV.\n"
        "- Selected method: %s\n"
        "- Outer 38-subject test used exactly once.\n"
        "- No production / /predict / SHAP changes.\n" % (WINNER, selected),
        encoding="utf-8",
    )

    # per-class outer test details
    per_class = {}
    for i, cn in enumerate(CLASS_NAMES):
        per_class[cn] = {
            "support_subjects": int(df_test[df_test[LABEL_COL] == i][SUBJECT_COL].nunique()),
            "support_rows": int((y_te == i).sum()),
        }
    (OUT_DIR / "outer_test_per_class.json").write_text(
        json.dumps(per_class, indent=2), encoding="utf-8")

    print("\n=== SECTION 4: SAVED ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("OOF subjects vs outer test overlap: 0 (verified before calibration)")
    print("Calibration CV folds subject-overlap: 0 (enforced in honest_calibration_cv)")
    print("Outer test touched exactly once: yes (final evaluation only)")
    print("Calibrator never fit on outer-test subjects: yes (fit on 112 training subjects' OOF only)")
    print("No calibration applied to /predict or production model.")


if __name__ == "__main__":
    main()
