"""
ML PIPELINE PHASE 2 -- PATIENT-GROUPED MODEL BENCHMARK (RESEARCH ONLY).

Benchmarks RF / Logistic Regression / SVM / XGBoost under subject-grouped
stratified 5-fold CV restricted to the 112 training subjects, then evaluates
the CV-selected winner ONCE on the untouched 38-subject outer holdout.

Rules:
- Subject ID is the independent unit; no subject crosses any boundary.
- No calibration, no production changes, no /predict changes.
"""

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

import sklearn
import warnings

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "model_benchmark"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
LABEL_COL = "group"
SUBJECT_COL = "Subject ID"
CLASS_NAMES = ["Nondemented", "Converted", "Demented"]
LABELS = [0, 1, 2]

N_SPLITS = 5
CV_RANDOM_STATE = 42
NP_SEED = 42
CONVERTED_LABEL = 1


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "CDR": "cdr", "SES": "ses", "Group": LABEL_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    return df[[SUBJECT_COL] + FEATURES + [LABEL_COL]]


def load_split():
    d = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = set(d["train_subjects"])
    test_subjects = set(d["test_subjects"])
    overlap = train_subjects & test_subjects
    if len(train_subjects) != 112 or len(test_subjects) != 38:
        raise SystemExit("STOP: saved split counts differ from 112/38")
    if overlap:
        raise SystemExit("STOP: saved split has subject overlap")
    if d["overlap"]:
        raise SystemExit("STOP: saved split reports overlap")
    if d["train_rows"] != 281 or d["test_rows"] != 92:
        raise SystemExit("STOP: saved split row counts differ from 281/92")
    return train_subjects, test_subjects


def metric_row(y_true, y_pred, y_proba, classes):
    r = {}
    r["accuracy"] = float(accuracy_score(y_true, y_pred))
    r["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    r["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    r["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    r["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    r["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0))
    p, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    r["converted_precision"] = float(p[CONVERTED_LABEL])
    r["converted_recall"] = float(rec[CONVERTED_LABEL])
    r["converted_f1"] = float(f1[CONVERTED_LABEL])
    if y_proba is not None:
        try:
            r["log_loss"] = float(log_loss(y_true, y_proba))
        except ValueError:
            r["log_loss"] = None
    return r


def mean_std(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def build_candidates():
    base_rf = dict(n_estimators=200, random_state=42, n_jobs=1)
    cands = {}

    cands["RandomForest"] = {
        "estimator": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(class_weight={0: 1, 1: 4, 2: 2}, **base_rf)),
        ]),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10],
            "clf__min_samples_leaf": [1, 3],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": [{0: 1, 1: 4, 2: 2}, "balanced"],
        },
    }

    cands["LogisticRegression"] = {
        "estimator": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
        ]),
        "param_grid": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [{0: 1, 1: 4, 2: 2}, "balanced"],
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
            "clf__class_weight": [{0: 1, 1: 4, 2: 2}, "balanced"],
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
                    objective="multi:softprob", eval_metric="mlogloss",
                )),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.1, 0.3],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__min_child_weight": [1, 3],
            },
        }
    return cands


def run_cv_for_candidate(name, est, param_grid, X, y, groups):
    """Grid search inside subject-grouped CV, then best-param grouped CV for OOF + fold metrics."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    gs = GridSearchCV(
        est, param_grid, cv=cv, scoring="balanced_accuracy",
        n_jobs=1, refit=True, error_score="raise",
    )
    gs.fit(X, y, groups=groups)
    best_params = gs.best_params_
    cv_score_mean = float(gs.best_score_)

    # Manual grouped CV with best params for fold-level metrics + OOF.
    best_est = gs.best_estimator_
    oof_rows = []
    fold_metrics = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups)):
        tr_sid = set(groups[tr_idx])
        va_sid = set(groups[va_idx])
        if tr_sid & va_sid:
            raise SystemExit("STOP: subject overlap between folds in %s" % name)
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        m = clone(best_est)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        proba = m.predict_proba(X_va)
        classes = m.classes_
        fm = metric_row(y_va, pred, proba, classes)
        fm["fold"] = fold_idx
        fold_metrics.append(fm)
        for i in range(len(X_va)):
            oof_rows.append({
                "subject_id": groups[va_idx][i],
                "fold": fold_idx,
                "true_label": int(y_va.iloc[i]),
                "predicted_class": int(pred[i]),
                "prob_Nondemented": float(proba[i, list(classes).index(0)]),
                "prob_Converted": float(proba[i, list(classes).index(1)]),
                "prob_Demented": float(proba[i, list(classes).index(2)]),
            })

    return {
        "name": name,
        "best_params": best_params,
        "cv_balanced_accuracy_mean": cv_score_mean,
        "fold_metrics": fold_metrics,
        "oof_rows": oof_rows,
        "summary": {k: mean_std(fold_metrics, k) for k in [
            "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "log_loss",
            "converted_precision", "converted_recall", "converted_f1",
        ]},
    }


def main():
    np.random.seed(NP_SEED)
    df = load_data()
    train_subjects, test_subjects = load_split()

    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]
    if df_train[SUBJECT_COL].nunique() != 112 or df_test[SUBJECT_COL].nunique() != 38:
        raise SystemExit("STOP: subject counts mismatch after applying split")
    if set(df_train[SUBJECT_COL]) & set(df_test[SUBJECT_COL]):
        raise SystemExit("STOP: overlap detected after applying split")

    print("=== SECTION 1/2: SPLIT + CV SETUP ===")
    print(f"training: {len(df_train)} rows / {df_train[SUBJECT_COL].nunique()} subjects")
    print(f"outer test: {len(df_test)} rows / {df_test[SUBJECT_COL].nunique()} subjects")
    print(f"train class dist (subjects): "
          f"{df_train.groupby(SUBJECT_COL)[LABEL_COL].first().value_counts().sort_index().to_dict()}")
    print(f"outer test class dist (subjects): "
          f"{df_test.groupby(SUBJECT_COL)[LABEL_COL].first().value_counts().sort_index().to_dict()}")

    print("\n=== SECTION 3/4: MODEL CANDIDATES ===")
    print(f"XGBoost available: {XGBOOST_AVAILABLE} (version "
          + (__import__('xgboost').__version__ if XGBOOST_AVAILABLE else "n/a") + ")")

    X = df_train[FEATURES]
    y = df_train[LABEL_COL]
    groups = df_train[SUBJECT_COL].to_numpy()

    candidates = build_candidates()
    results = {}
    for name, c in candidates.items():
        print(f"\n=== TUNING + CV: {name} ===")
        res = run_cv_for_candidate(name, c["estimator"], c["param_grid"], X, y, groups)
        results[name] = res
        print(f"best params: {res['best_params']}")
        print(f"CV balanced accuracy: {res['cv_balanced_accuracy_mean']:.4f}")
        print(f"  macro F1 {res['summary']['macro_f1']['mean']:.4f} | "
              f"conv recall {res['summary']['converted_recall']['mean']:.4f} | "
              f"conv F1 {res['summary']['converted_f1']['mean']:.4f} | "
              f"log loss {res['summary']['log_loss']['mean']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- SECTION 8/10: comparison + selection ----
    rows = []
    for name, res in results.items():
        rows.append({
            "model": name,
            "cv_balanced_acc_mean": res["summary"]["balanced_accuracy"]["mean"],
            "cv_balanced_acc_std": res["summary"]["balanced_accuracy"]["std"],
            "cv_macro_f1_mean": res["summary"]["macro_f1"]["mean"],
            "cv_macro_recall_mean": res["summary"]["macro_recall"]["mean"],
            "cv_macro_precision_mean": res["summary"]["macro_precision"]["mean"],
            "cv_weighted_f1_mean": res["summary"]["weighted_f1"]["mean"],
            "cv_accuracy_mean": res["summary"]["accuracy"]["mean"],
            "cv_log_loss_mean": res["summary"]["log_loss"]["mean"],
            "cv_converted_recall_mean": res["summary"]["converted_recall"]["mean"],
            "cv_converted_f1_mean": res["summary"]["converted_f1"]["mean"],
        })
    comp = pd.DataFrame(rows).sort_values("cv_balanced_acc_mean", ascending=False)
    comp.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    print("\n=== SECTION 8/10: MODEL COMPARISON (mean over %d grouped folds) ===" % N_SPLITS)
    print(comp.to_string(index=False))

    # Selection hierarchy: 1) balanced accuracy, 2) macro F1, 3) converted recall/F1,
    # 4) log loss, 5) accuracy / weighted F1.
    ranked = comp.sort_values(
        ["cv_balanced_acc_mean", "cv_macro_f1_mean", "cv_converted_recall_mean",
         "cv_log_loss_mean", "cv_accuracy_mean"],
        ascending=[False, False, False, True, False],
    )
    winner = ranked.iloc[0]["model"]
    print(f"\n=== SECTION 10: SELECTED WINNER: {winner} ===")
    print("Reason: best defensible balanced accuracy under subject-grouped CV, "
          "with macro F1 / converted recall / log loss tie-breakers per hierarchy.")

    # ---- OOF ----
    oof_dfs = []
    for name, res in results.items():
        od = pd.DataFrame(res["oof_rows"])
        od["model"] = name
        oof_dfs.append(od)
    oof_all = pd.concat(oof_dfs, ignore_index=True)
    oof_all.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    fold_rows = []
    for name, res in results.items():
        for fm in res["fold_metrics"]:
            fr = {"model": name}
            fr.update(fm)
            fold_rows.append(fr)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_metrics.csv", index=False)

    # ---- SECTION 11: OUTER TEST EVALUATION (once) ----
    winner_res = results[winner]
    winner_est = clone(candidates[winner]["estimator"])
    winner_est.set_params(**winner_res["best_params"])
    X_tr = df_train[FEATURES]
    y_tr = df_train[LABEL_COL]
    winner_est.fit(X_tr, y_tr)  # preprocessing learned from 112 subjects only
    X_te = df_test[FEATURES]
    y_te = df_test[LABEL_COL]
    y_pred = winner_est.predict(X_te)
    y_proba = winner_est.predict_proba(X_te)
    classes = winner_est.classes_

    outer = {
        "model": winner,
        "best_params": winner_res["best_params"],
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "macro_precision": float(precision_score(y_te, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "macro_recall": float(recall_score(y_te, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "weighted_f1": float(f1_score(y_te, y_pred, average="weighted", labels=LABELS, zero_division=0)),
        "log_loss": float(log_loss(y_te, y_proba)),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=LABELS).tolist(),
        "test_subjects": int(df_test[SUBJECT_COL].nunique()),
        "test_rows": int(len(df_test)),
        "converted_test_subjects": int(df_test[df_test[LABEL_COL] == CONVERTED_LABEL][SUBJECT_COL].nunique()),
    }
    p, rec, f1, sup = precision_recall_fscore_support(y_te, y_pred, labels=LABELS, zero_division=0)
    outer["per_class"] = {
        CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(rec[i]),
                         "f1": float(f1[i]), "support_rows": int(sup[i]),
                         "support_subjects": int(df_test[df_test[LABEL_COL] == i][SUBJECT_COL].nunique())}
        for i in LABELS
    }

    roc = None
    if outer["converted_test_subjects"] >= 8:
        try:
            roc = float(roc_auc_score(y_te, y_proba, multi_class="ovr", labels=LABELS, average="macro"))
        except ValueError:
            roc = None
    outer["roc_auc_macro_ovr"] = roc

    print("\n=== SECTION 11: OUTER TEST EVALUATION (winner, once) ===")
    print(f"test subjects={outer['test_subjects']} test rows={outer['test_rows']} "
          f"converted test subjects={outer['converted_test_subjects']}")
    print(f"accuracy={outer['accuracy']:.4f} balanced_accuracy={outer['balanced_accuracy']:.4f}")
    print(f"macro_precision={outer['macro_precision']:.4f} macro_recall={outer['macro_recall']:.4f} "
          f"macro_f1={outer['macro_f1']:.4f} weighted_f1={outer['weighted_f1']:.4f} log_loss={outer['log_loss']:.4f}")
    print("per-class:", json.dumps(outer["per_class"]))
    print("confusion matrix (rows=true, cols=pred):")
    print(np.array(outer["confusion_matrix"]))
    print("Converted-class metrics are statistically unstable (n=4 subjects).")

    # ---- SECTION 17: save experimental winner ----
    joblib.dump(winner_est, OUT_DIR / ("%s_winner.pkl" % winner))
    print("\nSaved experimental winner:", OUT_DIR / ("%s_winner.pkl" % winner))

    # ---- reproducibility ----
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
        "sex_mapping": SEX_MAP,
        "preprocessing": {
            "RandomForest": "SimpleImputer(strategy='median') only",
            "LogisticRegression": "SimpleImputer(strategy='median') -> StandardScaler",
            "SVM": "SimpleImputer(strategy='median') -> StandardScaler",
            "XGBoost": "SimpleImputer(strategy='median') only",
        },
        "selection_criterion": "balanced accuracy primary; macro F1 -> converted recall/F1 -> log loss -> accuracy/weighted F1",
    }

    (OUT_DIR / "cv_metrics.json").write_text(json.dumps({
        "reproducibility": repro,
        "comparison": comp.to_dict(orient="records"),
        "winner": winner,
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "best_model_selection.json").write_text(json.dumps({
        "winner": winner,
        "reason": "Best defensible balanced accuracy under subject-grouped CV.",
        "cv_balanced_accuracy": comp.set_index("model").loc[winner, "cv_balanced_acc_mean"],
        "comparison": comp.to_dict(orient="records"),
        "hierarchy": "1) balanced accuracy 2) macro F1 / macro recall 3) converted recall/F1 4) log loss 5) accuracy / weighted F1",
        "note": "outer 38-subject test set untouched during selection",
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "outer_test_evaluation.json").write_text(json.dumps(outer, indent=2), encoding="utf-8")
    (OUT_DIR / "benchmark_readme.md").write_text(
        "# Model Benchmark (ML Phase 2) - Research Only\n\n"
        "- Unit: Subject ID; subject-grouped StratifiedGroupKFold (5-fold) on the 112 training subjects.\n"
        "- Outer 38-subject holdout used exactly once for the winner.\n"
        "- No calibration applied. OOF predictions saved for the calibration phase.\n"
        "- Winner: %s (experimental subject-grouped benchmark winner; NOT final).\n"
        "- Production artifacts untouched.\n" % winner,
        encoding="utf-8",
    )

    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SECTION 12: COMPARISON ===")
    phase1 = {"accuracy": 0.5978, "balanced_accuracy": 0.4795, "macro_f1": 0.4750, "weighted_f1": 0.5961}
    print("Phase 1 RF baseline (leakage-free): accuracy=%.4f balanced=%.4f macroF1=%.4f weightedF1=%.4f"
          % (phase1["accuracy"], phase1["balanced_accuracy"], phase1["macro_f1"], phase1["weighted_f1"]))
    print("Winner outer test: accuracy=%.4f balanced=%.4f macroF1=%.4f weightedF1=%.4f"
          % (outer["accuracy"], outer["balanced_accuracy"], outer["macro_f1"], outer["weighted_f1"]))
    print("Historical row-leaked ~0.76 is NOT a valid target; shown only as reference.")


if __name__ == "__main__":
    main()
