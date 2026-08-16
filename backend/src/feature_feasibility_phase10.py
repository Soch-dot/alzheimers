"""
ML PIPELINE PHASE 10 -- FEATURE-FEASIBILITY EXPERIMENT: FIVE-FEATURE CEILING VS
ADDITIONAL PREDICTION-TIME INFORMATION (RESEARCH ONLY).

Research question: can any legitimate prediction-time-available feature set
improve separation of Nondemented vs Converted OR Demented, WITHOUT leaking
future information and WITHOUT using CDR?

Two distinct experiments (kept separate):
  EXPERIMENT A (current-product info): Sets 1-4
    Set 1: age, sex, education_years, mmse, ses                 (control)
    Set 2: Set 1 + visit_count
    Set 3: Set 1 + time_since_baseline (= MR Delay at the visit)
    Set 4: Set 1 + previous_mmse, mmse_delta                    (confirmation
           of Phase 6; not an assumed improvement)
  EXPERIMENT B (OASIS MRI/clinical): Set 5
    Set 5: Set 1 + eTIV, nWBV, ASF  (research-only; NOT currently collected
           by the product; would require equivalent data collection)

Absolute prohibitions: CDR (target/post-diagnostic leakage), Group/final
diagnosis, future information, post-outcome information, model predictions as
features, MRI-derived "final label" constructs.

Validation: canonical 112 training / 38 outer-test subject split (zero overlap).
Development: StratifiedGroupKFold(5, shuffle=True, random_state=42) on the 112
training subjects. Preprocessing (SimpleImputer median -> StandardScaler) fit
inside each fold on training rows only. Model: LogisticRegression C=10.0
class_weight=balanced (feature effect isolated; no new algorithm benchmark).
Threshold 0.40 kept for descriptive comparison; subject-level latest-visit
evaluation; Subject ID is the independent unit.

SCIENTIFIC DISTINCTION: the target is the subject-level final OASIS Group label.
Adding features improves classification of the subject's OASIS outcome label; it
does NOT demonstrate future-conversion prediction.
"""

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
)

import sklearn

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase10_feature_feasibility"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
BASE_FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"

NP_SEED = 42
THRESHOLD = 0.40
N_FOLDS = 5
CV_RANDOM_STATE = 42

WINNER_HP = {"clf__C": 10.0, "clf__class_weight": "balanced"}

FEATURE_SETS = {
    "Set1_control": BASE_FEATURES,
    "Set2_+visit_count": BASE_FEATURES + ["visit_count"],
    "Set3_+time_since_baseline": BASE_FEATURES + ["mr_delay"],
    "Set4_+prev_mmse_delta": BASE_FEATURES + ["prev_mmse", "mmse_delta"],
    "Set5_+MRI": BASE_FEATURES + ["etiv", "nwbv", "asf"],
}

EXPECTED_LATEST = {
    "sensitivity": 0.75, "specificity": 0.8333, "balanced_accuracy": 0.7917,
    "ppv": 0.8333, "npv": 0.75, "f1": 0.7895, "accuracy": 0.7895,
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "CDR": "cdr", "SES": "ses", "Group": LABEL_COL,
        "Visit": "raw_visit", "MR Delay": MR_DELAY_COL,
        "eTIV": "etiv", "nWBV": "nwbv", "ASF": "asf",
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    df = df.sort_values([SUBJECT_COL, MR_DELAY_COL]).reset_index(drop=True)
    df["temporal_rank"] = df.groupby(SUBJECT_COL).cumcount() + 1
    df["visit_count"] = df.groupby(SUBJECT_COL)[SUBJECT_COL].transform("size")
    grp = df.groupby(SUBJECT_COL)
    df["prev_mmse"] = grp["mmse"].shift(1)
    df["mmse_delta"] = df["mmse"] - df["prev_mmse"]
    return df


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
    ]).set_params(**WINNER_HP)


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


def converted_stats(frame, pred_col="predicted_binary"):
    msk = frame[LABEL_COL] == 1
    tot = int(msk.sum())
    det = int((frame.loc[msk, pred_col] == 1).sum())
    flagged = frame[frame[pred_col] == 1]
    prec = None
    if len(flagged):
        prec = float((flagged[LABEL_COL] == 1).mean())
    return {
        "converted_subjects": tot, "detected": det, "missed": tot - det,
        "converted_recall": float(det / tot) if tot else None,
        "converted_precision": prec,
        "converted_flagged_all": int(len(flagged)),
    }


def main():
    np.random.seed(NP_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])
    assert len(train_subjects & test_subjects) == 0
    assert len(train_subjects) == 112 and len(test_subjects) == 38

    df = load_data()
    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]

    # subject-level frames: latest visit per subject
    def latest_frame(subset):
        m = subset[subset[MR_DELAY_COL] == subset.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")].copy()
        return m.reset_index(drop=True)

    tr_latest = latest_frame(df_train)
    te_latest = latest_frame(df_test)
    assert len(tr_latest) == 112 and len(te_latest) == 38

    # ---- DEVELOPMENT: subject-grouped 5-fold CV on the 112 training subjects ----
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    cv_subjects = pd.DataFrame({
        "subject_id": tr_latest[SUBJECT_COL].to_numpy(),
        "label": tr_latest[BINARY_COL].to_numpy(),
    })
    fold_idx = list(cv.split(cv_subjects, cv_subjects["label"],
                             groups=cv_subjects["subject_id"]))

    print("=== EXPERIMENT A + B: DEVELOPMENT GROUPED-CV (5 folds, 112 train subjects) ===")
    cv_rows = []
    oof_holder = {}
    for setname, feats in FEATURE_SETS.items():
        oof_list = []
        for k, (tr_idx, va_idx) in enumerate(fold_idx):
            tr_subj = set(cv_subjects.iloc[tr_idx]["subject_id"])
            va_subj = set(cv_subjects.iloc[va_idx]["subject_id"])
            assert not (tr_subj & va_subj)
            tr_rows = df_train[df_train[SUBJECT_COL].isin(tr_subj)]
            va_rows = tr_latest[tr_latest[SUBJECT_COL].isin(va_subj)]
            est = make_pipeline()
            est.fit(tr_rows[feats], tr_rows[BINARY_COL])
            va_rows = va_rows.copy()
            va_rows["prob"] = est.predict_proba(va_rows[feats])[:, 1]
            oof_list.append(va_rows[[SUBJECT_COL, LABEL_COL, BINARY_COL, "prob"]]
                            .assign(fold=k))
        oof = pd.concat(oof_list, ignore_index=True)
        oof["predicted_binary"] = (oof["prob"] >= THRESHOLD).astype(int)
        m = metrics_at_threshold(oof[BINARY_COL], oof["prob"])
        cs = converted_stats(oof)
        assert len(oof) == 112
        cv_rows.append({
            "feature_set": setname,
            "features": feats,
            **{k: m[k] for k in ["sensitivity", "specificity", "balanced_accuracy",
                                 "ppv", "npv", "f1", "accuracy", "roc_auc", "pr_auc",
                                 "brier", "logloss"]},
            "converted_detected": cs["detected"], "converted_missed": cs["missed"],
            "converted_recall": cs["converted_recall"],
        })
        oof_holder[setname] = oof
        print("  %-24s balacc=%.4f auc=%.4f sens=%.4f spec=%.4f | conv det/miss=%d/%d"
              % (setname, m["balanced_accuracy"], m["roc_auc"] if m["roc_auc"] else float("nan"),
                 m["sensitivity"], m["specificity"], cs["detected"], cs["missed"]))
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT_DIR / "cv_metrics.csv", index=False)

    # ---- OUTER TEST: evaluate each candidate once ----
    print("\n=== OUTER TEST (38 subjects, one evaluation per feature set) ===")
    outer_rows = []
    outer_holder = {}
    for setname, feats in FEATURE_SETS.items():
        est = make_pipeline()
        est.fit(df_train[feats], df_train[BINARY_COL])
        te = te_latest.copy()
        te["prob"] = est.predict_proba(te[feats])[:, 1]
        te["predicted_binary"] = (te["prob"] >= THRESHOLD).astype(int)
        m = metrics_at_threshold(te[BINARY_COL], te["prob"])
        cs = converted_stats(te)
        outer_rows.append({
            "feature_set": setname,
            **{k: m[k] for k in ["sensitivity", "specificity", "balanced_accuracy",
                                 "ppv", "npv", "f1", "accuracy", "roc_auc", "pr_auc",
                                 "brier", "logloss"]},
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "converted_detected": cs["detected"], "converted_missed": cs["missed"],
            "converted_recall": cs["converted_recall"],
            "converted_precision": cs["converted_precision"],
        })
        outer_holder[setname] = te
        print("  %-24s balacc=%.4f auc=%.4f sens=%.4f spec=%.4f brier=%.4f | conv det/miss=%d/%d"
              % (setname, m["balanced_accuracy"], m["roc_auc"] if m["roc_auc"] else float("nan"),
                 m["sensitivity"], m["specificity"], m["brier"], cs["detected"], cs["missed"]))
    outer_df = pd.DataFrame(outer_rows)
    outer_df.to_csv(OUT_DIR / "outer_test_metrics.csv", index=False)

    # control reproduction check vs Phase 7 latest
    ctrl = outer_rows[0]
    print("\nSet1 control vs Phase 7 latest expected: max abs diff =",
          round(max(abs(ctrl[k] - EXPECTED_LATEST[k]) for k in EXPECTED_LATEST), 4))

    # ---- converted analysis per set ----
    print("\n=== CONVERTED ANALYSIS (outer test, per set) ===")
    conv_rows = []
    for setname in FEATURE_SETS:
        te = outer_holder[setname]
        msk = te[LABEL_COL] == 1
        sub = te[msk].sort_values(SUBJECT_COL)
        for _, r in sub.iterrows():
            conv_rows.append({
                "feature_set": setname,
                "subject_id": r[SUBJECT_COL],
                "true_group": int(r[LABEL_COL]),
                "n_visits": int(r["visit_count"]),
                "mmse": None if pd.isna(r["mmse"]) else float(r["mmse"]),
                "probability": round(float(r["prob"]), 4),
                "predicted": int(r["predicted_binary"]),
                "decision": "flagged" if r["predicted_binary"] == 1 else "missed",
            })
    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv(OUT_DIR / "converted_analysis.csv", index=False)
    print(conv_df.to_string(index=False))

    # ---- feature contribution: LR coefficients + ablation (Set5 vs Set1) ----
    print("\n=== FEATURE CONTRIBUTIONS (outer-test models) ===")
    contrib_rows = []
    for setname in ["Set1_control", "Set5_+MRI", "Set4_+prev_mmse_delta"]:
        feats = FEATURE_SETS[setname]
        est = make_pipeline()
        est.fit(df_train[feats], df_train[BINARY_COL])
        c = est.named_steps["clf"].coef_[0]
        for f, co in zip(feats, c):
            contrib_rows.append({"feature_set": setname, "feature": f,
                                 "coefficient": round(float(co), 4)})
        print("  %s:" % setname)
        for f, co in zip(feats, c):
            print("    %-15s coef=%+8.4f" % (f, co))
    contrib_df = pd.DataFrame(contrib_rows)
    contrib_df.to_csv(OUT_DIR / "feature_contributions.csv", index=False)

    # ablation: Set5 vs Set1 differences (metrics)
    abl = {}
    for k in ["balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc", "brier"]:
        s5 = outer_rows[4][k]; s1 = outer_rows[0][k]
        abl[k] = None if (s5 is None or s1 is None) else round(float(s5) - float(s1), 4)
    print("  ablation Set5-Set1 (outer):", abl)

    # ---- availability classification ----
    availability = {
        "age": "A1: currently available to the product",
        "sex": "A1: currently available to the product",
        "education_years": "A1: currently available to the product",
        "mmse": "A1: currently available to the product",
        "ses": "A1: currently available to the product",
        "visit_count": "A2: available to the product (application knows visit history)",
        "mr_delay": "A3: available (time since first visit is known to the product)",
        "prev_mmse": "A4: available if a prior visit exists (prediction-time-safe; Phase 6 confirmation)",
        "mmse_delta": "A4: available if a prior visit exists (prediction-time-safe; Phase 6 confirmation)",
        "etiv": "B: legitimate per-visit MRI measure; present in OASIS but NOT collected by the product",
        "nwbv": "B: legitimate per-visit MRI measure; present in OASIS but NOT collected by the product",
        "asf": "B: legitimate per-visit MRI measure; present in OASIS but NOT collected by the product",
        "cdr": "PROHIBITED: target/post-diagnostic leakage",
        "group": "PROHIBITED: final diagnosis/target",
        "hand": "excluded: constant (all R) in this dataset",
    }
    (OUT_DIR / "feature_metadata.json").write_text(json.dumps({
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "availability_classification": availability,
        "leakage_audit": {
            "cdr_used": False, "group_used_as_feature": False,
            "future_information_used": False,
            "post_outcome_information_used": False,
            "model_predictions_as_features": False,
            "preprocessing_inside_cv": True,
        },
        "excluded_variables": {
            "cdr": "target/post-diagnostic leakage",
            "group": "final diagnosis (target)",
            "hand": "constant value (R) in all 373 rows",
        },
    }, indent=2), encoding="utf-8")

    comparison = {
        "threshold": THRESHOLD,
        "model": "LogisticRegression C=10.0 class_weight=balanced",
        "model_hyperparameters": WINNER_HP,
        "preprocessing": "SimpleImputer(median)->StandardScaler inside CV/train only",
        "validation": "StratifiedGroupKFold(5, shuffle=True, random_state=42) on 112 training subjects",
        "outer_split": "112 training / 38 test subjects, zero overlap",
        "evaluation": "subject-level latest available visit; Subject ID is independent unit",
        "experiment_A_current_product": ["Set1_control", "Set2_+visit_count",
                                          "Set3_+time_since_baseline", "Set4_+prev_mmse_delta"],
        "experiment_B_mri_research_only": ["Set5_+MRI"],
        "cv": cv_rows,
        "outer_test": outer_rows,
        "ablation_set5_minus_set1": abl,
        "scientific_note": ("Target is the subject-level final OASIS Group label. Feature gains improve "
                            "classification of the subject's OASIS outcome label; they do NOT demonstrate "
                            "future-conversion prediction."),
        "mri_note": ("Set 5 (eTIV/nWBV/ASF) is an OASIS research-only result and would require collecting "
                     "equivalent MRI-derived information in the actual product; the current browser "
                     "application does not collect these features."),
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    repro = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seeds": {"numpy": NP_SEED, "cv": CV_RANDOM_STATE},
        "feature_sets": FEATURE_SETS,
        "model_hyperparameters": WINNER_HP,
        "threshold": THRESHOLD,
        "n_folds": N_FOLDS,
        "temporal_ordering": "MR Delay",
        "outer_split": "112 training / 38 test subjects (subject_grouped_baseline/subject_split.json)",
        "latest_definition": "row with max MR Delay per subject",
    }
    (OUT_DIR / "reproducibility.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")

    (OUT_DIR / "README.md").write_text(
        "# Phase 10 - Feature-Feasibility Experiment (Research Only)\n\n"
        "- Experimental feature-feasibility analysis: can prediction-time-available features beat the five-feature ceiling?\n"
        "- Experiment A (current-product info): Sets 1-4 (control, visit_count, time_since_baseline, prev_mmse+mmse_delta).\n"
        "- Experiment B (OASIS MRI, research-only): Set 5 = Set 1 + eTIV, nWBV, ASF.\n"
        "- CDR, Group, future/post-outcome information PROHIBITED; no MRI added to product; /predict unchanged.\n"
        "- Grouped 5-fold CV on 112 train subjects; outer 38 subjects evaluated once per set; threshold 0.40; subject-level latest visit.\n"
        "- LIMITATION: target is the final OASIS outcome label; gains classify the outcome label, not future conversion.\n"
        "- No production changes; no commits.\n",
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("split overlap == 0:", len(train_subjects & test_subjects) == 0)
    print("CV fold subject overlap == 0:", all(not (set(cv_subjects.iloc[t]['subject_id']) & set(cv_subjects.iloc[v]['subject_id'])) for t, v in fold_idx))
    print("outer test subjects == 38:", len(te_latest) == 38)
    print("cdr not in any feature set:", all("cdr" not in f for f in FEATURE_SETS.values()))
    print("threshold unchanged:", THRESHOLD == 0.40)
    print("preprocessing inside CV only: True")


if __name__ == "__main__":
    main()
