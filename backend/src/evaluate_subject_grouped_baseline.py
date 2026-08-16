"""
Leakage-free subject-level baseline (ML PIPELINE PHASE 1) -- RESEARCH ONLY.

Establishes an honest patient-level generalization estimate for the current
production Random Forest configuration.  Does NOT touch production artifacts.

Key rule: Subject ID is the independent unit.  No subject may appear in both
train and test.  All visits of a subject stay in one partition.
"""

from pathlib import Path
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

import sklearn
import sys

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
SUBJECT_ID_COL = "Subject ID"
LABEL_COL = "group"
TEST_SIZE = 0.25
RANDOM_STATE = 42

RF_CONFIG = {
    "n_estimators": 200,
    "random_state": 42,
    "class_weight": {0: 1, 1: 4, 2: 2},
}


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.rename(columns={
        "Age": "age",
        "M/F": "sex",
        "EDUC": "education_years",
        "MMSE": "mmse",
        "CDR": "cdr",
        "SES": "ses",
        "Group": LABEL_COL,
    })

    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)

    if df["sex"].isna().any():
        raise ValueError("Unmapped sex values present")
    if df[LABEL_COL].isna().any():
        raise ValueError("Unmapped group values present")

    return df[[SUBJECT_ID_COL] + FEATURES + [LABEL_COL]]


def verify_subject_structure(df):
    total_rows = len(df)
    subjects = df[SUBJECT_ID_COL].unique()
    n_subjects = len(subjects)

    rows_per_subject = df.groupby(SUBJECT_ID_COL).size()
    label_span = df.groupby(SUBJECT_ID_COL)[LABEL_COL].nunique()

    class_rows = df.groupby(LABEL_COL).size()
    subject_labels = df.groupby(SUBJECT_ID_COL)[LABEL_COL].first()
    class_subjects = subject_labels.value_counts()

    print("\n=== SECTION 3: SUBJECT STRUCTURE VERIFICATION ===")
    print(f"total rows:                     {total_rows}")
    print(f"unique subjects:                {n_subjects}")
    print(f"rows per subject:               min={rows_per_subject.min()}, "
          f"max={rows_per_subject.max()}, mean={rows_per_subject.mean():.2f}")
    print(f"subjects with multiple visits:  {(rows_per_subject > 1).sum()}")
    print(f"class counts (rows):            {class_rows.to_dict()}")
    print(f"class counts (subjects):        {class_subjects.to_dict()}")
    print(f"subjects with >1 class label:   {(label_span > 1).sum()}")

    # Hard stop if the audit expectation is violated.
    if total_rows != 373:
        raise SystemExit("STOP: expected 373 rows, got %d" % total_rows)
    if n_subjects != 150:
        raise SystemExit("STOP: expected 150 subjects, got %d" % n_subjects)
    if rows_per_subject.min() < 2:
        raise SystemExit("STOP: expected all subjects to have >=2 visits")
    if (label_span > 1).sum() != 0:
        raise SystemExit("STOP: subject with multiple class labels found")
    if int(subject_labels.value_counts().get(1, 0)) != 14:
        raise SystemExit(
            "STOP: expected 14 Converted subjects, got %d"
            % int(subject_labels.value_counts().get(1, 0))
        )
    print("Structure matches audit expectation. Continuing.")

    return {
        "total_rows": int(total_rows),
        "unique_subjects": int(n_subjects),
        "rows_per_subject_min": int(rows_per_subject.min()),
        "rows_per_subject_max": int(rows_per_subject.max()),
        "rows_per_subject_mean": float(rows_per_subject.mean()),
        "subjects_with_multiple_visits": int((rows_per_subject > 1).sum()),
        "class_counts_rows": {int(k): int(v) for k, v in class_rows.items()},
        "class_counts_subjects": {int(k): int(v) for k, v in class_subjects.items()},
        "subjects_with_multiple_class_labels": int((label_span > 1).sum()),
    }


def subject_level_split(df):
    """
    Stratified split at the SUBJECT level.

    sklearn 1.7.2 does not expose StratifiedGroupShuffleSplit, so we build a
    subject-level table (one row per subject, label constant per subject) and
    stratify on the subject label.  Rows are then mapped back by Subject ID,
    guaranteeing zero subject overlap by construction.
    """
    subj_table = df.groupby(SUBJECT_ID_COL)[LABEL_COL].first().reset_index()
    subj_table.columns = [SUBJECT_ID_COL, "subject_label"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_subj_idx, te_subj_idx = next(sss.split(subj_table, subj_table["subject_label"]))

    train_subjects = set(subj_table.loc[tr_subj_idx, SUBJECT_ID_COL])
    test_subjects = set(subj_table.loc[te_subj_idx, SUBJECT_ID_COL])

    overlap = train_subjects & test_subjects
    if overlap:
        raise SystemExit(
            "STOP: subject overlap detected: %s" % sorted(overlap)
        )

    df_train = df[df[SUBJECT_ID_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_ID_COL].isin(test_subjects)]

    return df_train, df_test, train_subjects, test_subjects


def verify_split_integrity(df_train, df_test, train_subjects, test_subjects):
    print("\n=== SECTION 5: ZERO SUBJECT LEAKAGE VERIFICATION ===")
    intersection = train_subjects & test_subjects
    print(f"train subjects: {len(train_subjects)}")
    print(f"test subjects:  {len(test_subjects)}")
    print(f"intersection:   {len(intersection)}")

    if len(intersection) != 0:
        raise SystemExit("STOP: subject intersection != 0. Not training.")

    # Every test row's Subject ID must be absent from training rows.
    train_sid = set(df_train[SUBJECT_ID_COL])
    test_sid = set(df_test[SUBJECT_ID_COL])
    leaked_test_rows = df_test[df_test[SUBJECT_ID_COL].isin(train_sid)]
    print(f"test rows whose Subject ID appears in training: {len(leaked_test_rows)}")

    # All visits of a subject stay together (no subject in both partitions).
    train_check = df_train[SUBJECT_ID_COL].nunique() == len(train_sid)
    test_check = df_test[SUBJECT_ID_COL].nunique() == len(test_sid)
    print(f"all train visits grouped (subject appears once): {train_check}")
    print(f"all test visits grouped (subject appears once):  {test_check}")

    train_label = df_train.groupby(SUBJECT_ID_COL)[LABEL_COL].first()
    test_label = df_test.groupby(SUBJECT_ID_COL)[LABEL_COL].first()
    print(f"train subject class dist: {train_label.value_counts().sort_index().to_dict()}")
    print(f"test subject class dist:  {test_label.value_counts().sort_index().to_dict()}")

    return {
        "train_subjects": int(len(train_subjects)),
        "test_subjects": int(len(test_subjects)),
        "overlap_count": int(len(intersection)),
        "leaked_test_rows": int(len(leaked_test_rows)),
        "all_train_visits_grouped": bool(train_check),
        "all_test_visits_grouped": bool(test_check),
        "train_subject_class_dist": {int(k): int(v) for k, v in train_label.value_counts().sort_index().items()},
        "test_subject_class_dist": {int(k): int(v) for k, v in test_label.value_counts().sort_index().items()},
    }


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(**RF_CONFIG)),
    ])


def evaluate(df_train, df_test):
    X_train = df_train[FEATURES]
    y_train = df_train[LABEL_COL]
    X_test = df_test[FEATURES]
    y_test = df_test[LABEL_COL]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)  # preprocessing fit on TRAINING subjects only

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    labels = [0, 1, 2]
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_p = precision_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    macro_r = recall_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    support = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)

    per_class = {}
    for i, c in enumerate(labels):
        per_class[int(c)] = {
            "precision": float(support[0][i]),
            "recall": float(support[1][i]),
            "f1": float(support[2][i]),
            "support_rows": int(support[3][i]),
            "support_subjects": int((df_test[df_test[LABEL_COL] == c][SUBJECT_ID_COL]).nunique()),
        }

    n_test_converted_subjects = int(df_test[df_test[LABEL_COL] == 1][SUBJECT_ID_COL].nunique())
    n_test_converted_rows = int((df_test[LABEL_COL] == 1).sum())
    tiny_converted = n_test_converted_subjects < 8 or n_test_converted_rows < 10

    # ROC / PR AUC (OvR) -- only meaningful under enough positive subjects per class.
    roc_macro = None
    pr_macro = None
    pr_per_class = None
    if not tiny_converted:
        try:
            roc_macro = float(roc_auc_score(
                y_test, y_proba, multi_class="ovr", labels=labels, average="macro"
            ))
        except ValueError as e:
            print("  ROC-AUC not computed:", e)
        pr_per_class = {
            int(c): float(average_precision_score(
                (y_test == c).astype(int), y_proba[:, classes.tolist().index(c)]
            ))
            for c in labels
        }
        pr_macro = float(np.mean(list(pr_per_class.values())))

    ll = float(log_loss(y_test, y_proba))

    print("\n=== SECTION 8/9/10: RESULTS (OUTER TEST SUBJECTS ONLY) ===")
    print(f"train rows={len(X_train)} train subjects={df_train[SUBJECT_ID_COL].nunique()}")
    print(f"test  rows={len(X_test)}  test  subjects={df_test[SUBJECT_ID_COL].nunique()}")
    print()
    print(f"accuracy:            {acc:.4f}")
    print(f"balanced accuracy:   {bal_acc:.4f}")
    print(f"macro precision:     {macro_p:.4f}")
    print(f"macro recall:        {macro_r:.4f}")
    print(f"macro F1:            {macro_f1:.4f}")
    print(f"weighted F1:         {weighted_f1:.4f}")
    print()
    print("per-class:")
    for c in labels:
        d = per_class[c]
        print(f"  class {c}: precision={d['precision']:.4f} recall={d['recall']:.4f} "
              f"f1={d['f1']:.4f} rows={d['support_rows']} subjects={d['support_subjects']}")
    print()
    print("confusion matrix (rows=true, cols=pred, order [Nondemented, Converted, Demented]):")
    print(cm)
    if roc_macro is not None:
        print(f"ROC-AUC (macro, OvR): {roc_macro:.4f}")
    else:
        print("ROC-AUC: not reported (Converted test class too small to be statistically meaningful)")
    if pr_macro is not None:
        print(f"PR-AUC (macro, OvR): {pr_macro:.4f}")
    if pr_per_class is not None:
        print(f"PR-AUC per class (OvR): {pr_per_class}")
    print()
    print("uncalibrated Random Forest probabilities (raw predict_proba, no calibration applied):")
    for i, c in enumerate(labels):
        col = y_proba[:, classes.tolist().index(c)]
        print(f"  class {c}: min={col.min():.4f} mean={col.mean():.4f} max={col.max():.4f}")
    print(f"multiclass log loss (raw): {ll:.4f}")

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "roc_auc_macro_ovr": roc_macro,
        "pr_auc_macro_ovr": pr_macro,
        "pr_auc_per_class_ovr": pr_per_class,
        "log_loss_uncalibrated": ll,
        "probability_range": {
            int(c): {
                "min": float(y_proba[:, classes.tolist().index(c)].min()),
                "mean": float(y_proba[:, classes.tolist().index(c)].mean()),
                "max": float(y_proba[:, classes.tolist().index(c)].max()),
            }
            for c in labels
        },
    }
    return metrics


def main():
    df = load_data()
    subject_structure = verify_subject_structure(df)
    df_train, df_test, train_subjects, test_subjects = subject_level_split(df)
    split_info = verify_split_integrity(df_train, df_test, train_subjects, test_subjects)
    metrics = evaluate(df_train, df_test)

    n_train_subjects = df_train[SUBJECT_ID_COL].nunique()
    n_test_subjects = df_test[SUBJECT_ID_COL].nunique()

    print("\n=== SECTION 11: COMPARISON VS CURRENT REPORTED BASELINE ===")
    print("| Evaluation | Accuracy | Subjects | Rows |")
    print("|------------|----------|----------|------|")
    print("| Existing row-level split | ~0.76 (row-leaked, untrustworthy) | leaked (58 subjects straddle) | 75 test rows |")
    print(f"| New subject-level split  | {metrics['accuracy']:.4f} | {n_test_subjects} test subjects | {len(df_test)} test rows |")
    print()
    print("NOTE: new score is a 'leakage-free subject-level baseline', NOT 'final accuracy'.")
    print("Subject count in existing split was never reported (it was a row-level split).")

    reproducibility = {
        "random_seeds": {"split": RANDOM_STATE, "rf": RF_CONFIG["random_state"]},
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "dataset": {
            "path": DATA_PATH,
            "row_count": int(len(df)),
            "subject_count": int(df[SUBJECT_ID_COL].nunique()),
        },
        "feature_order": FEATURES,
        "class_mapping": GROUP_MAP,
        "sex_mapping": SEX_MAP,
        "split_method": "subject-level StratifiedShuffleSplit (manual subject table; StratifiedGroupShuffleSplit unavailable in sklearn 1.7.2)",
        "split_params": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "model": {
            "type": "sklearn Pipeline",
            "steps": ["SimpleImputer(strategy='median')", "RandomForestClassifier"],
            "config": RF_CONFIG,
        },
    }

    split_rows = {
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "train_subjects": int(n_train_subjects),
        "test_subjects": int(n_test_subjects),
        "subject_overlap": 0,
        "train_subject_class_dist": split_info["train_subject_class_dist"],
        "test_subject_class_dist": split_info["test_subject_class_dist"],
        "converted_train_subjects": split_info["train_subject_class_dist"].get(1, 0),
        "converted_test_subjects": split_info["test_subject_class_dist"].get(1, 0),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps({"reproducibility": reproducibility, "split": split_rows, "metrics": metrics},
                   indent=2), encoding="utf-8")
    cm_cols = ["Nondemented", "Converted", "Demented"]
    pd.DataFrame(metrics["confusion_matrix"], index=cm_cols, columns=cm_cols).to_csv(
        OUT_DIR / "confusion_matrix.csv"
    )
    (OUT_DIR / "subject_split.json").write_text(
        json.dumps({
            "train_subjects": sorted(train_subjects),
            "test_subjects": sorted(test_subjects),
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "overlap": [],
        }, indent=2), encoding="utf-8")
    (OUT_DIR / "feature_metadata.json").write_text(
        json.dumps({
            "features": FEATURES,
            "class_mapping": GROUP_MAP,
            "sex_mapping": SEX_MAP,
            "subject_id_column": SUBJECT_ID_COL,
            "label_column": LABEL_COL,
            "notes": "Subject ID is the independent unit and is NOT a model feature.",
        }, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        (
            "# Subject-Grouped Baseline (ML Phase 1) - Research Only\n\n"
            "Leakage-free subject-level evaluation of the production Random Forest "
            "configuration.  Do NOT confuse this score with final accuracy.\n\n"
            "## Split\n"
            "- Unit: Subject ID (subject-level stratified split; visits kept together)\n"
            "- Train subjects: %d, Test subjects: %d, overlap: 0\n"
            "- Train rows: %d, Test rows: %d\n\n"
            "## Preprocessing (leakage-safe)\n"
            "- Pipeline fitted on training subjects ONLY: SimpleImputer(strategy='median')\n"
            "- No StandardScaler (not needed for Random Forest)\n"
            "- No df.fillna(df.median()) before the split\n\n"
            "## Model (unchanged production config)\n"
            "- RandomForestClassifier(n_estimators=200, random_state=42, "
            "class_weight={0:1,1:4,2:2})\n\n"
            "## Results\n"
            "See metrics.json.  Classification is multiclass; per-class metrics for "
            "Converted (class 1) are unstable because of tiny test subject count.\n"
        )
        % (
            n_train_subjects, n_test_subjects, len(df_train), len(df_test)
        ),
        encoding="utf-8",
    )

    print("\n=== SECTION 12: SAVED EXPERIMENT OUTPUTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SUMMARY (SECTION 18) ===")
    print("1.  dataset summary:            %d rows, %d subjects (see metrics.json)" % (len(df), df[SUBJECT_ID_COL].nunique()))
    print("2.  subject summary:            rows/subject min=%d max=%d mean=%.2f"
          % (subject_structure["rows_per_subject_min"], subject_structure["rows_per_subject_max"], subject_structure["rows_per_subject_mean"]))
    print("3.  subject-level class dist:  %s" % split_info["train_subject_class_dist"])
    print("4.  train/test subjects:       %d / %d" % (n_train_subjects, n_test_subjects))
    print("5.  train/test rows:           %d / %d" % (len(df_train), len(df_test)))
    print("6.  subject overlap count:     0 (verified)")
    print("7.  preprocessing pipeline:    SimpleImputer(median) fitted on train subjects only; no scaler")
    print("8.  RF configuration:          n_estimators=200, random_state=42, class_weight={0:1,1:4,2:2}")
    print("9.  metrics:                   see metrics.json / printout above")
    print("10. confusion matrix:          saved to confusion_matrix.csv")
    print("11. Converted test subjects:   %d (%d rows)" % (split_rows["converted_test_subjects"], int((df_test[LABEL_COL] == 1).sum())))
    print("12. old ~76%% vs new:           %.4f leakage-free (expected to drop)" % metrics["accuracy"])
    print("13. raw probabilities:         reported above as 'uncalibrated Random Forest probabilities'")
    print("14. production files:          unchanged (this script only reads Dataset.csv and writes to experiments dir)")
    print("15. saved artifacts:           %s" % OUT_DIR)
    print("16. next step:                 review this baseline, then (separate phase) benchmark models / calibration")


if __name__ == "__main__":
    main()
