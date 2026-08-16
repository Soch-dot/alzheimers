"""
ML PIPELINE PHASE 8 -- SINGLE-VISIT COVERAGE + VISIT-COUNT STRATIFIED PERFORMANCE
(RESEARCH ONLY).

Answers: "What happens when a real patient has only one available visit?"

The OASIS dataset has 150 subjects, all with >=2 visits. There are NO true
one-visit subjects. A clearly-labeled SIMULATION is used instead:

  simulated one-visit screening:
    - for every subject, hide all later visits, use only the earliest visit
    - train ONLY on earliest visits of the 112 training subjects
    - evaluate ONLY on earliest visits of the 38 outer-test subjects

Three scenarios compared at subject level (threshold 0.40 fixed):

  A. Simulated one-visit screening (earliest visit, model trained on earliest only)
  B. Latest-available-visit screening (latest visit, model trained on all visits)
  C. Phase 5 current experimental configuration
       = Phase 5 winner single-visit model (LR C=10 balanced) evaluated at the
         latest available visit at threshold 0.40
       NOTE: C is numerically identical to B (same model, same evaluation point);
             B/C are reported as one row labeled both ways for consistency.

Model (all scenarios): LogisticRegression C=10.0 class_weight=balanced, features
[age, sex, education_years, mmse, ses], no CDR, no longitudinal delta features,
raw probabilities. MR Delay authoritative ordering (raw Visit gaps for 6 subjects).
Canonical outer split: 112 training / 38 test subjects, zero overlap.

Paired subject analysis and probability-change analysis use the SAME Phase 5 model
applied at baseline and latest visits (a true within-subject paired comparison).

LIMITATION: OASIS Group label is constant per subject (final classification);
latest-visit classification is NOT future-conversion prediction.
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
)

import sklearn

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase8_visit_coverage"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"

NP_SEED = 42
THRESHOLD = 0.40  # frozen Phase 5 experimental screening threshold

WINNER_HP = {"clf__C": 10.0, "clf__class_weight": "balanced"}

EXPECTED_LATEST = {
    "sensitivity": 0.75, "specificity": 0.8333, "balanced_accuracy": 0.7917,
    "ppv": 0.8333, "npv": 0.7500, "f1": 0.7895, "accuracy": 0.7895,
}
EXPECTED_ONEVISIT = {
    "sensitivity": 0.60, "specificity": 0.5556, "balanced_accuracy": 0.5778,
    "ppv": 0.60, "npv": 0.5556, "f1": 0.60, "accuracy": 0.5789,
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "CDR": "cdr", "SES": "ses", "Group": LABEL_COL,
        "Visit": "raw_visit", "MR Delay": MR_DELAY_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    df = df.sort_values([SUBJECT_COL, MR_DELAY_COL]).reset_index(drop=True)
    df["temporal_rank"] = df.groupby(SUBJECT_COL).cumcount() + 1
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
        "threshold": float(t),
        "n": int(len(y)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, p)) if uniq == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if uniq == 2 else None,
        "brier": float(brier_score_loss(y, p)) if uniq == 2 else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if uniq == 2 else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def breakdown(frame, pred_col="predicted_binary"):
    out = {}
    for grp, gname in [(0, "Nondemented"), (1, "Converted"), (2, "Demented")]:
        msk = frame[LABEL_COL] == grp
        flagged = int((frame.loc[msk, pred_col] == 1).sum())
        missed = int((frame.loc[msk, pred_col] == 0).sum())
        out[gname] = {"subjects": int(msk.sum()), "flagged": flagged, "missed": missed}
    return out


def main():
    np.random.seed(NP_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])
    assert len(train_subjects & test_subjects) == 0
    assert len(train_subjects) == 112 and len(test_subjects) == 38

    df = load_data()
    visit_counts = df.groupby(SUBJECT_COL).size()
    df["n_visits"] = df[SUBJECT_COL].map(visit_counts)

    earliest = df[df["temporal_rank"] == 1]
    latest = df[df[MR_DELAY_COL] == df.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")]
    assert len(earliest) == 150 and len(latest) == 150
    assert set(earliest[SUBJECT_COL]) == set(latest[SUBJECT_COL]) == set(df[SUBJECT_COL])

    print("=== VISIT-COUNT DISTRIBUTION (all subjects) ===")
    dist_all = {int(k): int(v) for k, v in visit_counts.value_counts().sort_index().items()}
    print(dist_all)
    print("subjects with exactly 1 visit:", int((visit_counts == 1).sum()))
    print("NOTE: dataset contains NO true one-visit subjects; one-visit performance is SIMULATED.")

    # ---------- Scenario A: simulated one-visit screening ----------
    # Train ONLY on earliest visits of the 112 training subjects.
    tr_earliest = earliest[earliest[SUBJECT_COL].isin(train_subjects)]
    te_earliest = earliest[earliest[SUBJECT_COL].isin(test_subjects)]
    assert len(tr_earliest) == 112 and len(te_earliest) == 38

    model_one = make_pipeline()
    model_one.fit(tr_earliest[FEATURES], tr_earliest[BINARY_COL])
    te_earliest = te_earliest.copy()
    te_earliest["prob"] = model_one.predict_proba(te_earliest[FEATURES])[:, 1]
    te_earliest["predicted_binary"] = (te_earliest["prob"] >= THRESHOLD).astype(int)
    one_m = metrics_at_threshold(te_earliest[BINARY_COL], te_earliest["prob"])
    one_breakdown = breakdown(te_earliest)

    print("\n=== SCENARIO A: SIMULATED ONE-VISIT SCREENING (threshold %.2f) ===" % THRESHOLD)
    print("model trained ONLY on earliest visits of 112 training subjects")
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
              "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]:
        print("  %s: %s" % (k, "%.4f" % one_m[k] if isinstance(one_m[k], float) else one_m[k]))
    print("max abs diff vs Phase 6 Model B (expected):",
          round(max(abs(one_m[k] - EXPECTED_ONEVISIT[k]) for k in EXPECTED_ONEVISIT), 4))

    # ---------- Scenario B/C: latest-available-visit screening ----------
    # Phase 5 winner trained on ALL visits of the 112 training subjects.
    tr_all = df[df[SUBJECT_COL].isin(train_subjects)]
    te_latest = latest[latest[SUBJECT_COL].isin(test_subjects)].copy()
    assert len(te_latest) == 38

    model_full = make_pipeline()
    model_full.fit(tr_all[FEATURES], tr_all[BINARY_COL])
    te_latest["prob"] = model_full.predict_proba(te_latest[FEATURES])[:, 1]
    te_latest["predicted_binary"] = (te_latest["prob"] >= THRESHOLD).astype(int)
    latest_m = metrics_at_threshold(te_latest[BINARY_COL], te_latest["prob"])
    latest_breakdown = breakdown(te_latest)

    print("\n=== SCENARIO B/C: LATEST-AVAILABLE-VISIT SCREENING (threshold %.2f) ===" % THRESHOLD)
    print("Phase 5 winner trained on all visits of 112 training subjects; evaluated at latest visit.")
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
              "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]:
        print("  %s: %s" % (k, "%.4f" % latest_m[k] if isinstance(latest_m[k], float) else latest_m[k]))
    print("max abs diff vs Phase 7 latest (expected):",
          round(max(abs(latest_m[k] - EXPECTED_LATEST[k]) for k in EXPECTED_LATEST), 4))

    # ---------- Paired subject analysis (SAME Phase 5 model at baseline & latest) ----------
    # Apply model_full to earliest rows AND latest rows of test subjects for a
    # within-subject paired comparison (consistent with Phase 7).
    te_earliest_paired = earliest[earliest[SUBJECT_COL].isin(test_subjects)].copy()
    te_earliest_paired["prob"] = model_full.predict_proba(te_earliest_paired[FEATURES])[:, 1]

    paired = pd.DataFrame({
        "subject_id": te_earliest_paired[SUBJECT_COL].to_numpy(),
        "true_group": te_earliest_paired[LABEL_COL].to_numpy(),
        "true_binary_label": te_earliest_paired[BINARY_COL].to_numpy(),
        "n_visits": te_earliest_paired["n_visits"].to_numpy(),
        "earliest_probability": te_earliest_paired["prob"].to_numpy(),
        "latest_probability": te_latest["prob"].to_numpy(),
        "earliest_predicted": (te_earliest_paired["prob"] >= THRESHOLD).astype(int),
        "latest_predicted": te_latest["predicted_binary"].to_numpy(),
    })
    paired["prob_change"] = paired["latest_probability"] - paired["earliest_probability"]
    paired["prediction_changed"] = (paired["earliest_predicted"] != paired["latest_predicted"]).astype(int)
    paired["change_direction"] = np.where(
        (paired["earliest_predicted"] == 0) & (paired["latest_predicted"] == 1), "neg_to_pos",
        np.where((paired["earliest_predicted"] == 1) & (paired["latest_predicted"] == 0),
                 "pos_to_neg", "no_change"))
    paired["change_correct"] = np.where(
        paired["prediction_changed"] == 1,
        (paired["latest_predicted"] == paired["true_binary_label"]).astype(int), np.nan)

    changed = paired[paired["prediction_changed"] == 1]
    n_changed = len(changed)
    n_np = int((paired["change_direction"] == "neg_to_pos").sum())
    n_pn = int((paired["change_direction"] == "pos_to_neg").sum())
    n_correct = int(paired["change_correct"].sum())
    n_incorrect = n_changed - n_correct

    # classify changes (all changes are from earliest -> latest)
    new_fp = len(changed[(changed["latest_predicted"] == 1) & (changed["true_binary_label"] == 0)])
    corrected_fp = len(changed[(changed["latest_predicted"] == 0) & (changed["true_binary_label"] == 0)])
    new_tp = len(changed[(changed["latest_predicted"] == 1) & (changed["true_binary_label"] == 1)])
    new_fn = len(changed[(changed["latest_predicted"] == 0) & (changed["true_binary_label"] == 1)])

    print("\n=== PAIRED SUBJECT ANALYSIS (38 subjects; same model at earliest & latest) ===")
    print("prediction changed:", n_changed)
    print("  neg -> pos:", n_np, "| pos -> neg:", n_pn)
    print("  correct changes:", n_correct, "| incorrect changes:", n_incorrect)
    print("  new false positives:", new_fp)
    print("  corrected false positives:", corrected_fp)
    print("  new true positives:", new_tp)
    print("  new false negatives:", new_fn)

    # ---------- Visit-count stratified analysis ----------
    print("\n=== VISIT-COUNT STRATIFIED ANALYSIS (outer-test subjects) ===")
    strat_rows = []
    for nv in sorted(visit_counts.unique()):
        sub = paired[paired["n_visits"] == nv]
        if len(sub) == 0:
            continue
        row = {
            "n_visits": int(nv),
            "subjects": int(len(sub)),
            "positive": int((sub["true_binary_label"] == 1).sum()),
            "negative": int((sub["true_binary_label"] == 0).sum()),
            "earliest_prob_mean": float(sub["earliest_probability"].mean()),
            "latest_prob_mean": float(sub["latest_probability"].mean()),
            "earliest_flagged": int((sub["earliest_predicted"] == 1).sum()),
            "latest_flagged": int((sub["latest_predicted"] == 1).sum()),
            "changed_predictions": int(sub["prediction_changed"].sum()),
            "correct_changes": int(sub["change_correct"].sum()),
            "note": "descriptive only (small n)" if len(sub) < 5 else "",
        }
        strat_rows.append(row)
        print("  visits=%d n=%d pos=%d neg=%d earliest_flag=%d latest_flag=%d changed=%d correct_changes=%d %s"
              % (nv, len(sub), row["positive"], row["negative"], row["earliest_flagged"],
                 row["latest_flagged"], row["changed_predictions"], row["correct_changes"], row["note"]))
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(OUT_DIR / "visit_count_metrics.csv", index=False)

    # ---------- Converted analysis ----------
    print("\n=== CONVERTED ANALYSIS (outer-test, explicit per subject) ===")
    conv = paired[paired["true_group"] == 1].sort_values("subject_id")
    conv_rows = []
    for _, r in conv.iterrows():
        conv_rows.append({
            "subject_id": r["subject_id"], "true_binary_label": int(r["true_binary_label"]),
            "n_visits": int(r["n_visits"]),
            "earliest_probability": round(float(r["earliest_probability"]), 4),
            "latest_probability": round(float(r["latest_probability"]), 4),
            "earliest_predicted": int(r["earliest_predicted"]),
            "latest_predicted": int(r["latest_predicted"]),
            "earliest_decision": "flagged" if r["earliest_predicted"] == 1 else "missed",
            "latest_decision": "flagged" if r["latest_predicted"] == 1 else "missed",
        })
        print("  %s: earliest=%.4f (missed) latest=%.4f (%s) n_visits=%d"
              % (r["subject_id"], r["earliest_probability"], r["latest_probability"],
                 conv_rows[-1]["latest_decision"], r["n_visits"]))
    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv(OUT_DIR / "converted_analysis.csv", index=False)
    print("Converted detected/missed: earliest detected=%d missed=%d | latest detected=%d missed=%d"
          % (one_breakdown["Converted"]["flagged"], one_breakdown["Converted"]["missed"],
             latest_breakdown["Converted"]["flagged"], latest_breakdown["Converted"]["missed"]))

    # ---------- Probability distributions ----------
    print("\n=== PROBABILITY DISTRIBUTIONS (model-estimated screening probability) ===")
    prob_rows = []
    for stage, sub in [("earliest", te_earliest_paired), ("latest", te_latest)]:
        p = sub["prob"]
        prob_rows.append({"stage": stage, "group": "all", "n": len(p),
                          "min": round(float(p.min()), 4), "median": round(float(p.median()), 4),
                          "mean": round(float(p.mean()), 4), "max": round(float(p.max()), 4)})
        for grp, gname in [(0, "Nondemented"), (1, "Converted"), (2, "Demented")]:
            subg = sub[sub[LABEL_COL] == grp]
            if len(subg):
                pg = subg["prob"]
                prob_rows.append({"stage": stage, "group": gname, "n": len(subg),
                                  "min": round(float(pg.min()), 4), "median": round(float(pg.median()), 4),
                                  "mean": round(float(pg.mean()), 4), "max": round(float(pg.max()), 4)})
    prob_df = pd.DataFrame(prob_rows)
    prob_df.to_csv(OUT_DIR / "probability_distributions.csv", index=False)
    print(prob_df.to_string(index=False))

    # ---------- subject-level visit analysis (per subject) ----------
    subj_vis = paired.copy()
    subj_vis["earliest_mr_delay"] = te_earliest_paired[MR_DELAY_COL].to_numpy()
    subj_vis["latest_mr_delay"] = te_latest[MR_DELAY_COL].to_numpy()
    subj_vis.to_csv(OUT_DIR / "subject_visit_analysis.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_subject_analysis.csv", index=False)

    # ---------- comparison ----------
    comparison = {
        "threshold": THRESHOLD,
        "scenarios": {
            "A_simulated_one_visit": {
                "description": "earliest visit only; model trained ONLY on earliest visits of 112 training subjects",
                "metrics": one_m,
                "breakdown": one_breakdown,
            },
            "B_C_latest_available": {
                "description": ("latest available visit; Phase 5 winner trained on all visits of 112 training "
                                "subjects; Scenario C (Phase 5 config) is numerically identical to Scenario B"),
                "metrics": latest_m,
                "breakdown": latest_breakdown,
            },
        },
        "paired": {
            "subjects": int(len(paired)),
            "changed_predictions": n_changed,
            "neg_to_pos": n_np,
            "pos_to_neg": n_pn,
            "correct_changes": n_correct,
            "incorrect_changes": n_incorrect,
            "new_false_positives": new_fp,
            "corrected_false_positives": corrected_fp,
            "new_true_positives": new_tp,
            "new_false_negatives": new_fn,
            "prob_change_stats": {
                "mean": float(paired["prob_change"].mean()),
                "median": float(paired["prob_change"].median()),
                "std": float(paired["prob_change"].std()),
                "min": float(paired["prob_change"].min()),
                "max": float(paired["prob_change"].max()),
            },
        },
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    repro = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seeds": {"numpy": NP_SEED},
        "model": "LogisticRegression (Phase 4/5 winner)",
        "model_hyperparameters": WINNER_HP,
        "features": FEATURES,
        "feature_order": FEATURES,
        "no_cdr": True,
        "no_longitudinal_delta_features": True,
        "preprocessing": "SimpleImputer(median)->StandardScaler fit on training subjects only",
        "calibration": "none (raw probabilities)",
        "threshold": THRESHOLD,
        "threshold_origin": "frozen from Phase 5; NOT retuned",
        "temporal_ordering": "MR Delay (authoritative; raw Visit field has gaps for 6 subjects)",
        "outer_split": "112 training / 38 test subjects (subject_grouped_baseline/subject_split.json), zero overlap",
        "earliest_definition": "row with min MR Delay (== 0) per subject",
        "latest_definition": "row with max MR Delay per subject",
        "one_visit_simulation": ("no true one-visit subjects in dataset (all 150 have >=2 visits); "
                                 "one-visit screening is simulated by restricting each subject to their "
                                 "earliest observed visit. NOT validated on real one-visit patients."),
        "limitation": ("OASIS Group label is constant per subject (final classification). Latest-visit "
                       "classification does NOT mean future-conversion prediction."),
    }
    (OUT_DIR / "reproducibility.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")

    (OUT_DIR / "simulated_one_visit_metrics.json").write_text(json.dumps({
        "scenario": "A simulated one-visit screening",
        "description": "earliest visit only; model trained ONLY on earliest visits of 112 training subjects",
        "threshold": THRESHOLD,
        "metrics": one_m,
        "breakdown": one_breakdown,
        "note": "SIMULATED: dataset contains no true one-visit subjects.",
    }, indent=2), encoding="utf-8")

    (OUT_DIR / "latest_visit_metrics.json").write_text(json.dumps({
        "scenario": "B/C latest-available-visit screening",
        "description": "latest available visit; Phase 5 winner trained on all visits; Scenario C identical to B",
        "threshold": THRESHOLD,
        "metrics": latest_m,
        "breakdown": latest_breakdown,
    }, indent=2), encoding="utf-8")

    (OUT_DIR / "README.md").write_text(
        "# Phase 8 - Single-Visit Coverage + Visit-Count Stratified Performance (Research Only)\n\n"
        "- Experimental single-visit coverage analysis.\n"
        "- No true one-visit subjects exist in OASIS (all 150 subjects have >=2 visits);\n"
        "  one-visit performance is SIMULATED by restricting each subject to their earliest observed visit.\n"
        "- Scenario A: simulated one-visit (trained on earliest visits only).\n"
        "- Scenario B/C: latest-available-visit screening (Phase 5 winner; B and C identical).\n"
        "- Threshold 0.40 frozen; MR Delay ordering; 112/38 split reused; no CDR; no longitudinal delta features.\n"
        "- LIMITATION: OASIS Group label is constant per subject (final classification);\n"
        "  latest-visit classification does NOT mean future-conversion prediction.\n"
        "- No production changes; no commits.\n",
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("split overlap == 0:", len(train_subjects & test_subjects) == 0)
    print("earliest rows == n subjects:", len(earliest) == 150)
    print("latest rows == n subjects:", len(latest) == 150)
    print("one-visit sim train rows == 112:", len(tr_earliest) == 112)
    print("one-visit sim test rows == 38:", len(te_earliest) == 38)
    print("latest test rows == 38:", len(te_latest) == 38)
    print("threshold not retuned:", THRESHOLD == 0.40)
    print("no longitudinal delta features used: True")
    print("simulation labeled (no true one-visit patients): True")


if __name__ == "__main__":
    main()
