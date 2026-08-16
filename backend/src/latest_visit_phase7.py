"""
ML PIPELINE PHASE 7 -- LATEST-AVAILABLE-VISIT SCREENING + ELIGIBILITY ANALYSIS
(RESEARCH ONLY).

Compares two realistic product modes using the SAME experimental binary
screening candidate (Phase 4/5 winner: LogisticRegression C=10 class_weight
balanced, features [age, sex, education_years, mmse, ses], no CDR, no
longitudinal delta features, raw probabilities):

  Mode A (BASELINE screening):  subject's earliest visit  (min MR Delay == 0).
  Mode B (LATEST-AVAILABLE screening): subject's most recent visit
        (max MR Delay).

- Threshold fixed at 0.40 (Phase 5 experimental screening threshold; NOT
  retuned here).
- MR Delay is the authoritative temporal ordering (raw Visit field has gaps
  for 6 subjects; not used for ordering).
- Canonical outer split reused: 112 training / 38 test subjects, zero overlap.
- Paired subject-level comparison, probability-change analysis, Converted/
  Demented/Nondemented breakdown, and visit-availability/eligibility design.

SCIENTIFIC LIMITATION (must be stated):
The OASIS Group label is constant per subject and corresponds to final subject
classification. Latest-visit classification therefore does NOT mean the model
predicted future conversion. It means the model classifies the subject using
information available at the latest observed visit.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase7_latest_visit"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"

NP_SEED = 42
N_BOOT = 5000
THRESHOLD = 0.40  # frozen Phase 5 experimental screening threshold

WINNER_HP = {"clf__C": 10.0, "clf__class_weight": "balanced"}

EXPECTED_P5 = {
    "sensitivity": 0.75, "specificity": 0.8333, "balanced_accuracy": 0.7917,
    "ppv": 0.8333, "npv": 0.7500, "f1": 0.7895, "accuracy": 0.7895,
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
    # Canonical temporal order: MR Delay (strictly increasing within subject).
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


def bootstrap_ci(y, p, t=THRESHOLD, n_boot=N_BOOT, seed=NP_SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    idx = np.arange(len(y))
    cols = ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv"]
    samples = {c: [] for c in cols}
    for _ in range(n_boot):
        bi = rng.choice(idx, size=len(idx), replace=True)
        m = metrics_at_threshold(y[bi], p[bi], t)
        for c in cols:
            samples[c].append(m[c])
    out = {}
    for c in cols:
        arr = np.array(samples[c])
        out[c] = {"mean": float(arr.mean()),
                  "ci95_lo": float(np.percentile(arr, 2.5)),
                  "ci95_hi": float(np.percentile(arr, 97.5))}
    return out


def group_breakdown(frame):
    out = {}
    for grp, gname in [(0, "Nondemented"), (1, "Converted"), (2, "Demented")]:
        msk = frame[LABEL_COL] == grp
        flagged = int((frame.loc[msk, "predicted_binary"] == 1).sum())
        missed = int((frame.loc[msk, "predicted_binary"] == 0).sum())
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

    # ---- integrity: exactly one baseline & one latest row per subject ----
    n_base = df.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("min")
    n_latest = df.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")
    assert (df[MR_DELAY_COL] == n_base).groupby(df[SUBJECT_COL]).sum().eq(1).all()
    assert (df[MR_DELAY_COL] == n_latest).groupby(df[SUBJECT_COL]).sum().eq(1).all()
    assert ((df[MR_DELAY_COL] == n_base) == (df[MR_DELAY_COL] == 0)).all()  # baseline MR Delay==0

    base_rows = df[df[MR_DELAY_COL] == 0]
    latest_rows = df[df[MR_DELAY_COL] == n_latest]
    print("=== ELIGIBILITY CHECK ===")
    print("baseline rows == n subjects:", len(base_rows) == df[SUBJECT_COL].nunique())
    print("latest rows == n subjects:", len(latest_rows) == df[SUBJECT_COL].nunique())

    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]

    # ---- fit the existing experimental model on all 112 training subjects ----
    model = make_pipeline()
    model.fit(df_train[FEATURES], df_train[BINARY_COL])

    # ---- evaluate at baseline visit and latest visit (test subjects) ----
    test_base = base_rows[base_rows[SUBJECT_COL].isin(test_subjects)].copy()
    test_latest = latest_rows[latest_rows[SUBJECT_COL].isin(test_subjects)].copy()
    assert len(test_base) == 38 and len(test_latest) == 38
    assert set(test_base[SUBJECT_COL]) == test_subjects == set(test_latest[SUBJECT_COL])

    test_base["prob"] = model.predict_proba(test_base[FEATURES])[:, 1]
    test_latest["prob"] = model.predict_proba(test_latest[FEATURES])[:, 1]
    test_base["predicted_binary"] = (test_base["prob"] >= THRESHOLD).astype(int)
    test_latest["predicted_binary"] = (test_latest["prob"] >= THRESHOLD).astype(int)

    base_m = metrics_at_threshold(test_base[BINARY_COL], test_base["prob"])
    latest_m = metrics_at_threshold(test_latest[BINARY_COL], test_latest["prob"])

    print("\n=== MODE A: BASELINE SCREENING (threshold %.2f) ===" % THRESHOLD)
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
              "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]:
        print("  %s: %s" % (k, "%.4f" % base_m[k] if isinstance(base_m[k], float) else base_m[k]))

    print("\n=== MODE B: LATEST-AVAILABLE SCREENING (threshold %.2f) ===" % THRESHOLD)
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
              "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]:
        print("  %s: %s" % (k, "%.4f" % latest_m[k] if isinstance(latest_m[k], float) else latest_m[k]))

    # Phase 6 latest-visit reference check
    print("\nPhase 6 latest-visit reference: sens=0.75 spec=0.833 balacc=0.792")
    print("max abs diff:", round(max(abs(latest_m[k] - EXPECTED_P5[k]) for k in EXPECTED_P5), 4))

    # ---- paired subject comparison ----
    visit_counts = df.groupby(SUBJECT_COL).size()
    paired = pd.DataFrame({
        "subject_id": test_base[SUBJECT_COL].to_numpy(),
        "true_group": test_base[LABEL_COL].to_numpy(),
        "true_binary_label": test_base[BINARY_COL].to_numpy(),
        "n_visits": test_base[SUBJECT_COL].map(visit_counts).to_numpy(),
        "baseline_mr_delay": test_base[MR_DELAY_COL].to_numpy(),
        "latest_mr_delay": test_latest[MR_DELAY_COL].to_numpy(),
        "baseline_probability": test_base["prob"].to_numpy(),
        "latest_probability": test_latest["prob"].to_numpy(),
        "baseline_predicted": test_base["predicted_binary"].to_numpy(),
        "latest_predicted": test_latest["predicted_binary"].to_numpy(),
    })
    paired["probability_change"] = paired["latest_probability"] - paired["baseline_probability"]
    paired["prediction_changed"] = (paired["baseline_predicted"] != paired["latest_predicted"]).astype(int)
    paired["change_direction"] = np.where(
        (paired["baseline_predicted"] == 0) & (paired["latest_predicted"] == 1), "neg_to_pos",
        np.where((paired["baseline_predicted"] == 1) & (paired["latest_predicted"] == 0),
                 "pos_to_neg", "no_change"))
    # correctness of a change: latest prediction is correct
    paired["change_correct"] = np.where(
        paired["prediction_changed"] == 1,
        (paired["latest_predicted"] == paired["true_binary_label"]).astype(int),
        np.nan)

    n_changed = int(paired["prediction_changed"].sum())
    n_np = int((paired["change_direction"] == "neg_to_pos").sum())
    n_pn = int((paired["change_direction"] == "pos_to_neg").sum())
    n_correct = int(paired["change_correct"].sum())  # NaN-safe (only changed rows non-NaN)
    n_incorrect = n_changed - n_correct

    print("\n=== PAIRED SUBJECT COMPARISON (38 subjects) ===")
    print("subjects whose prediction changed (baseline vs latest):", n_changed)
    print("  changed negative -> positive:", n_np)
    print("  changed positive -> negative:", n_pn)
    print("  changes that were CORRECT (latest matches true label):", n_correct)
    print("  changes that were INCORRECT:", n_incorrect)

    # ---- probability change analysis ----
    print("\n=== PROBABILITY CHANGE (latest - baseline) per subject ===")
    pc = paired["probability_change"]
    print("mean=%.4f median=%.4f std=%.4f min=%.4f max=%.4f"
          % (pc.mean(), pc.median(), pc.std(), pc.min(), pc.max()))
    for grp, gname in [(0, "Nondemented"), (1, "Converted"), (2, "Demented")]:
        sub = paired[paired["true_group"] == grp]
        if len(sub):
            print("  %s (n=%d): mean change=%.4f median=%.4f"
                  % (gname, len(sub), sub["probability_change"].mean(), sub["probability_change"].median()))
    pos = paired[paired["true_binary_label"] == 1]
    neg = paired[paired["true_binary_label"] == 0]
    print("  positive subjects (n=%d): mean change=%.4f" % (len(pos), pos["probability_change"].mean()))
    print("  negative subjects (n=%d): mean change=%.4f" % (len(neg), neg["probability_change"].mean()))
    print("NOTE: described as 'change in screening score between available visits', "
          "NOT progression prediction.")

    # ---- Converted / Demented / Nondemented ----
    print("\n=== CONVERTED / DEMENTED / NONDEMENTED (flagged at threshold %.2f) ===" % THRESHOLD)
    base_bd = group_breakdown(test_base)
    latest_bd = group_breakdown(test_latest)
    for gname in ["Converted", "Demented", "Nondemented"]:
        print("  %s: baseline flagged=%d missed=%d | latest flagged=%d missed=%d"
              % (gname, base_bd[gname]["flagged"], base_bd[gname]["missed"],
                 latest_bd[gname]["flagged"], latest_bd[gname]["missed"]))

    # ---- visit availability ----
    print("\n=== VISIT AVAILABILITY / ELIGIBILITY ===")
    vc_all = df.groupby(SUBJECT_COL).size()
    vc_test = df_test.groupby(SUBJECT_COL).size()
    dist_all = {int(k): int(v) for k, v in vc_all.value_counts().sort_index().items()}
    dist_test = {int(k): int(v) for k, v in vc_test.value_counts().sort_index().items()}
    print("all subjects (%d):" % len(vc_all), dist_all)
    print("test subjects (%d):" % len(vc_test), dist_test)
    product_logic = {
        "1 visit": "current == baseline == latest (same row); screen with the single available visit.",
        "2 visits": "screen at latest available visit; no previous visit required.",
        "3+ visits": "screen at latest available visit; ignore earlier visits for scoring.",
        "general": "latest_visit = row with max MR Delay for the subject; baseline_visit = row with min MR Delay (0).",
        "no_history_required": True,
    }
    print(json.dumps(product_logic, indent=2))

    # ---- bootstrap CIs ----
    base_ci = bootstrap_ci(test_base[BINARY_COL].to_numpy(), test_base["prob"].to_numpy())
    latest_ci = bootstrap_ci(test_latest[BINARY_COL].to_numpy(), test_latest["prob"].to_numpy())

    # ---- artifacts ----
    repro = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seeds": {"numpy": NP_SEED, "bootstrap": NP_SEED},
        "model": "LogisticRegression (Phase 4/5 winner)",
        "model_hyperparameters": WINNER_HP,
        "features": FEATURES,
        "feature_order": FEATURES,
        "no_cdr": True,
        "no_longitudinal_delta_features": True,
        "preprocessing": "SimpleImputer(median)->StandardScaler fit on training subjects only",
        "calibration": "none (raw probabilities)",
        "threshold": THRESHOLD,
        "threshold_origin": "frozen from Phase 5; NOT retuned here",
        "temporal_ordering": "MR Delay (authoritative; raw Visit field has gaps for 6 subjects)",
        "outer_split": "112 training / 38 test subjects (subject_grouped_baseline/subject_split.json), zero overlap",
        "baseline_definition": "row with min MR Delay (== 0) per subject",
        "latest_definition": "row with max MR Delay per subject",
        "single_visit_handling": "for a real 1-visit patient, latest_visit == baseline_visit; patient is NOT rejected for lacking follow-up",
        "limitation": ("OASIS Group label is constant per subject and corresponds to final "
                       "subject classification. Latest-visit classification does NOT mean the "
                       "model predicted future conversion; it classifies the subject using "
                       "information available at the latest observed visit."),
    }
    (OUT_DIR / "reproducibility.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")

    (OUT_DIR / "baseline_metrics.json").write_text(json.dumps({
        "mode": "A baseline screening (earliest visit, MR Delay==0)",
        "threshold": THRESHOLD,
        "metrics": base_m,
        "bootstrap_ci": base_ci,
        "breakdown": base_bd,
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "latest_metrics.json").write_text(json.dumps({
        "mode": "B latest-available-visit screening (max MR Delay)",
        "threshold": THRESHOLD,
        "metrics": latest_m,
        "bootstrap_ci": latest_ci,
        "breakdown": latest_bd,
    }, indent=2), encoding="utf-8")

    paired.to_csv(OUT_DIR / "paired_subject_comparison.csv", index=False)
    pc_df = paired[["subject_id", "true_group", "true_binary_label", "baseline_probability",
                    "latest_probability", "probability_change", "baseline_predicted",
                    "latest_predicted", "prediction_changed", "change_direction",
                    "change_correct", "n_visits"]]
    pc_df.to_csv(OUT_DIR / "subject_probability_changes.csv", index=False)

    (OUT_DIR / "converted_analysis.json").write_text(json.dumps({
        "total_subjects_all": {"Converted": int((df.groupby(SUBJECT_COL)[LABEL_COL].first() == 1).sum()),
                               "Demented": int((df.groupby(SUBJECT_COL)[LABEL_COL].first() == 2).sum()),
                               "Nondemented": int((df.groupby(SUBJECT_COL)[LABEL_COL].first() == 0).sum())},
        "outer_test_subjects": {
            "Converted": latest_bd["Converted"]["subjects"],
            "Demented": latest_bd["Demented"]["subjects"],
            "Nondemented": latest_bd["Nondemented"]["subjects"],
        },
        "baseline_flagged": {k: v["flagged"] for k, v in base_bd.items()},
        "baseline_missed": {k: v["missed"] for k, v in base_bd.items()},
        "latest_flagged": {k: v["flagged"] for k, v in latest_bd.items()},
        "latest_missed": {k: v["missed"] for k, v in latest_bd.items()},
        "statistical_note": "Converted n=14 total / 4 in outer test; no statistical significance claimed.",
    }, indent=2), encoding="utf-8")

    (OUT_DIR / "visit_count_distribution.json").write_text(json.dumps({
        "all_subjects": dist_all,
        "test_subjects": dist_test,
        "product_logic": product_logic,
    }, indent=2), encoding="utf-8")

    comp = pd.DataFrame([{
        "setting": "baseline", **{k: base_m[k] for k in
            ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
             "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]},
        "n": base_m["n"], "positive": base_m["positive"], "negative": base_m["negative"],
        "tp": base_m["tp"], "fp": base_m["fp"], "tn": base_m["tn"], "fn": base_m["fn"],
    }, {
        "setting": "latest", **{k: latest_m[k] for k in
            ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
             "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]},
        "n": latest_m["n"], "positive": latest_m["positive"], "negative": latest_m["negative"],
        "tp": latest_m["tp"], "fp": latest_m["fp"], "tn": latest_m["tn"], "fn": latest_m["fn"],
    }])
    comp.to_csv(OUT_DIR / "comparison.csv", index=False)

    paired_stats = {
        "subjects": int(len(paired)),
        "changed_prediction": n_changed,
        "neg_to_pos": n_np,
        "pos_to_neg": n_pn,
        "changes_correct": n_correct,
        "changes_incorrect": n_incorrect,
        "prob_change": {"mean": float(pc.mean()), "median": float(pc.median()),
                        "std": float(pc.std()), "min": float(pc.min()), "max": float(pc.max())},
        "prob_change_by_group": {
            gname: {"n": int((paired["true_group"] == grp).sum()),
                    "mean": float(paired.loc[paired["true_group"] == grp, "probability_change"].mean()),
                    "median": float(paired.loc[paired["true_group"] == grp, "probability_change"].median())}
            for grp, gname in [(0, "Nondemented"), (1, "Converted"), (2, "Demented")]
            if (paired["true_group"] == grp).sum() > 0},
        "prob_change_positive_mean": float(pos["probability_change"].mean()),
        "prob_change_negative_mean": float(neg["probability_change"].mean()),
        "note": "probability_change = latest_probability - baseline_probability; described as 'change in screening score between available visits', NOT progression prediction.",
    }
    (OUT_DIR / "paired_analysis.json").write_text(json.dumps(paired_stats, indent=2), encoding="utf-8")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for grp, gname, color in [(1, "Converted", "C1"), (2, "Demented", "C2"), (0, "Nondemented", "C0")]:
        sub = paired[paired["true_group"] == grp]
        if len(sub):
            ax.scatter(sub["baseline_probability"], sub["latest_probability"],
                       label=gname, alpha=0.8, c=color, s=60)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.axvline(THRESHOLD, color="gray", ls=":", alpha=0.7)
    ax.axhline(THRESHOLD, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("Baseline probability"); ax.set_ylabel("Latest probability")
    ax.set_title("Baseline vs Latest screening score (threshold %.2f)" % THRESHOLD)
    ax.legend()
    ax2 = axes[1]
    ax2.hist(paired.loc[paired["true_binary_label"] == 1, "probability_change"],
             bins=16, alpha=0.6, label="Positive", color="C3")
    ax2.hist(paired.loc[paired["true_binary_label"] == 0, "probability_change"],
             bins=16, alpha=0.6, label="Negative", color="C0")
    ax2.axvline(0, color="k", ls="--", alpha=0.5)
    ax2.set_xlabel("Probability change (latest - baseline)")
    ax2.set_ylabel("Subjects")
    ax2.set_title("Change in screening score between available visits")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_vs_latest.png", dpi=150)
    plt.close()

    (OUT_DIR / "README.md").write_text(
        "# Phase 7 - Latest-Available-Visit Screening + Eligibility (Research Only)\n\n"
        "- Experimental latest-available-visit screening analysis.\n"
        "- Same Phase 4/5 binary candidate (LR C=10 balanced; age, sex, education_years, mmse, ses).\n"
        "- Threshold 0.40 frozen; MR Delay authoritative ordering; 112/38 split reused.\n"
        "- Baseline vs latest-available-visit comparison, paired subject analysis, probability changes.\n"
        "- No CDR; no longitudinal delta features; no production changes; no commits.\n"
        "- LIMITATION: OASIS Group label is constant per subject (final classification).\n"
        "  Latest-visit classification does NOT mean the model predicted future conversion;\n"
        "  it classifies the subject using information available at the latest observed visit.\n",
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("split overlap == 0:", len(train_subjects & test_subjects) == 0)
    print("exactly one baseline row per subject: True")
    print("exactly one latest row per subject: True")
    print("baseline MR Delay == 0 for all:", bool((base_rows[MR_DELAY_COL] == 0).all()))
    print("test subjects identical across modes:", set(test_base[SUBJECT_COL]) == set(test_latest[SUBJECT_COL]) == test_subjects)
    print("threshold not retuned:", THRESHOLD == 0.40)
    print("no longitudinal delta features used: True")
    print("outer-test labels not used for threshold selection: True")


if __name__ == "__main__":
    main()
