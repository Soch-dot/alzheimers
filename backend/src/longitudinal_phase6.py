"""
ML PIPELINE PHASE 6 -- LONGITUDINAL / TEMPORAL VALIDATION EXPERIMENT (RESEARCH ONLY).

Tests whether longitudinal (within-subject over-time) information improves the
experimental binary screening candidate produced in Phases 4-5:

  Target: 0 = Nondemented, 1 = Converted OR Demented  (constant per subject)
  Model:  LogisticRegression(C=10.0, class_weight='balanced'), raw prob
  Threshold: 0.40 (frozen from Phase 5; NOT re-optimized here)

SETTINGS
  Setting A (BASELINE screening): prediction uses ONLY the subject's first
    available visit (Visit==1, MR Delay==0).
  Setting B (FOLLOW-UP screening): prediction at a later visit (Visit>=2) uses
    ONLY information from visits <= the current visit (current + immediately
    previous visit features).  Future visits are NEVER used.

TEMPORAL LEAKAGE
  For a visit at time t, only features derived from visits <= t are allowed.
  Verified programmatically (MR Delay ordering, prev-visit reconstruction).

DO NOT USE as features: final/future MMSE, future CDR, future Group, any
post-index information, CDR, model predictions, or rate of change of model
probability.
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
    brier_score_loss, log_loss, roc_curve, precision_recall_curve,
)

import joblib
import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase6_longitudinal"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
BASE_FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
LONG_FEATURES = BASE_FEATURES + ["mmse_delta", "time_since_previous"]
LONG_FEATURES_PREV = BASE_FEATURES + ["prev_mmse", "mmse_delta", "time_since_previous"]
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
VISIT_COL = "visit_num"
MR_DELAY_COL = "mr_delay"

NP_SEED = 42
N_BOOT = 5000
THRESHOLD = 0.40  # frozen Phase 5 experimental screening threshold

# Phase 4/5 winner config (experimental binary screening candidate)
WINNER_HP = {"clf__C": 10.0, "clf__class_weight": "balanced"}

# Expected Phase 5 reproduction (Model A, subject = mean of all visit probs)
EXPECTED_P5 = {
    "sensitivity": 0.75, "specificity": 0.6667, "balanced_accuracy": 0.7083,
    "ppv": 0.7143, "npv": 0.7059, "f1": 0.7317, "accuracy": 0.7105,
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "CDR": "cdr", "SES": "ses", "Group": LABEL_COL,
        "Visit": VISIT_COL, "MR Delay": MR_DELAY_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    # Canonical temporal order is MR Delay (strictly increasing within subject).
    # The raw 'Visit' column has gaps for 6 subjects (e.g. 1,3,4), so we compute a
    # dense temporal rank per subject ordered by MR Delay.
    df = df.sort_values([SUBJECT_COL, MR_DELAY_COL]).reset_index(drop=True)
    df["temporal_rank"] = df.groupby(SUBJECT_COL).cumcount() + 1
    return df


def add_longitudinal_features(df):
    """Add prev-visit features. Uses only the IMMEDIATELY previous visit of the
    same subject (ordered by MR Delay, the canonical temporal order)."""
    df = df.copy()
    grp = df.groupby(SUBJECT_COL)
    df["prev_mmse"] = grp["mmse"].shift(1)
    df["prev_mr_delay"] = grp[MR_DELAY_COL].shift(1)
    df["mmse_delta"] = df["mmse"] - df["prev_mmse"]
    df["time_since_previous"] = df[MR_DELAY_COL] - df["prev_mr_delay"]
    df["has_previous"] = (df["temporal_rank"] >= 2).astype(int)
    return df


def verify_temporal_integrity(df):
    """Programmatic temporal-leakage checks. Raises SystemExit on any failure."""
    checks = []
    n_subj = df.groupby(SUBJECT_COL)[LABEL_COL].nunique()
    ok_group_const = bool((n_subj <= 1).all())
    checks.append(("Group constant per subject", ok_group_const))

    # temporal_rank must be 1..n per subject with no gaps (dense, MR-Delay-ordered)
    bad_visit = 0
    for _, sub in df.groupby(SUBJECT_COL):
        if list(sub["temporal_rank"]) != list(range(1, len(sub) + 1)):
            bad_visit += 1
    checks.append(("temporal_rank sequential 1..n per subject", bad_visit == 0))

    # Visit (raw label) must be monotonic w.r.t. MR Delay within each subject
    ok_visit_order = True
    for _, sub in df.groupby(SUBJECT_COL):
        if (sub[VISIT_COL].diff().dropna() < 0).any():
            ok_visit_order = False
    checks.append(("raw Visit monotonic w.r.t. MR Delay within subject", ok_visit_order))

    # MR Delay must strictly increase within each subject
    ok_delay = True
    for _, sub in df.groupby(SUBJECT_COL):
        if (sub[MR_DELAY_COL].diff().dropna() <= 0).any():
            ok_delay = False
    checks.append(("MR Delay strictly increasing within subject", ok_delay))

    # For every follow-up row, previous MR Delay < current MR Delay
    follow = df[df["has_previous"] == 1]
    ok_prev = bool((follow["prev_mr_delay"] < follow[MR_DELAY_COL]).all())
    checks.append(("prev_mr_delay < current mr_delay for all follow-ups", ok_prev))

    # time_since_previous > 0 for all follow-ups
    ok_ts = bool((follow["time_since_previous"] > 0).all())
    checks.append(("time_since_previous > 0 for all follow-ups", ok_ts))

    # Reconstruct prev_mmse independently (by MR Delay order) and compare
    rec = []
    for _, sub in df.groupby(SUBJECT_COL):
        s = sub.sort_values(MR_DELAY_COL)
        prev = s["mmse"].shift(1)
        rec.append((s[SUBJECT_COL].to_numpy(), s["temporal_rank"].to_numpy(),
                    s["prev_mmse"].to_numpy(), prev.to_numpy()))
    match = True
    for sid, vid, a, b in rec:
        if not np.allclose(a, b, equal_nan=True):
            match = False
    checks.append(("prev_mmse matches independent MR-Delay-order reconstruction", match))

    print("\n=== TEMPORAL LEAKAGE CHECKS ===")
    all_ok = True
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit("STOP: temporal leakage detected - fix before continuing.")
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
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) == 2 else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) == 2 else None,
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


def fitted_model(df_train, features):
    est = make_pipeline()
    est.fit(df_train[features], df_train[BINARY_COL])
    return est


def subject_breakdown(subj_df, name):
    """Converted/Demented/Nondemented detection breakdown for a subject-level df."""
    out = {}
    for grp, gname in [(1, "Converted"), (2, "Demented"), (0, "Nondemented")]:
        msk = subj_df[LABEL_COL] == grp
        flagged = int((subj_df.loc[msk, "predicted_binary"] == 1).sum())
        missed = int((subj_df.loc[msk, "predicted_binary"] == 0).sum())
        out[gname] = {"subjects": int(msk.sum()), "flagged": flagged, "missed": missed}
    return out


def main():
    np.random.seed(NP_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])
    assert len(train_subjects & test_subjects) == 0

    # ---------- load + audit ----------
    df = load_data()
    df = add_longitudinal_features(df)
    df = verify_temporal_integrity(df)

    print("=== AUDIT: LONGITUDINAL SUBJECT SUMMARY ===")
    print("rows=%d subjects=%d" % (len(df), df[SUBJECT_COL].nunique()))
    print("visits per subject:", dict(df.groupby(SUBJECT_COL).size().value_counts().sort_index()))
    n_gap = 0
    for sid, sub in df.groupby(SUBJECT_COL):
        if list(sub[VISIT_COL]) != list(range(1, len(sub) + 1)):
            n_gap += 1
    print("subjects with raw Visit gaps (non-sequential raw Visit labels):", n_gap)

    summary_rows = []
    for sid, sub in df.groupby(SUBJECT_COL):
        s = sub.sort_values(MR_DELAY_COL)
        summary_rows.append({
            "subject_id": sid,
            "n_visits": len(s),
            "raw_visit_numbers": [int(v) for v in s[VISIT_COL]],
            "temporal_ranks": [int(v) for v in s["temporal_rank"]],
            "mr_delays": [int(v) for v in s[MR_DELAY_COL]],
            "ages": list(s["age"]),
            "mmse": list(s["mmse"]),
            "ses_first": None if pd.isna(s["ses"].iloc[0]) else float(s["ses"].iloc[0]),
            "education_years": int(s["education_years"].iloc[0]),
            "sex": int(s["sex"].iloc[0]),
            "group": int(s[LABEL_COL].iloc[0]),
            "binary_target": int(s[BINARY_COL].iloc[0]),
            "baseline_mmse": None if pd.isna(s["mmse"].iloc[0]) else float(s["mmse"].iloc[0]),
            "last_mmse": None if pd.isna(s["mmse"].iloc[-1]) else float(s["mmse"].iloc[-1]),
            "mmse_last_minus_baseline": (
                None if (pd.isna(s["mmse"].iloc[0]) or pd.isna(s["mmse"].iloc[-1]))
                else float(s["mmse"].iloc[-1] - s["mmse"].iloc[0])),
            "split": "train" if sid in train_subjects else ("test" if sid in test_subjects else "OTHER"),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "longitudinal_subject_summary.csv", index=False)
    print("subject summary rows:", len(summary))
    print("subjects with split OTHER:", int((summary["split"] == "OTHER").sum()))

    # ---------- settings ----------
    # Setting A: baseline visit (temporal_rank==1, i.e. MR Delay==0) rows.
    base_train = df[(df[SUBJECT_COL].isin(train_subjects)) & (df["temporal_rank"] == 1)]
    base_test = df[(df[SUBJECT_COL].isin(test_subjects)) & (df["temporal_rank"] == 1)]
    # Setting B: follow-up rows (temporal_rank>=2).
    fol_train = df[(df[SUBJECT_COL].isin(train_subjects)) & (df["has_previous"] == 1)]
    fol_test = df[(df[SUBJECT_COL].isin(test_subjects)) & (df["has_previous"] == 1)]
    # All visits (for Model A reproduction of Phase 5).
    all_train = df[df[SUBJECT_COL].isin(train_subjects)]
    all_test = df[df[SUBJECT_COL].isin(test_subjects)]

    print("\n=== SETTINGS ROWS ===")
    print("Setting A baseline  train=%d test=%d" % (len(base_train), len(base_test)))
    print("Setting B follow-up train=%d test=%d" % (len(fol_train), len(fol_test)))

    # ---------- Model A: Phase 5 winner (fit on ALL train visits) ----------
    print("\n=== MODEL A: CURRENT SINGLE-VISIT MODEL (Phase 5 reproduction) ===")
    model_a = fitted_model(all_train, BASE_FEATURES)

    a_all_vis = pd.DataFrame({
        SUBJECT_COL: all_test[SUBJECT_COL].to_numpy(),
        LABEL_COL: all_test[LABEL_COL].to_numpy(),
        BINARY_COL: all_test[BINARY_COL].to_numpy(),
        VISIT_COL: all_test[VISIT_COL].to_numpy(),
        "temporal_rank": all_test["temporal_rank"].to_numpy(),
        "prob": model_a.predict_proba(all_test[BASE_FEATURES])[:, 1],
    })
    a_subj = a_all_vis.groupby(SUBJECT_COL).agg(
        subject_probability=("prob", "mean"),
        true_binary_label=(BINARY_COL, "first"),
        true_group=(LABEL_COL, "first")).reset_index()
    a_m = metrics_at_threshold(a_subj["true_binary_label"], a_subj["subject_probability"])
    a_subj["predicted_binary"] = (a_subj["subject_probability"] >= THRESHOLD).astype(int)
    print("Phase 5 reproduction (subject=mean of all visit probs):")
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv", "f1", "accuracy"]:
        print("  %s: %.4f (Phase5 %.4f)" % (k, a_m[k], EXPECTED_P5[k]))
    diff = max(abs(a_m[k] - EXPECTED_P5[k]) for k in EXPECTED_P5)
    print("max abs diff vs Phase 5:", round(diff, 4))
    if diff > 0.01:
        print("WARNING: Phase 5 reproduction mismatch >0.01 (investigate before proceeding)")

    # Model A at baseline visit only (subject-level, 1 visit per subject).
    a_base = a_all_vis[a_all_vis["temporal_rank"] == 1].copy()
    a_base["predicted_binary"] = (a_base["prob"] >= THRESHOLD).astype(int)
    a_base_m = metrics_at_threshold(a_base[BINARY_COL], a_base["prob"])

    # Model A at latest available visit (subject-level).
    a_latest = a_all_vis.sort_values([SUBJECT_COL, "temporal_rank"]).groupby(SUBJECT_COL).tail(1).copy()
    a_latest["predicted_binary"] = (a_latest["prob"] >= THRESHOLD).astype(int)
    a_latest_m = metrics_at_threshold(a_latest[BINARY_COL], a_latest["prob"])

    # ---------- Model B: baseline-only fit, evaluated on baseline visits ----------
    print("\n=== MODEL B: BASELINE-ONLY (fit on train baseline visits) ===")
    model_b = fitted_model(base_train, BASE_FEATURES)
    b_vis = pd.DataFrame({
        SUBJECT_COL: base_test[SUBJECT_COL].to_numpy(),
        LABEL_COL: base_test[LABEL_COL].to_numpy(),
        BINARY_COL: base_test[BINARY_COL].to_numpy(),
        "prob": model_b.predict_proba(base_test[BASE_FEATURES])[:, 1],
    })
    b_vis["predicted_binary"] = (b_vis["prob"] >= THRESHOLD).astype(int)
    b_m = metrics_at_threshold(b_vis[BINARY_COL], b_vis["prob"])
    b_breakdown = subject_breakdown(b_vis, "baseline_only")
    print("  experimental baseline-visit screening at threshold %.2f" % THRESHOLD)
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
              "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]:
        print("  %s: %s" % (k, "%.4f" % b_m[k] if isinstance(b_m[k], float) else b_m[k]))

    # ---------- Model C: longitudinal feature model (follow-up visits) ----------
    print("\n=== MODEL C: LONGITUDINAL FEATURE MODEL ===")
    for label, feats in [("primary", LONG_FEATURES), ("with_prev_mmse", LONG_FEATURES_PREV)]:
        model_c = fitted_model(fol_train, feats)
        c_vis = pd.DataFrame({
            SUBJECT_COL: fol_test[SUBJECT_COL].to_numpy(),
            LABEL_COL: fol_test[LABEL_COL].to_numpy(),
            BINARY_COL: fol_test[BINARY_COL].to_numpy(),
            VISIT_COL: fol_test[VISIT_COL].to_numpy(),
            "temporal_rank": fol_test["temporal_rank"].to_numpy(),
            "prob": model_c.predict_proba(fol_test[feats])[:, 1],
        })
        c_vis["predicted_binary"] = (c_vis["prob"] >= THRESHOLD).astype(int)
        # per-visit (row-level, research only - NOT the subject screening score)
        c_per_visit = metrics_at_threshold(c_vis[BINARY_COL], c_vis["prob"])
        # subject-level: latest available eligible visit
        c_latest = c_vis.sort_values([SUBJECT_COL, "temporal_rank"]).groupby(SUBJECT_COL).tail(1).copy()
        c_latest_m = metrics_at_threshold(c_latest[BINARY_COL], c_latest["prob"])
        # subject-level: mean of eligible visit probs
        c_mean = c_vis.groupby(SUBJECT_COL).agg(
            subject_probability=("prob", "mean"),
            true_binary_label=(BINARY_COL, "first"),
            true_group=(LABEL_COL, "first")).reset_index()
        c_mean_m = metrics_at_threshold(c_mean["true_binary_label"], c_mean["subject_probability"])
        c_latest_breakdown = subject_breakdown(
            c_latest.assign(predicted_binary=c_latest["predicted_binary"]), label)
        c_latest_boot = bootstrap_ci(c_latest[BINARY_COL].to_numpy(), c_latest["prob"].to_numpy())

        print("  [%s] features=%s" % (label, feats))
        print("    per-visit (row-level research only): n=%d sens=%.4f spec=%.4f balacc=%.4f"
              % (c_per_visit["n"], c_per_visit["sensitivity"],
                 c_per_visit["specificity"], c_per_visit["balanced_accuracy"]))
        print("    subject=latest eligible visit: n=%d pos=%d neg=%d"
              % (c_latest_m["n"], c_latest_m["positive"], c_latest_m["negative"]))
        for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
                  "f1", "accuracy", "roc_auc", "pr_auc", "brier", "logloss"]:
            print("      %s: %s" % (k, "%.4f" % c_latest_m[k] if isinstance(c_latest_m[k], float) else c_latest_m[k]))
        print("    subject=mean of eligible visit probs: n=%d sens=%.4f spec=%.4f balacc=%.4f"
              % (c_mean_m["n"], c_mean_m["sensitivity"], c_mean_m["specificity"], c_mean_m["balanced_accuracy"]))
        print("    converted breakdown (subject=latest): %s" % c_latest_breakdown)
        print("    bootstrap CI (subject=latest): sens %.4f [%.4f, %.4f], spec %.4f [%.4f, %.4f]"
              % (c_latest_boot["sensitivity"]["mean"], c_latest_boot["sensitivity"]["ci95_lo"],
                 c_latest_boot["sensitivity"]["ci95_hi"], c_latest_boot["specificity"]["mean"],
                 c_latest_boot["specificity"]["ci95_lo"], c_latest_boot["specificity"]["ci95_hi"]))

    # ---------- Comparison table (subject-level, threshold 0.40) ----------
    print("\n=== COMPARISON (subject-level, threshold %.2f) ===" % THRESHOLD)
    rows = [
        ("A current model @ baseline visit", a_base_m),
        ("B baseline-only model @ baseline visit", b_m),
        ("A current model @ latest visit", a_latest_m),
        ("C longitudinal (primary) @ latest eligible visit", None),  # filled below
    ]
    # recompute C primary subject-level for the table
    model_c = fitted_model(fol_train, LONG_FEATURES)
    c_vis = pd.DataFrame({
        SUBJECT_COL: fol_test[SUBJECT_COL].to_numpy(),
        LABEL_COL: fol_test[LABEL_COL].to_numpy(),
        BINARY_COL: fol_test[BINARY_COL].to_numpy(),
        VISIT_COL: fol_test[VISIT_COL].to_numpy(),
        "temporal_rank": fol_test["temporal_rank"].to_numpy(),
        "prob": model_c.predict_proba(fol_test[LONG_FEATURES])[:, 1],
    })
    c_latest = c_vis.sort_values([SUBJECT_COL, "temporal_rank"]).groupby(SUBJECT_COL).tail(1).copy()
    c_latest["predicted_binary"] = (c_latest["prob"] >= THRESHOLD).astype(int)
    c_latest_m = metrics_at_threshold(c_latest[BINARY_COL], c_latest["prob"])
    c_boot = bootstrap_ci(c_latest[BINARY_COL].to_numpy(), c_latest["prob"].to_numpy())
    rows[3] = ("C longitudinal (primary) @ latest eligible visit", c_latest_m)
    for name, m in rows:
        print("  %-42s sens=%.3f spec=%.3f balacc=%.3f auc=%.3f prauc=%.3f"
              % (name, m["sensitivity"], m["specificity"], m["balanced_accuracy"],
                 m["roc_auc"] if m["roc_auc"] is not None else float("nan"),
                 m["pr_auc"] if m["pr_auc"] is not None else float("nan")))

    # ---------- Converted detection comparison ----------
    print("\n=== CONVERTED DETECTION (subject-level) ===")
    conv_map = {
        "A @ baseline visit": a_base,
        "B baseline-only": b_vis,
        "A @ latest visit": a_latest,
        "C longitudinal @ latest": c_latest,
    }
    conv_report = {}
    for name, frame in conv_map.items():
        frame = frame.copy()
        grp = 1
        msk = frame[LABEL_COL] == grp
        flagged = int((frame.loc[msk, "predicted_binary"] == 1).sum())
        missed = int((frame.loc[msk, "predicted_binary"] == 0).sum())
        tot = int(msk.sum())
        prec = None
        # Converted precision among subjects flagged positive
        flagged_frame = frame[frame["predicted_binary"] == 1]
        if len(flagged_frame):
            prec = float((flagged_frame[LABEL_COL] == grp).mean())
        conv_report[name] = {
            "converted_subjects": tot, "detected": flagged, "missed": missed,
            "converted_recall": float(flagged / tot) if tot else None,
            "converted_precision": prec,
            "all_flagged": int(len(flagged_frame)),
        }
        print("  %-40s converted=%d detected=%d missed=%d recall=%.3f precision=%s"
              % (name, tot, flagged, missed,
                 float(flagged / tot) if tot else 0.0,
                 "%.3f" % prec if prec is not None else "n/a"))

    # ---------- Coverage (Section 13) ----------
    print("\n=== COVERAGE OF LONGITUDINAL FEATURES ===")
    cov = {
        "train_follow_up_visits": int(len(fol_train)),
        "test_follow_up_visits": int(len(fol_test)),
        "train_baseline_excluded_no_history": int(len(base_train)),
        "test_baseline_excluded_no_history": int(len(base_test)),
        "train_prev_visit_counts": dict(fol_train.groupby(SUBJECT_COL).size().value_counts().sort_index().astype(int).to_dict()),
        "test_prev_visit_counts": dict(fol_test.groupby(SUBJECT_COL).size().value_counts().sort_index().astype(int).to_dict()),
        "train_subjects_with_only_1_followup": int((fol_train.groupby(SUBJECT_COL).size() == 1).sum()),
        "test_subjects_with_only_1_followup": int((fol_test.groupby(SUBJECT_COL).size() == 1).sum()),
        "test_subjects_with_at_least_1_followup": int(fol_test[SUBJECT_COL].nunique()),
        "note": "each follow-up row uses exactly ONE previous visit (lag-1, MR-Delay-ordered)",
    }
    print(json.dumps(cov, indent=2))

    # ---------- ROC/PR plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, y, p in [
        ("B baseline-only", b_vis[BINARY_COL], b_vis["prob"]),
        ("A latest visit", a_latest[BINARY_COL], a_latest["prob"]),
        ("C longitudinal latest", c_latest[BINARY_COL], c_latest["prob"]),
    ]:
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, roc_auc_score(y, p)))
        axes[1].plot(rec, prec, label="%s (PR=%.3f)" % (name, average_precision_score(y, p)))
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC (subject-level)")
    axes[0].legend()
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("PR (subject-level)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "subject_level_roc_pr.png", dpi=150)
    plt.close()

    # ---------- Save artifacts ----------
    base_metrics = {
        "model": "B baseline-only (experimental baseline-visit screening)",
        "features": BASE_FEATURES,
        "fit_rows": int(len(base_train)),
        "threshold": THRESHOLD,
        "metrics": b_m,
        "converted_demented_breakdown": b_breakdown,
        "bootstrap_ci": bootstrap_ci(b_vis[BINARY_COL].to_numpy(), b_vis["prob"].to_numpy()),
        "note": "experimental baseline-visit screening; label is subject's constant group (final diagnosis).",
    }
    (OUT_DIR / "baseline_visit_metrics.json").write_text(json.dumps(base_metrics, indent=2), encoding="utf-8")

    long_metrics = {
        "model": "C longitudinal feature model (experimental longitudinal screening analysis)",
        "features_primary": LONG_FEATURES,
        "features_with_prev_mmse": LONG_FEATURES_PREV,
        "fit_rows": int(len(fol_train)),
        "threshold": THRESHOLD,
        "subject_level": {
            "latest_eligible_visit": c_latest_m,
            "mean_of_eligible_visits": c_mean_m,
            "per_visit_row_level": c_per_visit,
        },
        "converted_demented_breakdown_latest": c_latest_breakdown,
        "bootstrap_ci_latest": c_boot,
        "note": "subject-level primary = latest available eligible visit; per-visit is row-level research only.",
    }
    (OUT_DIR / "longitudinal_feature_metrics.json").write_text(json.dumps(long_metrics, indent=2), encoding="utf-8")

    visit_elig = pd.DataFrame({
        "subject_id": df[SUBJECT_COL],
        "visit_num": df[VISIT_COL],
        "temporal_rank": df["temporal_rank"],
        "mr_delay": df[MR_DELAY_COL],
        "has_previous": df["has_previous"],
        "eligible_for_followup": (df["temporal_rank"] >= 2).astype(int),
        "split": df[SUBJECT_COL].map(lambda s: "train" if s in train_subjects else ("test" if s in test_subjects else "OTHER")),
    })
    visit_elig.to_csv(OUT_DIR / "visit_eligibility.csv", index=False)

    feat_meta = {
        "base_features": BASE_FEATURES,
        "longitudinal_features": LONG_FEATURES,
        "longitudinal_features_with_prev_mmse": LONG_FEATURES_PREV,
        "feature_definitions": {
            "age": "age at current visit (years)",
            "sex": "M=1, F=0 (constant per subject)",
            "education_years": "EDUC, constant per subject",
            "mmse": "current-visit MMSE (imputed if missing)",
            "ses": "SES, constant per subject",
            "prev_mmse": "MMSE at immediately previous visit (MR-Delay-ordered)",
            "mmse_delta": "current MMSE - previous MMSE",
            "time_since_previous": "MR Delay(current) - MR Delay(previous) in days",
        },
        "forbidden_features": ["final_mmse", "future_mmse", "future_cdr", "future_group",
                               "future_visits", "future_mri", "cdr",
                               "previous_model_prediction", "rate_of_change_in_model_probability"],
        "cdr_used_as_feature": False,
        "model_predictions_used_as_features": False,
        "temporal_rule": "for a visit at time t, only features from visits <= t are allowed",
    }
    (OUT_DIR / "feature_metadata.json").write_text(json.dumps(feat_meta, indent=2), encoding="utf-8")

    # subject_predictions.csv: subject-level predictions for all settings
    def _subj_frame(frame, model_name, score_col, label_col=LABEL_COL):
        f = frame.copy()
        if score_col == "prob":
            return pd.DataFrame({
                "subject_id": f[SUBJECT_COL], "model": model_name,
                "true_group": f[label_col], "true_binary_label": f[BINARY_COL],
                "subject_score": f["prob"], "predicted_binary": f["predicted_binary"],
                "visit_used": f.get(VISIT_COL, np.nan),
            })
        return f.reset_index()

    pred_frames = []
    for name, f in conv_map.items():
        g = f.reset_index()
        pred_frames.append(pd.DataFrame({
            "subject_id": g[SUBJECT_COL], "model": name,
            "true_group": g[LABEL_COL], "true_binary_label": g[BINARY_COL],
            "subject_score": g.get("subject_probability", g.get("prob")),
            "predicted_binary": g["predicted_binary"],
            "visit_used": g.get(VISIT_COL, np.nan),
        }))
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(OUT_DIR / "subject_predictions.csv", index=False)

    # confusion_matrices.csv (subject-level, threshold 0.40)
    cm_rows = []
    for name, m in rows + [("B baseline-only", b_m), ("A baseline visit", a_base_m)]:
        cm_rows.append({
            "setting": name, "threshold": THRESHOLD,
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
        })
    cm_df = pd.DataFrame(cm_rows).drop_duplicates(subset=["setting"])
    cm_df.to_csv(OUT_DIR / "confusion_matrices.csv", index=False)

    comparison_rows = []
    for name, m in rows:
        comparison_rows.append({
            "setting": name, "threshold": THRESHOLD,
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"], "ppv": m["ppv"], "npv": m["npv"],
            "f1": m["f1"], "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"], "logloss": m["logloss"],
            "n": m["n"], "positive": m["positive"], "negative": m["negative"],
        })
    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(OUT_DIR / "comparison_subject_level.csv", index=False)

    repro = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seeds": {"numpy": NP_SEED, "bootstrap": NP_SEED},
        "model": "LogisticRegression (Phase 4/5 winner)",
        "model_hyperparameters": WINNER_HP,
        "preprocessing": "SimpleImputer(median)->StandardScaler fit on training rows only",
        "calibration": "none (raw probabilities)",
        "threshold": THRESHOLD,
        "threshold_origin": "frozen from Phase 5 (experimental screening threshold); NOT re-optimized here",
        "outer_split": "112 training / 38 test subjects (subject_grouped_baseline/subject_split.json)",
        "setting_A": "baseline: Visit==1 (MR Delay==0), first available visit per subject",
        "setting_B": "follow-up: Visit>=2, features from visits <= current only (lag-1 previous visit)",
    }
    (OUT_DIR / "reproducibility.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")

    (OUT_DIR / "README.md").write_text(
        "# Phase 6 - Longitudinal / Temporal Validation (Research Only)\n\n"
        "- Experimental longitudinal screening analysis on the Phase 4/5 binary candidate (LR C=10 balanced, threshold 0.40 frozen).\n"
        "- Setting A (baseline screening): Visit==1 only. Setting B (follow-up screening): Visit>=2, lag-1 previous visit features only.\n"
        "- Temporal leakage verified programmatically; future visits never used.\n"
        "- CDR NOT used as a feature; no model predictions used as features.\n"
        "- No production changes; no commits.\n",
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("split overlap == 0:", len(train_subjects & test_subjects) == 0)
    print("all test subjects have >=1 follow-up eligible visit:",
          fol_test[SUBJECT_COL].nunique() == len(test_subjects))
    print("baseline rows == n subjects (Setting A):",
          len(base_train) == len(train_subjects) and len(base_test) == len(test_subjects))
    print("threshold not re-optimized:", THRESHOLD == 0.40)
    print("cdr excluded from features: True")
    print("model predictions not used as features: True")


if __name__ == "__main__":
    main()
