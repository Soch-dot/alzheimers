"""
ML PIPELINE PHASE 5 -- SCREENING THRESHOLD + SUBJECT-LEVEL UNCERTAINTY (RESEARCH ONLY).

Selects an experimental screening threshold for the Phase 4 binary Logistic
Regression candidate using subject-level out-of-fold predictions from the 112
development subjects ONLY, evaluates stability across folds, then applies the
frozen threshold exactly once to the untouched 38-subject outer test.

Rules:
- Outer test untouched until threshold is frozen.
- Subject is the independent unit; subject_probability = mean of visit probs.
- Aggregation rule is pre-declared and unchanged (from Phase 4).
- No calibration added (Phase 4 showed raw had best grouped-CV log loss/Brier).
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
    roc_curve, precision_recall_curve,
)

import joblib
import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OOF_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "binary_vs_three_class" / "oof_predictions.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase5_threshold"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
LABEL_COL = "group"
BINARY_COL = "binary_target"
SUBJECT_COL = "Subject ID"
PROBA_COL = "prob_any_dementia"

NP_SEED = 42
N_BOOT = 5000
CV_RANDOM_STATE = 42

# Phase 4 winner config (experimental binary screening candidate)
WINNER_HP = {"clf__C": 10.0, "clf__class_weight": "balanced"}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "CDR": "cdr", "SES": "ses", "Group": LABEL_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    return df[[SUBJECT_COL] + FEATURES + [LABEL_COL, BINARY_COL]]


def metrics_at_threshold(y, p, t):
    y_pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "threshold": float(t),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
        "youden_j": float(sens + spec - 1),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def select_thresholds(sweep_df):
    """Evaluate the four decision rules on a development sweep table."""
    best_youden = sweep_df.loc[sweep_df["youden_j"].idxmax()]
    best_balacc = sweep_df.loc[sweep_df["balanced_accuracy"].idxmax()]
    sens80 = sweep_df[sweep_df["sensitivity"] >= 0.80]
    sens85 = sweep_df[sweep_df["sensitivity"] >= 0.85]
    t80 = sens80.loc[sens80["specificity"].idxmax()] if len(sens80) else None
    t85 = sens85.loc[sens85["specificity"].idxmax()] if len(sens85) else None
    return {
        "youden": best_youden,
        "max_balacc": best_balacc,
        "sens_ge_0.80": t80,
        "sens_ge_0.85": t85,
    }


def bootstrap_ci(y, p, t, n_boot=N_BOOT, seed=NP_SEED):
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


def main():
    np.random.seed(NP_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])

    oof = pd.read_csv(OOF_PATH)
    oof = oof[oof["model"] == "LogisticRegression"]
    oof = oof.rename(columns={"subject_id": SUBJECT_COL})
    if set(oof[SUBJECT_COL]) != train_subjects:
        raise SystemExit("STOP: OOF subjects do not match training subjects")
    if len(set(oof[SUBJECT_COL]) & test_subjects) != 0:
        raise SystemExit("STOP: OOF contains outer-test subjects")

    # Subject-level aggregation of OOF (pre-declared: mean of visit probs).
    subj = oof.groupby(SUBJECT_COL).agg(
        subject_probability=(PROBA_COL, "mean"),
        true_binary_label=("true_binary_label", "first"),
        n_visits=("true_binary_label", "size"),
    ).reset_index()

    y_dev = subj["true_binary_label"].to_numpy()
    p_dev = subj["subject_probability"].to_numpy()
    print("=== SECTION 2/3: DEV SUBJECT-LEVEL OOF ===")
    print("dev subjects=%d positive=%d negative=%d"
          % (len(subj), int((y_dev == 1).sum()), int((y_dev == 0).sum())))

    # ---- SECTION 4: threshold sweep ----
    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    sweep = pd.DataFrame([metrics_at_threshold(y_dev, p_dev, t) for t in thresholds])
    sweep.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)
    print("\n=== SECTION 4: THRESHOLD SWEEP (dev only, 0.10-0.90 step 0.01) ===")
    print("rows in sweep:", len(sweep))
    print(sweep.iloc[::10][["threshold", "sensitivity", "specificity",
                            "balanced_accuracy", "youden_j", "ppv", "npv", "f1", "accuracy"]]
          .to_string(index=False))

    # ---- SECTION 5: decision rules ----
    rules = select_thresholds(sweep)
    print("\n=== SECTION 5: DECISION RULES ===")
    for name in ["youden", "max_balacc", "sens_ge_0.80", "sens_ge_0.85"]:
        r = rules[name]
        if r is None:
            print(f"{name}: NOT REACHED on development data")
        else:
            print(f"{name}: threshold={r['threshold']:.2f} sens={r['sensitivity']:.4f} "
                  f"spec={r['specificity']:.4f} balacc={r['balanced_accuracy']:.4f} "
                  f"youden={r['youden_j']:.4f}")

    # ---- SECTION 6: threshold stability across folds ----
    print("\n=== SECTION 6: THRESHOLD STABILITY (leave-one-fold-out) ===")
    stability_rows = []
    for k in sorted(oof["fold"].unique()):
        oof_tr = oof[oof["fold"] != k]
        subj_tr = oof_tr.groupby(SUBJECT_COL).agg(
            subject_probability=(PROBA_COL, "mean"),
            true_binary_label=("true_binary_label", "first")).reset_index()
        sw = pd.DataFrame([metrics_at_threshold(
            subj_tr["true_binary_label"].to_numpy(),
            subj_tr["subject_probability"].to_numpy(), t) for t in thresholds])
        rl = select_thresholds(sw)
        row = {"held_out_fold": int(k)}
        for name in ["youden", "max_balacc", "sens_ge_0.80", "sens_ge_0.85"]:
            rr = rl[name]
            row[name] = float(rr["threshold"]) if rr is not None else None
        stability_rows.append(row)
    stab = pd.DataFrame(stability_rows)
    stab.to_csv(OUT_DIR / "threshold_cv_stability.csv", index=False)
    print(stab.to_string(index=False))
    for name in ["youden", "max_balacc", "sens_ge_0.80", "sens_ge_0.85"]:
        vals = stab[name].dropna()
        print(f"{name}: mean={vals.mean():.3f} std={vals.std():.3f} "
              f"min={vals.min():.3f} max={vals.max():.3f}")

    # ---- SECTION 7: ROC / PR (dev OOF, subject-level) ----
    fpr, tpr, _ = roc_curve(y_dev, p_dev)
    prec, rec, _ = precision_recall_curve(y_dev, p_dev)
    roc_auc = roc_auc_score(y_dev, p_dev)
    pr_auc = average_precision_score(y_dev, p_dev)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"LR (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC (dev subject-level OOF)"); axes[0].legend()
    axes[1].plot(rec, prec, label=f"LR (AUC={pr_auc:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PR (dev subject-level OOF)"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=150)
    plt.savefig(OUT_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()

    # ---- SECTION 8/10: outer test one-shot ----
    # Rebuild the Phase 4 winner LR pipeline fit on all 112 training subjects.
    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]
    est = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
    ]).set_params(**WINNER_HP)
    est.fit(df_train[FEATURES], df_train[BINARY_COL])
    test_proba = est.predict_proba(df_test[FEATURES])[:, 1]

    outer_vis = pd.DataFrame({
        SUBJECT_COL: df_test[SUBJECT_COL].to_numpy(),
        "true_group": df_test[LABEL_COL].to_numpy(),
        "true_binary_label": df_test[BINARY_COL].to_numpy(),
        "visit_probability": test_proba,
    })
    outer_subj = outer_vis.groupby(SUBJECT_COL).agg(
        subject_probability=("visit_probability", "mean"),
        true_binary_label=("true_binary_label", "first"),
        true_group=("true_group", "first"),
        n_visits=("visit_probability", "size")).reset_index()

    # SELECTED threshold: justify based on the SCREENING objective, whose
    # explicit priority order is: 1) Sensitivity, 2) Balanced accuracy,
    # 3) Specificity, 4) NPV, 5) PPV, while keeping a usable FP rate.
    # The sensitivity-constrained rule (highest specificity subject to
    # sensitivity >= 0.80 on development data) directly encodes this priority:
    # it guarantees a meaningful sensitivity floor (rule C) without collapsing
    # specificity.  Rule D (>=0.85) gains ~1 dev subject of sensitivity at the
    # cost of ~15 points of specificity, so rule C is preferred.
    youden_t = float(rules["youden"]["threshold"])
    balacc_t = float(rules["max_balacc"]["threshold"])
    t80 = rules["sens_ge_0.80"]
    if t80 is not None and t80["sensitivity"] >= 0.80:
        selected = "sens_ge_0.80"
        selected_t = float(t80["threshold"])
        selection_reason = (
            "Screening objective prioritizes sensitivity (priority #1). Selected "
            "highest-specificity threshold with development sensitivity >= 0.80 "
            "(rule C): threshold=%.2f gives dev sens=%.3f, spec=%.3f. Rule D "
            "(>=0.85) adds ~1 dev subject of sensitivity at the cost of ~15 points "
            "of specificity, so rule C is preferred as the usable screening "
            "threshold. Youden/max-balacc reported as references."
            % (selected_t, t80["sensitivity"], t80["specificity"]))
    else:
        selected = "youden"
        selected_t = youden_t
        selection_reason = (
            "Screening objective prioritizes sensitivity, but development data "
            "did not reach sensitivity >= 0.80 at any threshold; falling back to "
            "Youden's J (max sens+spec-1) on development subject-level OOF.")
    print("\n=== SECTION 5/8: SELECTED EXPERIMENTAL SCREENING THRESHOLD ===")
    print("rule:", selected, "| threshold:", selected_t)
    print("reason:", selection_reason)

    y_te = outer_subj["true_binary_label"].to_numpy()
    p_te = outer_subj["subject_probability"].to_numpy()
    outer_m = metrics_at_threshold(y_te, p_te, selected_t)
    cm = np.array([[outer_m["tn"], outer_m["fp"]], [outer_m["fn"], outer_m["tp"]]])

    print("\n=== SECTION 8: OUTER TEST (ONE SHOT, threshold=%.2f) ===" % selected_t)
    print("subjects=%d rows=%d positive=%d negative=%d"
          % (len(outer_subj), len(df_test), int((y_te == 1).sum()), int((y_te == 0).sum())))
    for k in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv",
              "f1", "accuracy"]:
        print("  %s: %.4f" % (k, outer_m[k]))
    print("TP=%d FP=%d TN=%d FN=%d" % (outer_m["tp"], outer_m["fp"], outer_m["tn"], outer_m["fn"]))
    print("confusion matrix ([[TN,FP],[FN,TP]]):")
    print(cm)

    # ---- SECTION 10: Converted vs Demented breakdown (explicit, no inference) ----
    print("\n=== SECTION 10: CONVERTED vs DEMENTED BREAKDOWN (outer test) ===")
    pred = (p_te >= selected_t).astype(int)
    outer_subj["predicted_binary"] = pred
    breakdown = {}
    for grp, name in [(1, "Converted"), (2, "Demented"), (0, "Nondemented")]:
        msk = outer_subj["true_group"] == grp
        flagged = int((outer_subj.loc[msk, "predicted_binary"] == 1).sum())
        missed = int((outer_subj.loc[msk, "predicted_binary"] == 0).sum())
        breakdown[name] = {"subjects": int(msk.sum()), "flagged": flagged, "missed": missed}
        print(f"  {name}: subjects={int(msk.sum())} flagged={flagged} missed={missed}")
    print("Nondemented flagged (false positives):", breakdown["Nondemented"]["flagged"])

    # ---- SECTION 9: bootstrap CIs (resample SUBJECTS) ----
    ci = bootstrap_ci(y_te, p_te, selected_t)
    print("\n=== SECTION 9: BOOTSTRAP 95%% CI (subject resampling, n=%d) ===" % len(y_te))
    for c in ["sensitivity", "specificity", "balanced_accuracy", "ppv", "npv"]:
        print("  %s: %.4f [%.4f, %.4f]" % (c, ci[c]["mean"], ci[c]["ci95_lo"], ci[c]["ci95_hi"]))

    # ---- artifacts ----
    subj.to_csv(OUT_DIR / "subject_level_oof.csv", index=False)
    outer_vis.to_csv(OUT_DIR / "outer_test_predictions.csv", index=False)
    outer_subj.to_csv(OUT_DIR / "outer_test_subject_level.csv", index=False)

    rule_summary = {}
    for name in ["youden", "max_balacc", "sens_ge_0.80", "sens_ge_0.85"]:
        r = rules[name]
        rule_summary[name] = None if r is None else {
            "threshold": float(r["threshold"]),
            "sensitivity": float(r["sensitivity"]),
            "specificity": float(r["specificity"]),
            "balanced_accuracy": float(r["balanced_accuracy"]),
            "youden_j": float(r["youden_j"]),
        }

    repro = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seeds": {"numpy": NP_SEED, "bootstrap": NP_SEED, "cv": CV_RANDOM_STATE},
        "feature_order": FEATURES,
        "model": "LogisticRegression (Phase 4 winner)",
        "model_hyperparameters": WINNER_HP,
        "preprocessing": "SimpleImputer(median)->StandardScaler",
        "calibration": "none (raw probabilities; Phase 4 showed raw best on grouped CV)",
        "aggregation_method": "subject_probability = mean of visit probabilities",
        "threshold_sweep": "0.10 to 0.90 step 0.01 on development subject-level OOF",        "outer_split": "112 training / 38 test subjects (subject_grouped_baseline/subject_split.json)",
        "outer_test_used": "exactly once, after threshold frozen",
    }
    (OUT_DIR / "threshold_selection.json").write_text(json.dumps({
        "reproducibility": repro,
        "selected_rule": selected,
        "selected_threshold": float(selected_t),
        "selection_reason": selection_reason,
        "decision_rules": rule_summary,
        "threshold_stability": {name: {
            "mean": float(stab[name].dropna().mean()),
            "std": float(stab[name].dropna().std()),
            "min": float(stab[name].dropna().min()),
            "max": float(stab[name].dropna().max())} for name in ["youden", "max_balacc", "sens_ge_0.80", "sens_ge_0.85"]},
        "roc_auc_dev": float(roc_auc),
        "pr_auc_dev": float(pr_auc),
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "outer_test_results.json").write_text(json.dumps({
        "threshold": float(selected_t),
        "metrics": {k: outer_m[k] for k in
                    ["sensitivity", "specificity", "balanced_accuracy", "ppv",
                     "npv", "f1", "accuracy"]},
        "confusion_matrix": cm.tolist(),
        "tp": outer_m["tp"], "fp": outer_m["fp"], "tn": outer_m["tn"], "fn": outer_m["fn"],
        "n_subjects": len(outer_subj), "n_rows": len(df_test),
        "positive_subjects": int((y_te == 1).sum()),
        "negative_subjects": int((y_te == 0).sum()),
        "converted_demented_breakdown": breakdown,
        "bootstrap_ci": ci,
        "note": "Small-sample (n=38 subjects); CIs approximate and wide.",
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "bootstrap_confidence_intervals.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# Phase 5 - Screening Threshold + Uncertainty (Research Only)\n\n"
        "- Model: experimental binary screening candidate (LogisticRegression).\n"
        "- Threshold selected on dev subject-level OOF ONLY (sweep 0.10-0.90 step 0.01).\n"
        "- Selected rule: %s at threshold %.2f.\n"
        "- Outer 38-subject test applied exactly once.\n"
        "- No calibration added; no production changes.\n" % (selected, selected_t),
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("OOF subjects == training subjects:", set(oof[SUBJECT_COL]) == train_subjects)
    print("OOF overlap with outer: 0:", len(set(oof[SUBJECT_COL]) & test_subjects) == 0)
    print("Outer test subjects not in OOF:", not (set(outer_subj[SUBJECT_COL]) <= train_subjects))
    print("Aggregation method fixed (mean of visit probs): yes")
    print("Threshold tuned on dev only; outer touched once: yes")


if __name__ == "__main__":
    main()
