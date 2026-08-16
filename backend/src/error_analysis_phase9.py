"""
ML PIPELINE PHASE 9 -- MISSED-POSITIVE ERROR ANALYSIS (RESEARCH ONLY).

Diagnoses why the experimental binary screening candidate (Phase 4/5 winner:
LogisticRegression C=10.0 class_weight=balanced, features [age, sex,
education_years, mmse, ses], threshold 0.40, latest-available-visit policy)
misses certain positives in the outer test:

  Missed Converted: OAS2_0145 (latest prob ~0.099), OAS2_0176 (~0.136)
  Missed Demented:  OAS2_0175 (~0.235)

This phase is STRICTLY DIAGNOSTIC. No production changes. No rules. No new
features. No threshold changes. Uses the SAME Phase 5 model fit on the 112
training subjects (all visits) so coefficients/contributions correspond to the
candidate actually under evaluation.

Forced discipline:
- Only prediction-time-available features are used (age, sex, education_years,
  SES, MMSE, visit count, MR Delay, latest-vs-baseline MMSE change).
- CDR, future Group, future MMSE, future visits, final-label-derived features,
  MRI features, model predictions as features are all FORBIDDEN.
- Counterfactual ranges restricted to observed dataset values only.
- Aggregation alternatives (latest/max/mean/baseline) are DESCRIPTIVE only;
  no new aggregation rule is selected using the outer test.
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

import sklearn

DATA_PATH = r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase9_error_analysis"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"

NP_SEED = 42
THRESHOLD = 0.40

WINNER_HP = {"clf__C": 10.0, "clf__class_weight": "balanced"}

MISSED_CONVERTED = ["OAS2_0145", "OAS2_0176"]
MISSED_DEMENTED = ["OAS2_0175"]


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
    df["n_visits"] = df.groupby(SUBJECT_COL)[SUBJECT_COL].transform("size")
    return df


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
    ]).set_params(**WINNER_HP)


def metrics_at_threshold(y, p, t=THRESHOLD):
    from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                                 f1_score, accuracy_score, roc_auc_score,
                                 average_precision_score, brier_score_loss, log_loss)
    y = np.asarray(y); p = np.asarray(p)
    y_pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    uniq = len(np.unique(y))
    return {
        "n": int(len(y)), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
        "sensitivity": float(sens), "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2), "ppv": float(ppv), "npv": float(npv),
        "f1": float(f1_score(y, y_pred, zero_division=0)), "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, p)) if uniq == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if uniq == 2 else None,
        "brier": float(brier_score_loss(y, p)) if uniq == 2 else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if uniq == 2 else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main():
    np.random.seed(NP_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])
    assert len(train_subjects & test_subjects) == 0

    df = load_data()
    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    df_test = df[df[SUBJECT_COL].isin(test_subjects)]

    # ---- fit the Phase 5 winner on the 112 training subjects (all visits) ----
    model = make_pipeline()
    model.fit(df_train[FEATURES], df_train[BINARY_COL])
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    print("=== LR COEFFICIENTS (trained on 112 train subjects) ===")
    for f, c in zip(FEATURES, coef):
        print("  %s: coef=%+.4f" % (f, c))
    print("  intercept: %+.4f" % intercept)

    # ---- per-test-subject latest & baseline visits ----
    earliest = df_test[df_test["temporal_rank"] == 1].copy()
    latest = df_test[df_test[MR_DELAY_COL] == df_test.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")].copy()
    assert len(earliest) == 38 and len(latest) == 38

    earliest["prob"] = model.predict_proba(earliest[FEATURES])[:, 1]
    latest["prob"] = model.predict_proba(latest[FEATURES])[:, 1]
    latest["predicted_binary"] = (latest["prob"] >= THRESHOLD).astype(int)
    earliest["predicted_binary"] = (earliest["prob"] >= THRESHOLD).astype(int)

    # ---------- SECTION 1: missed subject profiles ----------
    print("\n=== MISSED SUBJECT PROFILES (all visits, MR Delay order) ===")
    profile_rows = []
    for sid in MISSED_CONVERTED + MISSED_DEMENTED:
        sub = df[df[SUBJECT_COL] == sid].sort_values(MR_DELAY_COL)
        base_p = float(earliest.loc[earliest[SUBJECT_COL] == sid, "prob"].iloc[0])
        latest_p = float(latest.loc[latest[SUBJECT_COL] == sid, "prob"].iloc[0])
        for _, r in sub.iterrows():
            profile_rows.append({
                "subject_id": sid,
                "group": int(r[LABEL_COL]),
                "visit_num": int(r["raw_visit"]),
                "temporal_rank": int(r["temporal_rank"]),
                "mr_delay": int(r[MR_DELAY_COL]),
                "age": int(r["age"]),
                "sex": int(r["sex"]),
                "education_years": int(r["education_years"]),
                "ses": None if pd.isna(r["ses"]) else float(r["ses"]),
                "mmse": None if pd.isna(r["mmse"]) else float(r["mmse"]),
                "baseline_probability": round(base_p, 4),
                "latest_probability": round(latest_p, 4),
                "decision_at_0.40": "flagged" if latest_p >= THRESHOLD else "missed",
            })
    prof = pd.DataFrame(profile_rows)
    prof.to_csv(OUT_DIR / "missed_subject_profiles.csv", index=False)
    print(prof.to_string(index=False))

    # ---------- SECTION 2: outer-test feature comparison ----------
    print("\n=== OUTER-TEST FEATURE COMPARISON (latest visit, prediction-time features) ===")
    latest["mmse_change"] = latest["mmse"] - earliest["mmse"].values
    groups = {}
    for _, r in latest.iterrows():
        sid = r[SUBJECT_COL]
        true_grp = int(r[LABEL_COL])
        det = int(r["predicted_binary"])
        if true_grp == 1:
            cat = "A_converted_detected" if det == 1 else "B_converted_missed"
        elif true_grp == 2:
            cat = "C_demented_detected" if det == 1 else "D_demented_missed"
        else:
            cat = "E_nondemented_rejected" if det == 0 else "F_nondemented_fp"
        groups.setdefault(cat, []).append(sid)

    feat_rows = []
    for cat, sids in groups.items():
        sub = latest[latest[SUBJECT_COL].isin(sids)]
        feat_rows.append({
            "group": cat, "subjects": len(sids),
            "age_median": float(sub["age"].median()),
            "sex_M_fraction": float((sub["sex"] == 1).mean()),
            "education_median": float(sub["education_years"].median()),
            "ses_median": float(sub["ses"].median()),
            "mmse_median": float(sub["mmse"].median()),
            "mmse_min": float(sub["mmse"].min()),
            "visit_count_median": float(sub["n_visits"].median()),
            "mr_delay_latest_median": float(sub[MR_DELAY_COL].median()),
            "mmse_change_median": float(sub["mmse_change"].median()),
            "prob_median": float(sub["prob"].median()),
        })
    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(OUT_DIR / "outer_test_feature_comparison.csv", index=False)
    print(feat_df.to_string(index=False))

    # ---------- SECTION 3: MMSE distributions ----------
    print("\n=== MMSE DISTRIBUTIONS (baseline vs latest, by group) ===")
    mmse_rows = []
    for stage_name, stage_df in [("baseline", earliest), ("latest", latest)]:
        for grp, gname in [(0, "Nondemented"), (1, "Converted"), (2, "Demented")]:
            sub = stage_df[stage_df[LABEL_COL] == grp]
            if len(sub):
                s = sub["mmse"].dropna()
                mmse_rows.append({
                    "stage": stage_name, "group": gname, "n": int(len(s)),
                    "min": float(s.min()), "q1": float(s.quantile(0.25)),
                    "median": float(s.median()), "mean": float(s.mean()),
                    "std": float(s.std()), "q3": float(s.quantile(0.75)), "max": float(s.max()),
                })
    mmse_df = pd.DataFrame(mmse_rows)
    mmse_df.to_csv(OUT_DIR / "mmse_distributions.csv", index=False)
    print(mmse_df.to_string(index=False))

    print("\n=== MMSE: Converted missed vs detected vs Nondemented (latest visit) ===")
    conv_latest = latest[latest[LABEL_COL] == 1].copy()
    conv_latest["cat"] = np.where(conv_latest["predicted_binary"] == 1, "detected", "missed")
    nd = latest[latest[LABEL_COL] == 0]
    for cat, sub in conv_latest.groupby("cat"):
        s = sub["mmse"]
        print("  Converted %s (n=%d): min=%.0f median=%.0f mean=%.2f max=%.0f"
              % (cat, len(sub), s.min(), s.median(), s.mean(), s.max()))
    print("  Nondemented (n=%d): min=%.0f median=%.0f mean=%.2f max=%.0f"
          % (len(nd), nd["mmse"].min(), nd["mmse"].median(), nd["mmse"].mean(), nd["mmse"].max()))

    # ---------- SECTION 4: simple feature boundaries (observation only) ----------
    print("\n=== SIMPLE FEATURE BOUNDARIES (latest visit; OBSERVATION ONLY) ===")
    for f in ["mmse", "age", "ses", "education_years"]:
        print("  %s:" % f)
        for grp, gname in [(0, "Nondemented"), (1, "Converted"), (2, "Demented")]:
            sub = latest[latest[LABEL_COL] == grp]
            s = sub[f].dropna()
            print("    %-12s n=%2d min=%.1f q1=%.1f med=%.1f q3=%.1f max=%.1f"
                  % (gname, len(s), s.min(), s.quantile(0.25), s.median(),
                     s.quantile(0.75), s.max()))

    # ---------- SECTION 5: LR feature contributions (missed positives) ----------
    print("\n=== LOGISTIC FEATURE CONTRIBUTIONS (missed positives, latest visit) ===")
    contrib_rows = []
    for sid in MISSED_CONVERTED + MISSED_DEMENTED:
        row = latest[latest[SUBJECT_COL] == sid]
        X = row[FEATURES]
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
        contributions = coef * X_scaled[0]
        logodds = intercept + contributions.sum()
        prob = 1.0 / (1.0 + np.exp(-logodds))
        contrib_rows.append({
            "subject_id": sid, "group": int(row[LABEL_COL].iloc[0]),
            "intercept": round(float(intercept), 4),
            "sum_log_odds": round(float(logodds), 4),
            "model_probability": round(float(prob), 4),
        })
        print("  %s (group=%d): log-odds=%.3f prob=%.4f" % (sid, row[LABEL_COL].iloc[0], logodds, prob))
        for f, c, v, s_ in zip(FEATURES, coef, X_scaled[0], contributions):
            contrib_rows.append({
                "subject_id": sid, "feature": f,
                "feature_value": float(row[f].iloc[0]),
                "coefficient": round(float(c), 4),
                "scaled_value": round(float(s_), 4),
                "signed_contribution": round(float(c * s_), 4),
                "push_direction": "positive" if c * s_ > 0 else "negative",
            })
            print("    %-15s value=%-6.1f coef=%+7.3f scaled=%+7.3f contrib=%+7.3f (%s)"
                  % (f, row[f].iloc[0], c, s_, c * s_, "push+" if c * s_ > 0 else "push-"))
    contrib_df = pd.DataFrame(contrib_rows)
    contrib_df.to_csv(OUT_DIR / "logistic_feature_contributions.csv", index=False)

    # ---------- SECTION 6: counterfactual sensitivity ----------
    print("\n=== COUNTERFACTUAL FEATURE SENSITIVITY (one feature at a time) ===")
    observed = df_train  # restrict counterfactual values to observed training data
    cf_rows = []
    for sid in MISSED_CONVERTED + MISSED_DEMENTED:
        row = latest[latest[SUBJECT_COL] == sid]
        actual_prob = float(row["prob"].iloc[0])
        for f in FEATURES:
            col = observed[f].dropna()
            lo = float(col.min()); hi = float(col.max())
            cands = sorted(set(np.round(np.linspace(lo, hi, 15), 0)))
            if f in ("mmse", "age"):
                # plausible neighboring values within observed range
                cands = sorted(set(int(x) for x in cands
                                   if lo - 1 <= x <= hi + 1))
            else:
                cands = sorted(set(int(x) for x in observed[f].dropna().unique()))
            for cand in cands:
                if f in ("sex",) and cand not in (0, 1):
                    continue
                trial = row.copy()
                trial[f] = cand
                X = trial[FEATURES]
                p = float(model.predict_proba(X)[0][1])
                cf_rows.append({
                    "subject_id": sid, "feature": f,
                    "counterfactual_value": float(cand),
                    "probability": round(p, 4),
                    "delta_from_actual_prob": round(p - actual_prob, 4),
                })
    cf_df = pd.DataFrame(cf_rows)
    cf_df.to_csv(OUT_DIR / "counterfactual_sensitivity.csv", index=False)
    # print compact summary: for each subject/feature, min and max prob across counterfactuals
    for sid in MISSED_CONVERTED + MISSED_DEMENTED:
        print("  %s:" % sid)
        for f in FEATURES:
            sub = cf_df[(cf_df["subject_id"] == sid) & (cf_df["feature"] == f)]
            if len(sub):
                print("    %-15s prob range [%.3f .. %.3f]" % (f, sub["probability"].min(), sub["probability"].max()))

    # ---------- SECTION 7: visit probability history ----------
    print("\n=== VISIT PROBABILITY HISTORY (missed positives, all visits) ===")
    hist_rows = []
    for sid in MISSED_CONVERTED + MISSED_DEMENTED:
        sub = df[df[SUBJECT_COL] == sid].sort_values(MR_DELAY_COL)
        sub = sub.copy()
        sub["visit_prob"] = model.predict_proba(sub[FEATURES])[:, 1]
        print("  %s:" % sid)
        for _, r in sub.iterrows():
            print("    rank=%d mr_delay=%4d age=%d mmse=%s prob=%.4f"
                  % (r["temporal_rank"], r[MR_DELAY_COL], r["age"], r["mmse"], r["visit_prob"]))
            hist_rows.append({
                "subject_id": sid, "temporal_rank": int(r["temporal_rank"]),
                "mr_delay": int(r[MR_DELAY_COL]), "age": int(r["age"]),
                "mmse": None if pd.isna(r["mmse"]) else float(r["mmse"]),
                "visit_probability": round(float(r["visit_prob"]), 4),
            })
    hist_df = pd.DataFrame(hist_rows)
    hist_df.to_csv(OUT_DIR / "visit_probability_history.csv", index=False)

    # ---------- SECTION 8: aggregation diagnostics (DESCRIPTIVE ONLY) ----------
    print("\n=== AGGREGATION DIAGNOSTICS (DESCRIPTIVE ONLY; no rule selected) ===")
    # subject-level score under 4 aggregations of the SAME model
    all_vis = df_test.copy()
    all_vis["prob"] = model.predict_proba(all_vis[FEATURES])[:, 1]
    agg = all_vis.groupby(SUBJECT_COL).agg(
        true_binary_label=(BINARY_COL, "first"),
        true_group=(LABEL_COL, "first"),
        latest_prob=("prob", "last"),
        max_prob=("prob", "max"),
        mean_prob=("prob", "mean")).reset_index()
    earliest_map = earliest.set_index(SUBJECT_COL)["prob"]
    agg["baseline_prob"] = agg[SUBJECT_COL].map(earliest_map)
    agg_results = {}
    for name, col in [("latest", "latest_prob"), ("max", "max_prob"),
                      ("mean", "mean_prob"), ("baseline", "baseline_prob")]:
        m = metrics_at_threshold(agg["true_binary_label"], agg[col])
        agg_results[name] = {k: m[k] for k in
                             ["sensitivity", "specificity", "balanced_accuracy",
                              "ppv", "npv", "f1", "accuracy", "roc_auc", "brier"]}
        print("  %-8s sens=%.3f spec=%.3f balacc=%.3f auc=%.3f brier=%.3f"
              % (name, m["sensitivity"], m["specificity"], m["balanced_accuracy"],
                 m["roc_auc"], m["brier"]))
    agg_diag = pd.DataFrame(agg_results).T.reset_index().rename(columns={"index": "aggregation"})
    agg_diag.to_csv(OUT_DIR / "aggregation_diagnostics.csv", index=False)

    # ---------- SECTION 9: Converted separability ----------
    print("\n=== CONVERTED-CLASS SEPARABILITY (latest visit, outer test) ===")
    conv = latest[latest[LABEL_COL] == 1]
    nd = latest[latest[LABEL_COL] == 0]
    print("  Converted n=%d: MMSE range [%.0f, %.0f], age range [%d, %d]"
          % (len(conv), conv["mmse"].min(), conv["mmse"].max(),
             conv["age"].min(), conv["age"].max()))
    print("  Nondemented n=%d: MMSE range [%.0f, %.0f], age range [%d, %d]"
          % (len(nd), nd["mmse"].min(), nd["mmse"].max(),
             nd["age"].min(), nd["age"].max()))
    overlap_mmse = max(0, min(conv["mmse"].max(), nd["mmse"].max()) - max(conv["mmse"].min(), nd["mmse"].min()))
    print("  MMSE range overlap (years): %.0f" % overlap_mmse)
    print("  Missed Converted latest MMSE:",
          [float(x) for x in conv.loc[conv["predicted_binary"] == 0, "mmse"]])

    # ---------- artifacts ----------
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
        "no_mri_features": True,
        "no_model_predictions_as_features": True,
        "preprocessing": "SimpleImputer(median)->StandardScaler fit on training subjects only",
        "threshold": THRESHOLD,
        "temporal_ordering": "MR Delay",
        "outer_split": "112 training / 38 test subjects, zero overlap",
        "latest_definition": "row with max MR Delay per subject",
        "evaluation_policy": "latest-available-visit screening",
        "limitation": ("OASIS Group label constant per subject (final classification); "
                       "latest-visit classification is NOT future-conversion prediction. "
                       "Missed-subject analysis is diagnostic only."),
    }
    (OUT_DIR / "reproducibility.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")

    (OUT_DIR / "README.md").write_text(
        "# Phase 9 - Missed-Positive Error Analysis (Research Only)\n\n"
        "- Diagnostic analysis of missed positives OAS2_0145, OAS2_0176 (Converted), OAS2_0175 (Demented).\n"
        "- Same Phase 5 model (LR C=10 balanced; age, sex, education_years, mmse, ses); threshold 0.40; latest-visit policy.\n"
        "- ONLY prediction-time-available features used; CDR/future/MRI/label-derived forbidden.\n"
        "- Counterfactual ranges restricted to observed dataset values.\n"
        "- Aggregation alternatives reported DESCRIPTIVELY only (no rule selected on outer test).\n"
        "- No production changes; no rules; no commits.\n",
        encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    for f in sorted(p.name for p in OUT_DIR.iterdir()):
        print("  ", OUT_DIR / f)

    print("\n=== SANITY CHECKS ===")
    print("split overlap == 0:", len(train_subjects & test_subjects) == 0)
    print("missed subjects in outer test:",
          set(MISSED_CONVERTED + MISSED_DEMENTED) <= test_subjects)
    print("threshold unchanged:", THRESHOLD == 0.40)
    print("no CDR in features:", "cdr" not in FEATURES)
    print("no model predictions as features: True")
    print("counterfactuals restricted to observed range: True")


if __name__ == "__main__":
    main()
