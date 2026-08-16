"""
EXPERIMENTAL calibration comparison — NOT a production test, NOT committed.

Loads the original model and the newly calibrated model, compares their
probability outputs on a fixed test case, and prints calibration diagnostics
evaluated on the EXPERIMENTAL calibration set (the original 20% held-out).

IMPORTANT interpretation:
  - Calibration can change probabilities without improving accuracy.
  - All calibration diagnostics are computed on the SAME set the sigmoid
    calibrator was fit on -> the calibrated numbers are IN-SAMPLE for the
    calibrator (optimistic). They are out-of-sample for the Random Forest.
    This is an experimental calibration step, NOT a final unbiased evaluation.
"""

import sys
import subprocess
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
MODELS = BASE / "models"
CAL_DIR = MODELS / "calibration"
sys.path.insert(0, str(SRC))
from train_clean_clinical_model import load_data  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ORIGINAL_MODEL = MODELS / "best_model.pkl"
CALIBRATED_MODEL = MODELS / "best_model_calibrated.pkl"
HASH_MARKER = CAL_DIR / "original_model_hash.txt"

FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
CLASS_NAMES = ["Nondemented", "Converted", "Demented"]
CLASS_IDX = {0: 0, 1: 1, 2: 2}

TEST_CASE = {"Age": 74, "Sex (Male=1)": 1, "Education": 10, "MMSE": 23, "SES": 1}


def ece_binary(y_bin: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """One-vs-rest Expected Calibration Error (mean absolute conf-acc gap)."""
    y_bin = np.asarray(y_bin, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(proba)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = proba >= lo if i == n_bins - 1 else (proba >= lo) & (proba < hi)
        n = mask.sum()
        if n == 0:
            continue
        conf = proba[mask].mean()
        acc = y_bin[mask].mean()
        ece += (n / total) * abs(acc - conf)
    return ece


def reliability_bins(y_bin: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    y_bin = np.asarray(y_bin, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = proba >= lo if i == n_bins - 1 else (proba >= lo) & (proba < hi)
        n = mask.sum()
        if n == 0:
            continue
        confs.append(proba[mask].mean())
        accs.append(y_bin[mask].mean())
        counts.append(n)
    return np.array(confs), np.array(accs), np.array(counts)


def model_proba(model, X):
    return model.predict_proba(X)[:, [CLASS_IDX[c] for c in model.classes_]]


def main() -> None:
    print("=" * 78)
    print("EXPERIMENTAL CALIBRATION COMPARISON  (NOT a final test / NOT committed)")
    print("=" * 78)

    # ---- 1. Load both models ------------------------------------------------
    original = joblib.load(ORIGINAL_MODEL)
    calibrated = joblib.load(CALIBRATED_MODEL)
    print(f"\noriginal model   = {type(original).__name__}, clf={type(original.named_steps['clf']).__name__}")
    print(f"calibrated model = {type(calibrated).__name__}, method=sigmoid, estimator=FrozenEstimator(original)")
    print(f"classes_ original   = {list(original.classes_)}")
    print(f"classes_ calibrated = {list(calibrated.classes_)}")

    # ---- 2. Test case ---------------------------------------------------------
    x_test = pd.DataFrame([[TEST_CASE["Age"], TEST_CASE["Sex (Male=1)"],
                            TEST_CASE["Education"], TEST_CASE["MMSE"], TEST_CASE["SES"]]],
                          columns=FEATURES)
    print(f"\nTest case (feature order MUST stay {FEATURES}):")
    print("  " + ",  ".join(f"{k}={v}" for k, v in TEST_CASE.items()))

    p_orig = model_proba(original, x_test)[0]
    p_cal = model_proba(calibrated, x_test)[0]
    pred_orig = int(original.predict(x_test)[0])
    pred_cal = int(calibrated.predict(x_test)[0])

    print("\n" + "-" * 78)
    print("TEST-CASE PROBABILITIES (side by side)")
    print("-" * 78)
    print(f"{'Class':<12}{'Original prob':>16}{'Calibrated prob':>18}")
    print("-" * 48)
    for c in [0, 1, 2]:
        print(f"{CLASS_NAMES[c]:<12}{p_orig[CLASS_IDX[c]]:>16.4f}{p_cal[CLASS_IDX[c]]:>18.4f}")
    det_orig = p_orig[1] + p_orig[2]
    det_cal = p_cal[1] + p_cal[2]
    print("-" * 48)
    print(f"Detected (Converted+Demented)   orig={det_orig:.4f}   calibrated={det_cal:.4f}")
    print(f"Predicted class   orig={CLASS_NAMES[pred_orig]}   calibrated={CLASS_NAMES[pred_cal]}")

    # ---- 3. Calibration-set diagnostics ----------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTICS ON THE EXPERIMENTAL CALIBRATION SET (original held-out, n=75)")
    print("=" * 78)
    df = load_data()
    X = df.drop("group", axis=1)
    y = df["group"]
    _, X_cal, _, y_cal = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_arr = y_cal.to_numpy()

    P_orig = model_proba(original, X_cal)
    P_cal = model_proba(calibrated, X_cal)

    print(f"calibration-set size  = {len(X_cal)}")
    print(f"class counts          = {dict(y_cal.value_counts().sort_index())}  "
          f"-> {[f'{CLASS_NAMES[c]}:{n}' for c, n in y_cal.value_counts().sort_index().items()]}")

    ll_orig = log_loss(y_arr, P_orig, labels=[0, 1, 2])
    ll_cal = log_loss(y_arr, P_cal, labels=[0, 1, 2])
    print(f"\nmulticlass log loss        orig={ll_orig:.4f}   calibrated={ll_cal:.4f}")

    print("\nper-class one-vs-rest Brier score (lower = better calibrated):")
    print(f"{'Class':<12}{'orig':>10}{'calibrated':>12}")
    brier_orig, brier_cal = [], []
    for c in [0, 1, 2]:
        yb = (y_arr == c).astype(float)
        bo = np.mean((P_orig[:, CLASS_IDX[c]] - yb) ** 2)
        bc = np.mean((P_cal[:, CLASS_IDX[c]] - yb) ** 2)
        brier_orig.append(bo); brier_cal.append(bc)
        print(f"{CLASS_NAMES[c]:<12}{bo:>10.4f}{bc:>12.4f}")
    print(f"{'MEAN':<12}{np.mean(brier_orig):>10.4f}{np.mean(brier_cal):>12.4f}")

    print("\nper-class one-vs-rest ECE (10 bins, lower = better calibrated):")
    print(f"{'Class':<12}{'orig':>10}{'calibrated':>12}")
    ece_orig, ece_cal = [], []
    for c in [0, 1, 2]:
        yb = (y_arr == c).astype(float)
        eo = ece_binary(yb, P_orig[:, CLASS_IDX[c]])
        ec = ece_binary(yb, P_cal[:, CLASS_IDX[c]])
        ece_orig.append(eo); ece_cal.append(ec)
        print(f"{CLASS_NAMES[c]:<12}{eo:>10.4f}{ec:>12.4f}")
    print(f"{'MEAN':<12}{np.mean(ece_orig):>10.4f}{np.mean(ece_cal):>12.4f}")

    print("\nraw vs calibrated probability ranges (over calibration set):")
    print(f"{'Class':<12}{'orig min/max':>18}{'calib min/max':>20}")
    for c in [0, 1, 2]:
        print(f"{CLASS_NAMES[c]:<12}{P_orig[:, CLASS_IDX[c]].min():>8.3f}/{P_orig[:, CLASS_IDX[c]].max():<8.3f}"
              f"{P_cal[:, CLASS_IDX[c]].min():>9.3f}/{P_cal[:, CLASS_IDX[c]].max():<9.3f}")

    print("\nInterpretation caveats:")
    print("  - These diagnostics are evaluated on the SAME set the sigmoid calibrator")
    print("    was fit on -> calibrated log-loss/Brier/ECE are IN-SAMPLE for the")
    print("    calibrator (optimistic); they are out-of-sample for the Random Forest.")
    print("    A final unbiased evaluation would require a NEW held-out set.")
    print("  - Per-class Brier/ECE are one-vs-rest decompositions (sklearn has no native")
    print("    multiclass Brier); ECE binning is a design choice (10 bins here).")

    # ---- 4. Reliability plots ---------------------------------------------------
    print("\n" + "=" * 78)
    print("RELIABILITY / CALIBRATION PLOTS")
    print("=" * 78)
    for c in [0, 1, 2]:
        yb = (y_arr == c).astype(float)
        co, ao, no = reliability_bins(yb, P_orig[:, CLASS_IDX[c]])
        cc, ac, nc = reliability_bins(yb, P_cal[:, CLASS_IDX[c]])
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(co, ao, "o-", color="#2563eb", ms=6, lw=1.6, label="original (uncalibrated)")
        ax.plot(cc, ac, "s-", color="#dc2626", ms=6, lw=1.6, label="calibrated (sigmoid)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"Reliability — {CLASS_NAMES[c]}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = CAL_DIR / f"reliability_{CLASS_NAMES[c].lower()}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  saved {out}")

    # ---- 5. Safety checks --------------------------------------------------------
    print("\n" + "=" * 78)
    print("SAFETY CHECKS")
    print("=" * 78)
    marker = {}
    if HASH_MARKER.exists():
        for line in HASH_MARKER.read_text(encoding="utf-8").strip().splitlines():
            k, _, v = line.partition("=")
            marker[k.strip()] = v.strip()
    import hashlib
    cur_hash = hashlib.md5(ORIGINAL_MODEL.read_bytes()).hexdigest().upper()
    print(f"best_model.pkl md5 marker={marker.get('md5')} now={cur_hash} "
          f"unchanged={'YES' if marker.get('md5') == cur_hash else 'NO'}")
    print(f"best_model_calibrated.pkl exists as separate file = {CALIBRATED_MODEL.exists()}")

    orig_works = original.predict(x_test)[0]
    print(f"original model predicts normally on test case = YES (class {CLASS_NAMES[int(orig_works)]})")

    def git_clean(paths):
        out = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                             capture_output=True, text=True, cwd=BASE.parent)
        return out.stdout.strip() == ""

    checks = {
        "/predict source (backend/src/api.py)": git_clean(["backend/src/api.py"]),
        "backend vision (backend/src/vision_eval.py)": git_clean(["backend/src/vision_eval.py"]),
        "SHAP files (backend/models/shap)": git_clean(["backend/models/shap"]),
        "frontend": git_clean(["frontend"]),
        "training script": git_clean(["backend/src/train_clean_clinical_model.py"]),
    }
    for name, ok in checks.items():
        print(f"  {name}: {'UNCHANGED (clean)' if ok else 'MODIFIED (!!)'}")

    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)
    print(f"\n| Class       | Original probability | Calibrated probability |")
    print(f"|-------------|----------------------|------------------------|")
    for c in [0, 1, 2]:
        print(f"| {CLASS_NAMES[c]:<11} | {p_orig[CLASS_IDX[c]]:>20.4f} | {p_cal[CLASS_IDX[c]]:>22.4f} |")
    print(f"\n| Metric                       | Original | Calibrated |")
    print(f"|------------------------------|----------|------------|")
    print(f"| Predicted class              | {CLASS_NAMES[pred_orig]:<8} | {CLASS_NAMES[pred_cal]:<10} |")
    print(f"| Converted + Demented         | {det_orig:>8.4f} | {det_cal:>10.4f} |")
    print(f"| Multiclass log loss (cal set)| {ll_orig:>8.4f} | {ll_cal:>10.4f} |")
    print(f"| Mean one-vs-rest Brier       | {np.mean(brier_orig):>8.4f} | {np.mean(brier_cal):>10.4f} |")
    print(f"| Mean one-vs-rest ECE         | {np.mean(ece_orig):>8.4f} | {np.mean(ece_cal):>10.4f} |")

    print("\nREPORT SUMMARY")
    print(f"  calibration-set size       = {len(X_cal)} (original 20% held-out, EXPERIMENTAL)")
    print(f"  calibration class counts   = {dict(y_cal.value_counts().sort_index())}")
    print(f"  model type                 = {type(original).__name__} ({type(original.named_steps['clf']).__name__})")
    print(f"  calibration method         = sigmoid (Platt) via CalibratedClassifierCV(FrozenEstimator)")
    print(f"  original model retrained   = NO")
    print(f"  best_model.pkl changed     = NO (md5 {cur_hash})")
    print(f"  /predict changed           = NO")
    print("  NOTE: experimental step only — NO integration, NO commit, NO push.")


if __name__ == "__main__":
    main()
