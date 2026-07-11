"""
Clean 5-feature clinical model training
"""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


def load_data():
    base = Path(__file__).resolve().parents[1]
    df = pd.read_csv(r"D:\WebUI\cursor\alzheimers_ml_project\backend\data\raw\Dataset.csv")

    df = df.rename(columns={
        "Age": "age",
        "M/F": "sex",
        "EDUC": "education_years",
        "MMSE": "mmse",
        "CDR": "cdr",
        "SES": "ses",
        "Group": "group",
    })

    df["sex"] = df["sex"].map({"M": 1, "F": 0})
    df["group"] = df["group"].map({"Nondemented": 0, "Converted": 1, "Demented": 2})

    df = df[["age", "sex", "education_years", "mmse", "ses", "group"]]
    df = df.fillna(df.median())

    return df


def train():
    df = load_data()
    X = df.drop("group", axis=1)
    y = df["group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight={0: 1, 1: 4, 2: 2}
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    base = Path(__file__).resolve().parents[1]
    out = base / "models/best_model.pkl"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(model, out)
    print("\nSaved:", out)

    generate_shap_explanation(model, X_test, base)


def generate_shap_explanation(model, X_test, base):
    print("\nGenerating SHAP explanations...")

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_scaled_df)

    shap_dir = base / "models" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["Nondemented", "Converted", "Demented"]

    if isinstance(shap_values, list):
        shap_values_per_class = shap_values
    else:
        shap_values_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    plt.figure()
    shap.summary_plot(
        shap_values_per_class,
        X_test_scaled_df,
        class_names=class_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary.png", dpi=150)
    plt.close()

    for i, class_name in enumerate(class_names):
        plt.figure()
        shap.summary_plot(
            shap_values_per_class[i],
            X_test_scaled_df,
            show=False
        )
        plt.title(f"SHAP summary - {class_name}")
        plt.tight_layout()
        plt.savefig(shap_dir / f"shap_summary_{class_name.lower()}.png", dpi=150)
        plt.close()

    joblib.dump({
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": list(X_test.columns),
    }, shap_dir / "shap_data.pkl")

    print(f"SHAP summary plot saved to: {shap_dir / 'shap_summary.png'}")
    print(f"Per-class SHAP plots saved to: {shap_dir}")
    print(f"SHAP raw values saved to: {shap_dir / 'shap_data.pkl'}")


if __name__ == "__main__":
    train()
