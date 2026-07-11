"""
Clean 6-feature clinical model training
"""

from pathlib import Path
import joblib
import pandas as pd
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


if __name__ == "__main__":
    train()

