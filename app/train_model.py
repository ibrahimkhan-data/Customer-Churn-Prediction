"""
One-time training script.
Creates: models/churn_model.pkl, models/scaler.pkl, models/model_columns.pkl
Run:  python train_model.py
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "Telco-Customer-Churn.csv")
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
MODELS_DIR = os.path.join(BASE, "models")
NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Using local dataset: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("Dataset not found — downloading from GitHub...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)          # save a local copy for next time
    print(f"Saved a copy to: {DATA_PATH}")
    return df


def main():
    df = load_data()
    os.makedirs(MODELS_DIR, exist_ok=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    y = (df["Churn"] == "Yes").astype(int)
    X_raw = df.drop(columns=["customerID", "Churn"])

    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print(f"Samples  : {len(df)}   Features: {X.shape[1]}")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, proba):.4f}")
    print(classification_report(y_test, preds, target_names=["Stay", "Churn"]))

    joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "model_columns.pkl"))
    print(f"\nDone! Model files saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()