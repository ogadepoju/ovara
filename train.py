"""
train.py — Run this ONCE locally to generate trained_model.pkl
Trains a Logistic Regression model (best performer on corrected pipeline)
with StandardScaler, SimpleImputer, and threshold=0.40 for deployment.

Usage:
    python train.py

Requires:
    pip install -r requirements.txt
    KAGGLE_USERNAME and KAGGLE_KEY environment variables (or kaggle.json)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

MODEL_SAVE_PATH = "trained_model.pkl"

def run():
    import kagglehub
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import f1_score, classification_report
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import RandomOverSampler

    print("⏳ Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("shreyasvedpathak/pcos-dataset")
    df   = pd.read_csv(f"{path}/PCOS_data.csv")

    print("🧹 Removing clinical laboratory features...")
    df.drop(columns=[
        "Hb(g/dl)", "  I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
        "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "TSH (mIU/L)", "AMH(ng/mL)",
        "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)", "RBS(mg/dl)",
        "Follicle No. (L)", "Follicle No. (R)", "Avg. F size (L) (mm)",
        "Avg. F size (R) (mm)", "Endometrium (mm)", "Unnamed: 44"],
        errors="ignore", inplace=True)

    # Impute missing values
    df["Marraige Status (Yrs)"].fillna(df["Marraige Status (Yrs)"].mean(), inplace=True)
    df["Fast food (Y/N)"].fillna(df["Fast food (Y/N)"].mode()[0], inplace=True)

    # Correlation-based feature selection
    corr          = df.corrwith(df["PCOS (Y/N)"]).abs().sort_values(ascending=False)
    selected      = corr[corr > 0.1].index.tolist()
    data          = df[selected]
    feature_names = [c for c in selected if c != "PCOS (Y/N)"]

    print(f"✅ {len(feature_names)} features selected: {feature_names}")

    X = data[feature_names]
    y = data["PCOS (Y/N)"]

    # Corrected pipeline: split FIRST, then oversample training data only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

    # Imputer fitted on training data only
    imputer  = SimpleImputer(strategy="median")
    X_res    = pd.DataFrame(imputer.fit_transform(X_res),   columns=feature_names)
    X_test_i = pd.DataFrame(imputer.transform(X_test),      columns=feature_names)

    # StandardScaler fitted on training data only — required for Logistic Regression
    scaler      = StandardScaler()
    X_res_sc    = scaler.fit_transform(X_res)
    X_test_sc   = scaler.transform(X_test_i)

    print("🧠 Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_res_sc, y_res)

    # Evaluate
    preds = model.predict(X_test_sc)
    f1    = f1_score(y_test, preds, average="weighted")
    cv_f1 = cross_val_score(model, X_res_sc, y_res, cv=5, scoring="f1_weighted").mean()

    print(f"\n📊 Test F1:        {f1:.4f}")
    print(f"📊 5-Fold CV F1:   {cv_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["No PCOS", "PCOS"]))

    # Decision threshold = 0.40 (improves recall for PCOS-positive class)
    threshold = 0.40
    probs     = model.predict_proba(X_test_sc)[:, 1]
    preds_t   = (probs >= threshold).astype(int)
    f1_t      = f1_score(y_test, preds_t, average="weighted")
    print(f"📊 F1 at threshold={threshold}: {f1_t:.4f}")
    print(f"\nClassification Report at threshold={threshold}:")
    print(classification_report(y_test, preds_t, target_names=["No PCOS", "PCOS"]))

    bundle = {
        "model":          model,
        "model_name":     "Logistic Regression",
        "imputer":        imputer,
        "scaler":         scaler,
        "feature_names":  feature_names,
        "threshold":      threshold,
        "version":        1,
        "last_retrain":   datetime.utcnow().isoformat(),
    }
    joblib.dump(bundle, MODEL_SAVE_PATH)

    size_mb = __import__("os").path.getsize(MODEL_SAVE_PATH) / 1024 / 1024
    print(f"\n✅ Saved to {MODEL_SAVE_PATH} ({size_mb:.1f} MB)")
    print(f"\nNext steps:")
    print(f"  1. git add {MODEL_SAVE_PATH}")
    print(f"  2. git commit -m 'upgrade to Logistic Regression model'")
    print(f"  3. git push")

if __name__ == "__main__":
    run()
