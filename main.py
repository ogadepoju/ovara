"""
Obyrin — PCOS Home Risk Screening
Production server. Loads a pre-trained Logistic Regression model
from trained_model.pkl at startup. No training, no feedback collection.

Serves:  GET /             → consumer questionnaire
API:     POST /api/predict → risk prediction (threshold=0.40)
         GET  /api/info    → model metadata
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import warnings, os
import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = "trained_model.pkl"

app = FastAPI(title="Obyrin — PCOS Risk Screening", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

STATE = {
    "model":         None,
    "model_name":    None,
    "imputer":       None,
    "scaler":        None,
    "feature_names": [],
    "threshold":     0.40,
    "version":       0,
    "last_retrain":  None,
}


# ── Startup — load pre-trained model ───────────────────────────────────────────
@app.on_event("startup")
def startup():
    if not os.path.exists(MODEL_PATH):
        print("❌ trained_model.pkl not found!")
        print("   Run: python train.py   to generate it, then commit to the repo.")
        return

    print("📦 Loading Logistic Regression model...")
    bundle = joblib.load(MODEL_PATH)
    STATE.update({
        "model":         bundle["model"],
        "model_name":    bundle["model_name"],
        "imputer":       bundle["imputer"],
        "scaler":        bundle["scaler"],
        "feature_names": bundle["feature_names"],
        "threshold":     bundle.get("threshold", 0.40),
        "version":       bundle.get("version", 1),
        "last_retrain":  bundle.get("last_retrain", "bundled"),
    })
    print(f"✅ Ready — {STATE['model_name']} v{STATE['version']}")
    print(f"   Features : {len(STATE['feature_names'])}")
    print(f"   Threshold: {STATE['threshold']}")


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/api/info")
def get_info():
    return {
        "model_name":    STATE["model_name"],
        "model_version": STATE["version"],
        "num_features":  len(STATE["feature_names"]),
        "feature_names": STATE["feature_names"],
        "threshold":     STATE["threshold"],
        "last_retrain":  STATE["last_retrain"],
        "ready":         STATE["model"] is not None,
    }


# ── Predict ─────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict

@app.post("/api/predict")
def predict(req: PredictRequest):
    if STATE["model"] is None:
        raise HTTPException(503, "Model not loaded. Please try again in a moment.")

    feature_names = STATE["feature_names"]
    row, missing  = [], []
    for f in feature_names:
        if f in req.features:
            row.append(float(req.features[f]))
        else:
            row.append(0.0)
            missing.append(f)
    if missing:
        print(f"⚠️  Features filled with 0: {missing}")

    arr = np.array(row).reshape(1, -1)

    # Imputer → Scaler → LR
    if STATE["imputer"] is not None:
        arr = STATE["imputer"].transform(arr)
    if STATE["scaler"] is not None:
        arr = STATE["scaler"].transform(arr)

    proba     = STATE["model"].predict_proba(arr)[0].tolist()
    threshold = STATE["threshold"]
    pred      = int(proba[1] >= threshold)

    return {
        "prediction":       pred,
        "probability":      proba,
        "threshold_used":   threshold,
        "model":            STATE["model_name"],
        "model_version":    STATE["version"],
        "missing_features": missing,
    }

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
