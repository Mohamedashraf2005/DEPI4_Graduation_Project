# ============================================================
# رقيب · Road-Risk Prediction API (FastAPI)
# Predicts accident severity from intuitive road / environment factors.
# Run:  uvicorn app:app --reload --port 8000   (docs at /docs)
# ============================================================
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from risk_rules import road_risk_index

BASE = Path(__file__).parent
model = joblib.load(BASE / "road_risk_model.joblib")
schema = json.load(open(BASE / "feature_schema.json", encoding="utf-8"))
COLS = schema["numeric"] + list(schema["categorical"].keys())

app = FastAPI(title="Raqib Road-Risk API", version="2.0",
              description="Predicts road accident severity (Slight / Serious / Fatal) from road & environmental factors.")

class RoadFeatures(BaseModel):
    Road_surface_type: Optional[str] = None          # Asphalt / Gravel / Earth ...
    Road_surface_conditions: Optional[str] = None     # Dry / Wet or damp / Snow / Flood over 3cm. deep
    Light_conditions: Optional[str] = None            # Daylight / Darkness - lights lit / unlit / no lighting
    Weather_conditions: Optional[str] = None          # Normal / Raining / Fog or mist / Windy / Snow ...
    Road_allignment: Optional[str] = None             # Tangent flat / curves / steep grades
    Lanes_or_Medians: Optional[str] = None            # One way / divided / undivided
    Types_of_Junction: Optional[str] = None           # No junction / Y / T / X / O / Crossing
    Number_of_vehicles_involved: Optional[float] = None
    Hour: Optional[float] = None                       # 0-23
    Day_of_week: Optional[str] = None

RISK_W = {"Slight Injury": 0.2, "Serious Injury": 0.6, "Fatal injury": 1.0}

@app.get("/health")
def health():
    return {"status": "ok", "classes": schema["classes"], "n_features": len(COLS)}

@app.get("/schema")
def get_schema():
    return schema

@app.post("/predict")
def predict(f: RoadFeatures):
    data = f.model_dump()
    df = pd.DataFrame([{c: data.get(c, None) for c in COLS}])
    pred = str(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    probs = {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)}
    # intuitive rule-based road-risk index (drives the UI gauge)
    ri = road_risk_index(data)
    return {
        "road_risk": ri,                       # <- primary, intuitive
        "ml_model": {                          # <- analytical (data-driven severity)
            "predicted_severity": pred,
            "probabilities": probs,
        },
    }
