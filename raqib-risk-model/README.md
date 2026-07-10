# رقيب · Road-Risk Prediction (Model + API)

Predicts road **accident severity** (Slight / Serious / Fatal) from road and environmental
factors — so the authority can prioritize and fix dangerous roads proactively.

## Files
- `train.py` — trains the model and saves the artifacts.
- `road_risk_model.joblib` — the trained pipeline (preprocessing + model).
- `feature_schema.json` — expected columns, classes, and accepted category values.
- `app.py` — FastAPI service (`/predict`, `/health`, `/schema`).
- `important_columns.txt` — the key columns explained.
- `performance_report.md` — metrics and honest interpretation.
- `road_risk_model.ipynb` — runnable notebook (EDA → train → evaluate).

## Setup
```bash
pip install -r requirements.txt
```

## Retrain (optional)
```bash
python train.py "RTA Dataset.csv"
```

## Run the API
```bash
uvicorn app:app --reload --port 8000
# docs: http://localhost:8000/docs
```

## Example request
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{
  \"Road_surface_type\": \"Earth roads\",
  \"Road_surface_conditions\": \"Flood over 3cm. deep\",
  \"Light_conditions\": \"Darkness - no lighting\",
  \"Weather_conditions\": \"Raining\",
  \"Road_allignment\": \"Steep grade downward with mountainous terrain\",
  \"Lanes_or_Medians\": \"Undivided Two way\",
  \"Types_of_Junction\": \"Y Shape\",
  \"Number_of_vehicles_involved\": 3,
  \"Hour\": 2,
  \"Day_of_week\": \"Saturday\"
}"
```

## Response (hybrid)
```json
{
  "road_risk": {"risk_score": 0.85, "risk_level": "High", "top_factors": ["surface_cond","light","surface_type"]},
  "ml_model": {"predicted_severity": "Slight Injury", "probabilities": {"Slight Injury": 0.7, "Serious Injury": 0.23, "Fatal injury": 0.06}}
}
```

- `road_risk` = transparent, intuitive rule-based index over road conditions (drives the UI gauge).
- `ml_model` = data-driven severity model trained on RTA.
