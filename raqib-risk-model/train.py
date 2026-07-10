# ============================================================
# رقيب · Train the Road-Risk model (RTA Dataset) — intuitive road features
# Usage:  python train.py "RTA Dataset.csv"
# Produces: road_risk_model.joblib + feature_schema.json
# ============================================================
import sys, json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score

CSV = sys.argv[1] if len(sys.argv) > 1 else "RTA Dataset.csv"

# Intuitive road / environment / time features only (what a UI form would show)
FEATURES = ["Road_surface_type", "Road_surface_conditions", "Light_conditions",
            "Weather_conditions", "Road_allignment", "Lanes_or_Medians",
            "Types_of_Junction", "Number_of_vehicles_involved", "Hour", "Day_of_week"]
NUM = ["Number_of_vehicles_involved", "Hour"]

df = pd.read_csv(CSV).replace(['na','Unknown','unknown','nan','NaN',''], np.nan)
df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
y = df['Accident_severity'].astype(str)
X = df[FEATURES]
CAT = [c for c in FEATURES if c not in NUM]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
model = Pipeline([("pre", pre),
                  ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=300, class_weight="balanced"))])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(Xtr, ytr); pred = model.predict(Xte)
print("balanced_accuracy = %.3f" % balanced_accuracy_score(yte, pred))
print("macro_F1          = %.3f" % f1_score(yte, pred, average='macro'))
print(classification_report(yte, pred))
print("confusion matrix:\n", confusion_matrix(yte, pred, labels=sorted(y.unique())))

model.fit(X, y)
joblib.dump(model, "road_risk_model.joblib")
schema = {"target": "Accident_severity", "classes": sorted(y.unique()),
          "numeric": NUM,
          "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in CAT}}
json.dump(schema, open("feature_schema.json", "w"), ensure_ascii=False, indent=2)
print("Saved model + schema | features:", len(FEATURES))
