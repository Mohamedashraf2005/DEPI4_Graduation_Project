# ============================================================
# رقيب · Transparent, intuitive road-risk index (rule-based)
# Maps road / environment conditions to a 0-1 risk score with
# explainable weights. Guarantees intuitive behaviour for the UI.
# ============================================================

# per-value danger (0 = safe, 1 = most dangerous). Unknown -> 0.3 default.
SURFACE_COND = {"Dry": 0.0, "Wet or damp": 0.55, "Snow": 0.8, "Flood over 3cm. deep": 1.0}
LIGHT = {"Daylight": 0.0, "Darkness - lights lit": 0.4,
         "Darkness - lights unlit": 0.75, "Darkness - no lighting": 1.0}
WEATHER = {"Normal": 0.0, "Cloudy": 0.15, "Windy": 0.35, "Other": 0.3,
           "Raining": 0.6, "Raining and Windy": 0.75, "Fog or mist": 0.85, "Snow": 0.9}
SURFACE_TYPE = {"Asphalt roads": 0.0, "Asphalt roads with some distress": 0.45,
                "Gravel roads": 0.7, "Earth roads": 0.9, "Other": 0.3}
ALIGN = {"Tangent road with flat terrain": 0.0,
         "Tangent road with mild grade and flat terrain": 0.2,
         "Tangent road with rolling terrain": 0.35,
         "Tangent road with mountainous terrain and": 0.55,
         "Gentle horizontal curve": 0.5, "Sharp reverse curve": 0.85,
         "Steep grade upward with mountainous terrain": 0.8,
         "Steep grade downward with mountainous terrain": 0.9,
         "Escarpments": 0.8}
LANES = {"Double carriageway (median)": 0.0,
         "Two-way (divided with solid lines road marking)": 0.15,
         "One way": 0.2,
         "Two-way (divided with broken lines road marking)": 0.45,
         "Undivided Two way": 0.75, "other": 0.4}
JUNCTION = {"No junction": 0.0, "O Shape": 0.5, "T Shape": 0.55, "Y Shape": 0.6,
            "Crossing": 0.7, "X Shape": 0.75, "Other": 0.4}

# weight of each factor in the final index
WEIGHTS = {"surface_cond": 0.20, "light": 0.16, "weather": 0.16, "surface_type": 0.14,
           "align": 0.12, "lanes": 0.08, "junction": 0.06, "vehicles": 0.04, "hour": 0.04}

def _g(table, val): 
    return table.get(val, 0.3) if val is not None else 0.3

def _vehicles(n):
    try: n = float(n)
    except (TypeError, ValueError): return 0.3
    return min(max((n - 1) / 3.0, 0.0), 1.0)   # 1->0, 4+->1

def _hour(h):
    try: h = int(float(h))
    except (TypeError, ValueError): return 0.3
    if 22 <= h or h <= 5: return 0.9      # night
    if 6 <= h <= 7 or 18 <= h <= 21: return 0.45  # dusk/dawn
    return 0.1                             # daytime

def road_risk_index(f: dict):
    parts = {
        "surface_cond": _g(SURFACE_COND, f.get("Road_surface_conditions")),
        "light": _g(LIGHT, f.get("Light_conditions")),
        "weather": _g(WEATHER, f.get("Weather_conditions")),
        "surface_type": _g(SURFACE_TYPE, f.get("Road_surface_type")),
        "align": _g(ALIGN, f.get("Road_allignment")),
        "lanes": _g(LANES, f.get("Lanes_or_Medians")),
        "junction": _g(JUNCTION, f.get("Types_of_Junction")),
        "vehicles": _vehicles(f.get("Number_of_vehicles_involved")),
        "hour": _hour(f.get("Hour")),
    }
    score = sum(parts[k] * WEIGHTS[k] for k in parts)
    level = "High" if score >= 0.5 else ("Medium" if score >= 0.28 else "Low")
    top = sorted(parts.items(), key=lambda kv: -kv[1] * WEIGHTS[kv[0]])[:3]
    return {"risk_score": round(score, 3), "risk_level": level,
            "top_factors": [k for k, _ in top], "breakdown": {k: round(v, 2) for k, v in parts.items()}}
