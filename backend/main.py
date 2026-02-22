import joblib
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="MATCHA Emergency Hospital Recommendation API",
    description="ML-powered hospital routing for emergency patients",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend")
def recommend(data: dict):
    return {"hospital_name": "Hospital 1"}

# -----------------------------
# Load ML model & encoders
# -----------------------------
try:
    rf_model = joblib.load("model.pkl")
    problem_encoder = joblib.load("problem_encoder.pkl")
    hospitals_df = pd.read_csv("hospital_master_dataset.csv")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Required file missing: {e}. "
        "Make sure model.pkl, problem_encoder.pkl, and hospital_master_dataset.csv are present."
    )

# Normalize column names to avoid mismatch between generation & training scripts
hospitals_df.rename(
    columns={"avg_wait_time_min": "avg_wait_time"},
    inplace=True
)

# Pre-validate required columns exist
REQUIRED_HOSPITAL_COLS = [
    "hospital_id", "name", "latitude", "longitude",
    "specialization", "equipment_score", "success_rate",
    "avg_wait_time", "available_doctors", "icu_beds",
    "available_beds"
]
missing_cols = [c for c in REQUIRED_HOSPITAL_COLS if c not in hospitals_df.columns]
if missing_cols:
    raise RuntimeError(f"hospital_master_dataset.csv is missing columns: {missing_cols}")

# Constants
SEVERITY_MAP = {"Critical": 3, "High": 2, "Moderate": 1}
VALID_SEVERITIES = set(SEVERITY_MAP.keys())
VALID_PROBLEMS = set(problem_encoder.classes_)

# Average ambulance speed assumption (km/h)
AVG_SPEED_KMH = 60.0


# -----------------------------
# Haversine distance (km)
# -----------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns the great-circle distance in kilometers between two GPS coordinates.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def eta_minutes_from_km(distance_km: float) -> int:
    """Estimate ETA in minutes based on distance and average ambulance speed."""
    return max(1, int(round((distance_km / AVG_SPEED_KMH) * 60)))


# -----------------------------
# Request schema
# -----------------------------
class RecommendRequest(BaseModel):
    latitude: float
    longitude: float
    problem: str
    severity: str
    age: float
    top_n: int = 3  # How many recommendations to return

    @validator("severity")
    def validate_severity(cls, v):
        if v not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{v}'. Must be one of: {sorted(VALID_SEVERITIES)}"
            )
        return v

    @validator("problem")
    def validate_problem(cls, v):
        if v not in VALID_PROBLEMS:
            raise ValueError(
                f"Unknown problem type '{v}'. Must be one of: {sorted(VALID_PROBLEMS)}"
            )
        return v

    @validator("age")
    def validate_age(cls, v):
        if not (0 < v <= 120):
            raise ValueError("Age must be between 1 and 120.")
        return v

    @validator("top_n")
    def validate_top_n(cls, v):
        if not (1 <= v <= 10):
            raise ValueError("top_n must be between 1 and 10.")
        return v

    @validator("latitude")
    def validate_latitude(cls, v):
        if not (-90 <= v <= 90):
            raise ValueError("Latitude must be between -90 and 90.")
        return v

    @validator("longitude")
    def validate_longitude(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError("Longitude must be between -180 and 180.")
        return v


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "MATCHA Emergency Backend Running 🚀",
        "valid_severities": sorted(VALID_SEVERITIES),
        "valid_problems": sorted(VALID_PROBLEMS),
        "total_hospitals_loaded": len(hospitals_df)
    }


@app.get("/hospitals")
def list_hospitals():
    """Return all hospitals in the master dataset (for debugging/frontend use)."""
    return hospitals_df[["hospital_id", "name", "specialization", "available_beds"]].to_dict(orient="records")


# -----------------------------
# Recommendation endpoint
# -----------------------------
@app.post("/recommend")
def recommend(data: RecommendRequest):
    """
    Returns top-N hospital recommendations for an incoming emergency patient.

    Filters out hospitals with no available beds, then scores each remaining
    hospital using an ML classifier and returns ranked results with ETA.
    """
    df = hospitals_df.copy()

    # --- Step 1: Filter out hospitals with no available beds ---
    df = df[df["available_beds"] > 0].reset_index(drop=True)

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="No hospitals with available beds found. Please try again shortly."
        )

    # --- Step 2: Compute real-world distances via Haversine ---
    df["distance_km"] = df.apply(
        lambda row: haversine_km(
            data.latitude, data.longitude,
            row["latitude"], row["longitude"]
        ),
        axis=1
    )

    # Normalize distance to [0, 1] range for the model
    max_dist = df["distance_km"].max()
    df["distance"] = df["distance_km"] / (max_dist + 1e-9)

    # --- Step 3: Feature engineering ---
    df["spec_match"] = (
        df["specialization"].str.lower() == data.problem.lower()
    ).astype(float)

    df["severity_encoded"] = float(SEVERITY_MAP[data.severity])
    df["age"] = float(data.age)

    # Encode problem — already validated so transform is safe
    df["problem_encoded"] = float(problem_encoder.transform([data.problem])[0])

    # --- Step 4: Build feature matrix (must match training column order) ---
    FEATURE_COLS = [
        "severity_encoded",
        "age",
        "problem_encoded",
        "distance",
        "equipment_score",
        "success_rate",
        "avg_wait_time",
        "available_doctors",
        "icu_beds",
        "spec_match"
    ]

    # Ensure all feature columns are float
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)

    X_live = df[FEATURE_COLS]

    # --- Step 5: ML inference ---
    df["ml_probability"] = rf_model.predict_proba(X_live)[:, 1]

    # --- Step 6: Sort and return top-N ---
    top_hospitals = df.sort_values("ml_probability", ascending=False).head(data.top_n)

    recommendations = []
    for rank, (_, row) in enumerate(top_hospitals.iterrows(), start=1):
        eta = eta_minutes_from_km(row["distance_km"])
        recommendations.append({
            "rank": rank,
            "hospital_id": row["hospital_id"],
            "hospital_name": row["name"],
            "specialization": row["specialization"],
            "spec_match": bool(row["spec_match"]),
            "available_beds": int(row["available_beds"]),
            "icu_beds": int(row["icu_beds"]),
            "distance_km": round(float(row["distance_km"]), 3),
            "eta_minutes": eta,
            "equipment_score": round(float(row["equipment_score"]), 3),
            "success_rate": round(float(row["success_rate"]), 3),
            "avg_wait_time_min": round(float(row["avg_wait_time"]), 1),
            "available_doctors": int(row["available_doctors"]),
            "ml_confidence": round(float(row["ml_probability"]), 4),
        })

    return {
        "patient": {
            "severity": data.severity,
            "problem": data.problem,
            "age": data.age,
            "latitude": data.latitude,
            "longitude": data.longitude,
        },
        "recommendations": recommendations,
        "hospitals_considered": len(df),
        "hospitals_with_beds": len(df),
    }
