import pandas as pd
import numpy as np
import random

np.random.seed(42)

# ---------------------------------
# CONFIG
# ---------------------------------
num_samples = 1000

severities = ["Low", "Moderate", "High", "Critical"]
problems = ["Cardiac", "Trauma", "Respiratory", "General"]

hospitals = [f"Thapar Hospital {i+1}" for i in range(10)]

data = []

for _ in range(num_samples):

    severity = random.choice(severities)
    problem = random.choice(problems)
    age = random.randint(1, 90)

    # Distance between 0 and 1 (normalized)
    distance = np.round(np.random.uniform(0.01, 1.0), 3)

    # Occupancy between 0 and 1
    occupancy_rate = np.round(np.random.uniform(0.1, 0.95), 3)

    # Specialization match probability higher for correct problem
    spec_match = np.random.choice([0, 1], p=[0.3, 0.7])

    # Smart hospital selection logic (simulated ground truth)
    if severity == "Critical":
        score = (0.6 * distance) + (0.2 * occupancy_rate) - (0.2 * spec_match)
    elif severity == "High":
        score = (0.4 * distance) + (0.3 * occupancy_rate) - (0.3 * spec_match)
    elif severity == "Moderate":
        score = (0.3 * distance) + (0.4 * occupancy_rate) - (0.3 * spec_match)
    else:
        score = (0.2 * distance) + (0.5 * occupancy_rate) - (0.3 * spec_match)

    # Simulate hospital ID based on score (lower score better)
    hospital_id = int((score * 10) % 10)
    selected_hospital = hospitals[hospital_id]

    data.append([
        severity,
        problem,
        age,
        distance,
        occupancy_rate,
        spec_match,
        selected_hospital
    ])

df = pd.DataFrame(data, columns=[
    "severity",
    "problem",
    "age",
    "distance",
    "occupancy_rate",
    "spec_match",
    "selected_hospital"
])

df.to_csv("patient_training_data.csv", index=False)

print("✅ Synthetic dataset generated successfully!")
print("Rows created:", len(df))