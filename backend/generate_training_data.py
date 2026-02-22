import pandas as pd
import numpy as np

# ---------------------------------
# 📍 Load Hospital Master Dataset
# ---------------------------------
hospitals = pd.read_csv("hospital_master_dataset.csv")

# ---------------------------------
# 👤 Generate Synthetic Patients
# ---------------------------------
num_patients = 500   # You can increase to 1000+

severities = ["Critical", "High", "Moderate"]
problems = ["Cardiac", "Trauma", "Respiratory", "General"]

patients = []

for _ in range(num_patients):
    patients.append({
        "severity": np.random.choice(severities),
        "age": np.random.randint(1, 90),
        "problem": np.random.choice(problems),
        "latitude": np.random.uniform(30.350, 30.360),
        "longitude": np.random.uniform(76.364, 76.372)
    })

patients_df = pd.DataFrame(patients)

# ---------------------------------
# 🏗 Create Training Rows
# ---------------------------------
training_rows = []

for _, patient in patients_df.iterrows():
    hospital_scores = []

    for _, hospital in hospitals.iterrows():

        # Calculate distance
        distance = np.sqrt(
            (hospital["latitude"] - patient["latitude"])**2 +
            (hospital["longitude"] - patient["longitude"])**2
        )

        # Specialization match
        spec_match = int(hospital["specialization"].lower() == patient["problem"].lower())

        # Create simple smart suitability score
        score = (
            (1 / (distance + 0.001)) * 0.4 +
            hospital["equipment_score"] * 0.1 +
            hospital["success_rate"] * 0.2 -
            hospital["avg_wait_time_min"] * 0.01 +
            hospital["available_doctors"] * 0.05 +
            hospital["icu_beds"] * 0.02 +
            spec_match * 0.3
        )

        # Increase importance for critical severity
        if patient["severity"] == "Critical":
            score += hospital["icu_beds"] * 0.05

        hospital_scores.append((hospital["hospital_id"], score))

        training_rows.append({
            "severity": patient["severity"],
            "age": patient["age"],
            "problem": patient["problem"],
            "hospital_id": hospital["hospital_id"],
            "distance": distance,
            "equipment_score": hospital["equipment_score"],
            "success_rate": hospital["success_rate"],
            "avg_wait_time": hospital["avg_wait_time_min"],
            "available_doctors": hospital["available_doctors"],
            "icu_beds": hospital["icu_beds"],
            "spec_match": spec_match,
            "label": 0  # temporary
        })

    # Mark best hospital for this patient
    best_hospital_id = max(hospital_scores, key=lambda x: x[1])[0]

    for row in training_rows[-len(hospitals):]:
        if row["hospital_id"] == best_hospital_id:
            row["label"] = 1

# ---------------------------------
# 💾 Save Training Dataset
# ---------------------------------
training_df = pd.DataFrame(training_rows)
training_df.to_csv("final_training_dataset.csv", index=False)

print("✅ Training dataset generated successfully!")