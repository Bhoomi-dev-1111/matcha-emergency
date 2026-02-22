import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("final_training_dataset.csv")

# Encode severity
severity_map = {"Critical": 3, "High": 2, "Moderate": 1}
df["severity_encoded"] = df["severity"].map(severity_map)

# Encode problem
problem_encoder = LabelEncoder()
df["problem_encoded"] = problem_encoder.fit_transform(df["problem"])

# Select features
X = df[
    [
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
]

y = df["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(problem_encoder, "problem_encoder.pkl")

print("✅ Model trained and saved successfully!")