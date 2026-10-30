
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/orders_modeling.csv")
MODEL_PATH = Path("models/late_risk_model.joblib")
FEATURES_PATH = Path("models/feature_columns.joblib")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

X = df[features]
y = df["Late_delivery_risk"].astype(int)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

probabilities = model.predict_proba(X_test)[:, 1]

print(f"Test-set late-risk rate: {y_test.mean():.3f}")
print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")

rows = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    predictions = (probabilities >= threshold).astype(int)

    rows.append({
        "threshold": threshold,
        "orders_flagged": int(predictions.sum()),
        "percent_flagged": 100 * predictions.mean(),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
    })

results = pd.DataFrame(rows)

print("\n--- Threshold comparison ---")
print(results.round(3).to_string(index=False))

# Operational view: review only the 10% of orders with the highest scores.
test_results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "risk_score": probabilities,
})

cutoff = test_results["risk_score"].quantile(0.90)
top_10 = test_results[test_results["risk_score"] >= cutoff]

print("\n--- Highest-risk 10% of orders (approximately) ---")
print(f"Risk-score cutoff: {cutoff:.4f}")
print(f"Orders selected: {len(top_10):,} of {len(test_results):,}")
print(f"Late-risk rate among selected: {top_10['actual'].mean():.3f}")
print(f"Overall late-risk rate: {test_results['actual'].mean():.3f}")
