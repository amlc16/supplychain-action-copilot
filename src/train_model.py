
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/orders_modeling.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Late_delivery_risk"

# These features are intended to describe the order at order time.
# Exclude order IDs and fields that directly reveal the delivery outcome.
numeric_features = [
    "scheduled_shipping_days",
    "order_month",
    "order_day_of_week",
    "item_count",
    "total_quantity",
    "total_sales",
    "total_item_value",
    "total_discount",
    "mean_discount_rate",
    "mean_product_price",
    "mean_benefit_per_order",
]

categorical_features = [
    "order_type",
    "customer_segment",
    "market",
    "order_region",
    "shipping_mode",
    "customer_country",
    "order_country",
]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run python src/prepare_data.py first."
    )

df = pd.read_csv(DATA_PATH)
# Check that all required columns are present.
features = numeric_features + categorical_features
missing = [col for col in features + [TARGET] if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[features].copy()
y = df[TARGET].astype(int)

# Split by rows after aggregation: each order is now a single record.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
# Define preprocessing pipelines for numeric and categorical features.
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# Combine numeric and categorical pipelines into a single preprocessor.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)
# Define models to compare.
models = {
    "Logistic Regression (baseline)": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}
# Train and evaluate each model, keeping track of the best one by average precision.
results = []
best_name = None
best_pipeline = None
best_ap = -1.0 

for name, classifier in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print(f"\n{'=' * 65}")
    print(f"Training: {name}")

    pipeline.fit(X_train, y_train)

    # Evaluate on the test set.
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    # Use a threshold of 0.50 for classification.
    predictions = (probabilities >= 0.5).astype(int)
    # Calculate average precision and ROC-AUC.
    ap = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Average precision (PR-AUC summary): {ap:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification report at threshold 0.50:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not flagged", "Late-risk"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions))

    results.append(
        {
            "model": name,
            "average_precision": ap,
            "roc_auc": roc_auc,
        }
    )

    if ap > best_ap:
        best_ap = ap
        best_name = name
        best_pipeline = pipeline

results_df = pd.DataFrame(results).sort_values(
    "average_precision", ascending=False
)

print(f"\n{'=' * 65}")
print("MODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nBest model by average precision: {best_name}")

joblib.dump(best_pipeline, MODEL_DIR / "late_risk_model.joblib")
joblib.dump(features, MODEL_DIR / "feature_columns.joblib")

print(f"Saved model to {MODEL_DIR / 'late_risk_model.joblib'}")
