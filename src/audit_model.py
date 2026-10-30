
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/orders_modeling.csv"
MODEL_PATH = "models/late_risk_model_temporal.joblib"

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "order_type",
    "customer_segment",
    "market",
    "order_region",
    "shipping_mode",
    "customer_country",
    "order_country",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def evaluate(name, y_true, probabilities):
    n_top = int(np.ceil(len(y_true) * 0.10))
    top_indices = np.argsort(probabilities)[-n_top:]

    print(f"\n{name}")
    print("-" * 45)
    print(f"Average precision: {average_precision_score(y_true, probabilities):.4f}")
    print(f"ROC-AUC:           {roc_auc_score(y_true, probabilities):.4f}")
    print(
        f"Top 10% late rate: {y_true.iloc[top_indices].mean():.4f}"
    )
    print(
        f"Mean predicted risk: {probabilities.mean():.4f}"
    )


# Load and recreate the chronological train/test split.
df = pd.read_csv(DATA_PATH)
df["order_date"] = pd.to_datetime(df["order_date"])
df = df.sort_values("order_date").reset_index(drop=True)

split_index = int(len(df) * 0.80)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[FEATURES]
y_train = train["Late_delivery_risk"].astype(int)

X_test = test[FEATURES]
y_test = test["Late_delivery_risk"].astype(int)

# Evaluate the existing full model.
import joblib

full_model = joblib.load(MODEL_PATH)
full_probabilities = full_model.predict_proba(X_test)[:, 1]

evaluate("FULL MODEL", y_test, full_probabilities)

# Train a simple baseline using only shipping mode and scheduled days.
baseline_numeric = ["scheduled_shipping_days"]
baseline_categorical = ["shipping_mode"]

baseline_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            baseline_numeric,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            baseline_categorical,
        ),
    ]
)

baseline_model = Pipeline([
    ("preprocessor", baseline_preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

baseline_model.fit(
    train[baseline_numeric + baseline_categorical],
    y_train,
)

baseline_probabilities = baseline_model.predict_proba(
    test[baseline_numeric + baseline_categorical]
)[:, 1]

evaluate("SHIPPING-ONLY BASELINE", y_test, baseline_probabilities)

# Inspect actual late-delivery rates across important categories.
print("\nLATE-DELIVERY RATE BY SHIPPING MODE")
print("-" * 45)
print(
    test.groupby("shipping_mode")["Late_delivery_risk"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
    .to_string()
)

print("\nLATE-DELIVERY RATE BY SCHEDULED SHIPPING DAYS")
print("-" * 45)
print(
    test.groupby("scheduled_shipping_days")["Late_delivery_risk"]
    .agg(["count", "mean"])
    .sort_index()
    .to_string()
)

# Inspect the highest-risk 10% of test orders.
top_n = int(np.ceil(len(test) * 0.10))
top_indices = np.argsort(full_probabilities)[-top_n:]
top_orders = test.iloc[top_indices].copy()
top_orders["predicted_risk"] = full_probabilities[top_indices]

print("\nHIGHEST-RISK 10%: FEATURE SUMMARY")
print("-" * 45)
print(f"Orders: {len(top_orders)}")
print(
    f"Actual late-delivery rate: "
    f"{top_orders['Late_delivery_risk'].mean():.4f}"
)
# Inspect feature distributions for the highest-risk orders.
print("\nShipping mode:")
print(top_orders["shipping_mode"].value_counts().to_string())
print("\nScheduled shipping days:")
print(
    top_orders["scheduled_shipping_days"]
    .value_counts()
    .sort_index()
    .to_string()
)
print("\nPredicted risk distribution:")
print(top_orders["predicted_risk"].describe().to_string())
