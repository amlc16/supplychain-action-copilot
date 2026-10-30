
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/orders_modeling.csv")

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

df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
df = df.dropna(subset=["order_date", "Late_delivery_risk"])
df = df.sort_values("order_date").reset_index(drop=True)

split_at = int(len(df) * 0.80)
train = df.iloc[:split_at].copy()
test = df.iloc[split_at:].copy()

print(f"Training orders: {len(train):,}")
print(f"Test orders: {len(test):,}")
print(
    f"Training dates: {train['order_date'].min()} to "
    f"{train['order_date'].max()}"
)
print(
    f"Test dates: {test['order_date'].min()} to "
    f"{test['order_date'].max()}"
)

X_train = train[numeric_features + categorical_features]
y_train = train["Late_delivery_risk"].astype(int)
X_test = test[numeric_features + categorical_features]
y_test = test["Late_delivery_risk"].astype(int)

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric_features,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]

print(f"\nTest late-risk rate: {y_test.mean():.3f}")
print(f"Average precision: {average_precision_score(y_test, scores):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, scores):.4f}")

# Select approximately the top 10% of test orders for review.
cutoff = pd.Series(scores).quantile(0.90)
selected = scores >= cutoff

print("\n--- Top 10% review queue ---")
print(f"Risk-score cutoff: {cutoff:.4f}")
print(f"Orders selected: {selected.sum():,} / {len(test):,}")
print(f"Fraction selected: {selected.mean():.1%}")
print(f"Late-risk rate in selected: {y_test.to_numpy()[selected].mean():.3f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/late_risk_model_temporal.joblib")
print("\nSaved models/late_risk_model_temporal.joblib")
