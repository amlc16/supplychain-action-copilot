
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.read_csv("data/orders_modeling.csv")
df["order_date"] = pd.to_datetime(df["order_date"])
df = df.sort_values("order_date").reset_index(drop=True)

numeric_features = [
    "scheduled_shipping_days",
    "order_month",
]

categorical_features = [
    "shipping_mode",
    "order_region",
    "order_type",
]

features = numeric_features + categorical_features

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
y_train = train["Late_delivery_risk"].astype(int)

X_test = test[features]
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
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

average_precision = average_precision_score(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)

top_n = int(np.ceil(len(test) * 0.10))
top_indices = np.argsort(probabilities)[-top_n:]

print("\nCOMPACT MODEL RESULTS")
print("-" * 45)
print(f"Average precision: {average_precision:.4f}")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"Top 10% late rate: {y_test.iloc[top_indices].mean():.4f}")
print(f"Mean predicted risk: {probabilities.mean():.4f}")

print("\nFEATURES USED")
print(features)

# Save separately so the existing dashboard model remains untouched.
joblib.dump(model, "models/late_risk_model_compact.joblib")
joblib.dump(features, "models/compact_feature_columns.joblib")

print("\nCompact model saved successfully.")
