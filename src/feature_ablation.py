
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

split = int(len(df) * 0.80)
train = df.iloc[:split]
test = df.iloc[split:]

y_train = train["Late_delivery_risk"].astype(int)
y_test = test["Late_delivery_risk"].astype(int)

numeric = [
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

categorical = [
    "order_type",
    "customer_segment",
    "market",
    "order_region",
    "shipping_mode",
    "customer_country",
    "order_country",
]


def run_experiment(name, num_features, cat_features):
    features = num_features + cat_features

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_features,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(train[features], y_train)
    probabilities = model.predict_proba(test[features])[:, 1]

    top_n = int(np.ceil(len(test) * 0.10))
    top_indices = np.argsort(probabilities)[-top_n:]

    print(f"\n{name}")
    print("-" * 45)
    print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")
    print(f"ROC-AUC:           {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Top 10% late rate: {y_test.iloc[top_indices].mean():.4f}")
    print(f"Mean predicted risk: {probabilities.mean():.4f}")


run_experiment("FULL MODEL", numeric, categorical)

run_experiment(
    "WITHOUT SHIPPING FEATURES",
    [f for f in numeric if f != "scheduled_shipping_days"],
    [f for f in categorical if f != "shipping_mode"],
)

run_experiment(
    "WITHOUT SHIPPING OR FINANCIAL FEATURE",
    [
        f for f in numeric
        if f not in ["scheduled_shipping_days", "mean_benefit_per_order"]
    ],
    [f for f in categorical if f != "shipping_mode"],
)
