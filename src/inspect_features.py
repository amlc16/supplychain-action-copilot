
import pandas as pd

df = pd.read_csv("data/orders_modeling.csv")

target = "Late_delivery_risk"

categorical_features = [
    "shipping_mode",
    "order_type",
    "customer_segment",
    "market",
    "order_region",
    "order_month",
    "order_day_of_week",
]

numeric_features = [
    "scheduled_shipping_days",
    "item_count",
    "total_quantity",
    "total_sales",
    "total_discount",
    "mean_discount_rate",
    "mean_product_price",
    "mean_benefit_per_order",
]

print("\nTARGET DISTRIBUTION")
print("-" * 50)
print(df[target].value_counts(normalize=True).sort_index())

print("\nCATEGORICAL FEATURES")
for feature in categorical_features:
    print(f"\n{feature}")
    print("-" * 50)

    summary = (
        df.groupby(feature, dropna=False)[target]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )

    print(summary.head(20).to_string())

print("\nNUMERICAL FEATURES")
for feature in numeric_features:
    print(f"\n{feature}")
    print("-" * 50)

    summary = (
        df.groupby(feature, dropna=False)[target]
        .agg(["count", "mean"])
    )

    print(summary.describe().round(3).to_string())

    print("\nCorrelation with target:")
    print(df[[feature, target]].corr().iloc[0, 1].round(4))
