from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
TARGET = "Late_delivery_risk"
ORDER_ID = "Order Id"

csv_files = [
    path for path in DATA_DIR.glob("*.csv")
    if path.name != "orders_modeling.csv"
]

if len(csv_files) != 1:
    raise ValueError(
        f"Expected one original CSV in data/, found: "
        f"{[p.name for p in csv_files]}"
    )

df = pd.read_csv(csv_files[0], encoding="latin1", low_memory=False)

# Keep only rows with a valid binary target and order ID.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET, ORDER_ID]).copy()

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("The target contains values other than 0 and 1.")

# Confirm every item belonging to an order has the same target.
label_counts = df.groupby(ORDER_ID)[TARGET].nunique()
inconsistent_orders = label_counts[label_counts > 1]

print(f"Original item rows: {len(df):,}")
print(f"Unique orders: {df[ORDER_ID].nunique():,}")
print(f"Orders with inconsistent labels: {len(inconsistent_orders):,}")

if not inconsistent_orders.empty:
    raise ValueError(
        "Some orders have conflicting target labels. "
        "Inspect these orders before continuing."
    )

# Parse the order date to derive features available at order time.
order_date = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)
# Drop rows with invalid order dates.
df["order_date_parsed"] = order_date
df["order_month"] = order_date.dt.month
df["order_day_of_week"] = order_date.dt.dayofweek

# Convert numeric inputs explicitly.
numeric_columns = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Product Price",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# One row per order. Use order-time information only.
orders = df.groupby(ORDER_ID, as_index=False).agg(
    Late_delivery_risk=(TARGET, "first"),
    order_type=("Type", "first"),
    scheduled_shipping_days=("Days for shipment (scheduled)", "first"),
    customer_segment=("Customer Segment", "first"),
    market=("Market", "first"),
    order_region=("Order Region", "first"),
    shipping_mode=("Shipping Mode", "first"),
    customer_country=("Customer Country", "first"),
    order_country=("Order Country", "first"),
    order_date=("order_date_parsed", "first"),
    order_month=("order_month", "first"),
    order_day_of_week=("order_day_of_week", "first"),
    item_count=(TARGET, "size"),
    total_quantity=("Order Item Quantity", "sum"),
    total_sales=("Sales", "sum"),
    total_item_value=("Order Item Total", "sum"),
    total_discount=("Order Item Discount", "sum"),
    mean_discount_rate=("Order Item Discount Rate", "mean"),
    mean_product_price=("Product Price", "mean"),
    mean_benefit_per_order=("Benefit per order", "mean"),
)

output_path = DATA_DIR / "orders_modeling.csv"
orders.to_csv(output_path, index=False)

print("\n--- Order-level dataset ---")
print(f"Orders: {len(orders):,}")
print(f"Columns: {len(orders.columns)}")
print(f"Saved to: {output_path}")

print("\n--- Target distribution by order ---")
print(orders["Late_delivery_risk"].value_counts().sort_index())
print(
    orders["Late_delivery_risk"]
    .value_counts(normalize=True)
    .sort_index()
    .round(3)
)

print("\n--- Missing values ---")
print(orders.isna().sum().sort_values(ascending=False).head(10))

print("\n--- Preview ---")
print(orders.head().to_string(index=False))