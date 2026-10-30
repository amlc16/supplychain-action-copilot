from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV found in data/. Put the extracted DataCo CSV there."
    )

if len(csv_files) > 1:
    raise ValueError(
        f"Found multiple CSV files: {[f.name for f in csv_files]}. "
        "Keep only the intended dataset in data/ for now."
    )

csv_path = csv_files[0]
print(f"Loading: {csv_path}")

df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)

print("\n--- Dataset overview ---")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"File size: {csv_path.stat().st_size / (1024**2):.1f} MB")

print("\n--- Column names ---")
for column in df.columns:
    print(column)

print("\n--- First five rows ---")
print(df.head().to_string())

print("\n--- Missing values (top 10 columns) ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

target = "Late_delivery_risk"

print("\n--- Target inspection ---")
if target not in df.columns:
    print(f"WARNING: Target column '{target}' was not found.")
else:
    print(df[target].value_counts(dropna=False).sort_index())
    print("\nTarget proportions:")
    print(df[target].value_counts(normalize=True, dropna=False).sort_index().round(3))

print("\n--- Potential outcome-leakage columns ---")
suspect_columns = [
    "Delivery Status",
    "Days for shipping (real)",
    "Order Status",
]

for column in suspect_columns:
    if column in df.columns:
        print(f"{column}: present — do not use as a prediction feature")