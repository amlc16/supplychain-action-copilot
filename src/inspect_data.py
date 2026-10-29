import pandas as pd
from pathlib import Path

# Check if the data directory exists and contains CSV files
data_dir = Path("data")
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV found in data/. Download and extract the dataset first."
    )

# Load the first CSV file into a DataFrame
df = pd.read_csv(csv_files[1], encoding="latin1", low_memory=False)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values in the 10 columns with the most missing data:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# print name of the file that was loaded
print("\nLoaded file:", csv_files[0].name)

