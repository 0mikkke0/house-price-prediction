from pathlib import Path
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ames_housing.csv"

print("Downloading Ames Housing dataset from OpenML...")
dataset = fetch_openml(data_id=42165, as_frame=True, parser="auto")
df = dataset.frame.copy()
df.to_csv(OUT, index=False)

print(f"Saved {len(df):,} rows to {OUT}")
print("Columns:", len(df.columns))
