from pathlib import Path
import pandas as pd


def load_suppliers(csv_path: str = "data/suppliers_dataset.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path.resolve()}")
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_suppliers()
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print("-", col)