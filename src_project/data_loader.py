import pandas as pd
from .config import DATA_PATH

def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load CSV and Strip column names"""
    csv_path = path or DATA_PATH
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # It will remove extra spaces
    return df