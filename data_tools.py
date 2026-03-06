from __future__ import annotations
import os
import pandas as pd

DATA_DIR = "data/"

def _csv_path(filename: str) -> str:
    # prevent path traversal
    filename = os.path.basename(filename)
    return os.path.join(DATA_DIR, filename)

def load_csv(filename: str) -> pd.DataFrame:
    path = _csv_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    return pd.read_csv(path)

def df_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
    }

def missing_values(df: pd.DataFrame, top_n: int = 10) -> dict:
    mv = df.isna().sum().sort_values(ascending=False)
    mv = mv[mv > 0].head(top_n)
    return {"missing": mv.to_dict()}

def top_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    return df.head(n).to_dict(orient="records")

def value_counts(df: pd.DataFrame, column: str, n: int = 10) -> dict:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in CSV.")
    vc = df[column].value_counts(dropna=False).head(n)
    return {"counts": vc.to_dict()}

def group_mean(df: pd.DataFrame, group_col: str, metric_col: str, n: int = 10) -> dict:
    for c in (group_col, metric_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in CSV.")
    g = (
        df.groupby(group_col, dropna=False)[metric_col]
        .mean(numeric_only=True)
        .sort_values(ascending=False)
        .head(n)
    )
    return {"mean_by_group": g.to_dict()}
