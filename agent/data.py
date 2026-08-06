import gc
import pandas as pd
import pyarrow.parquet as pq
from . import config


def safe_read_parquet(path: str, cols: list[str]) -> pd.DataFrame:
    """Read only columns that exist in the file's schema; warn instead of crashing on drift."""
    available = set(pq.ParquetFile(path).schema.names)
    missing = [c for c in cols if c not in available]
    if missing:
        print(f"Skipping missing columns in {path}: {missing}")
    return pd.read_parquet(path, columns=[c for c in cols if c in available])


def load_dataset() -> pd.DataFrame:
    products_df = safe_read_parquet(config.PRODUCT_PATH, config.PRODUCT_COLS)
    reviews_df = safe_read_parquet(config.REVIEW_PATH, config.REVIEW_COLS)

    df = reviews_df.merge(products_df, on="parent_asin", how="left", suffixes=("_review", "_product"))
    del products_df, reviews_df
    gc.collect()

    for c in df.select_dtypes(include="float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include="int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")

    print(f"Combined dataset: {df.shape}")
    return df
