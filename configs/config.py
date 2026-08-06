"""
Project Library Path configuration.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

PRODUCT_METADATA_PATH = PROCESSED_DATA_DIR / "All_Beauty_products.parquet"

MODEL_DIR = BASE_DIR / "models"

ML_MODEL_DIR = MODEL_DIR / "ml"

LLM_MODEL_DIR = MODEL_DIR / "llm"
