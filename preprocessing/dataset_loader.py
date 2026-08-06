"""
Loads Amazon Review datasets stored as JSONL files.
"""

from pathlib import Path
import polars as pl
from configs.config import RAW_DATA_DIR


class DatasetLoader:

    def __init__(self, category: str = "All_Beauty") -> None:

        self.category = category
        self.dataset_path = (
            RAW_DATA_DIR
            / f"{category}.jsonl"
        )

    def load(self) -> pl.DataFrame:

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        print(f"\nLoading dataset: {self.category}")
        df = pl.read_ndjson(self.dataset_path)
        print(f"Loaded {df.height:,} reviews.")
        return df


    @staticmethod
    def preview(df: pl.DataFrame, rows: int = 5) -> None:
        print("\nDataset Preview\n")
        print(df.head(rows))


    @staticmethod
    def information(df: pl.DataFrame) -> None:

        print("\nDataset Information\n")
        print(f"Rows    : {df.height:,}")
        print(f"Columns : {df.width}")
        print("\nColumns\n")

        for column in df.columns:
            print(f"- {column}")

        print("\nSchema\n")
        print(df.schema)

