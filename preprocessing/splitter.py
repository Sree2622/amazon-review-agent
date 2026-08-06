"""
Creates Train, Validation and Test datasets from the processed Amazon Reviews dataset.
"""

from pathlib import Path
import polars as pl
from sklearn.model_selection import train_test_split
from configs.config import PROCESSED_DATA_DIR


class DatasetSplitter:
    def __init__(
        self,
        category: str = "All_Beauty",
        train_size: float = 0.8,
        validation_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> None:

        if abs(train_size + validation_size + test_size - 1.0) > 1e-6:
            raise ValueError(
                "Train, validation and test sizes must sum to 1."
            )

        self.category = category
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

        self.input_path = (
            PROCESSED_DATA_DIR /
            f"{category}.parquet"
        )

    def split(self) -> None:
        """
        Split processed dataset.
        """

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found:\n{self.input_path}"
            )

        print("\nLoading processed dataset...\n")

        df = pl.read_parquet(self.input_path)

        pdf = df.to_pandas()

        print(f"Total Reviews : {len(pdf):,}")


        # Train vs Remaining
        train_df, remaining_df = train_test_split(
            pdf,
            train_size=self.train_size,
            stratify=pdf["label"],
            random_state=self.random_state,
        )


        # Validation vs Test
        validation_ratio = (
            self.validation_size /
            (self.validation_size + self.test_size)
        )

        validation_df, test_df = train_test_split(
            remaining_df,
            train_size=validation_ratio,
            stratify=remaining_df["label"],
            random_state=self.random_state,
        )


        # Save
        train_path = PROCESSED_DATA_DIR / "train.parquet"
        validation_path = PROCESSED_DATA_DIR / "validation.parquet"
        test_path = PROCESSED_DATA_DIR / "test.parquet"

        pl.from_pandas(train_df).write_parquet(train_path)
        pl.from_pandas(validation_df).write_parquet(validation_path)
        pl.from_pandas(test_df).write_parquet(test_path)

        print("\nDataset Split Complete\n")

        print(f"Train      : {len(train_df):,}")
        print(f"Validation : {len(validation_df):,}")
        print(f"Test       : {len(test_df):,}")

        print("\nSaved Files")

        print(train_path)
        print(validation_path)
        print(test_path)


def main() -> None:

    splitter = DatasetSplitter()

    splitter.split()


if __name__ == "__main__":
    main()