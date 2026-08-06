"""
Load processed datasets for machine learning.
"""

import polars as pl

from configs.config import PROCESSED_DATA_DIR

class MLDataset:

    FEATURES = [
        "title",
        "text",
        "helpful_vote",
        "verified_purchase",
        "review_length",
        "word_count",
    ]

    TARGET = "label"

    def __init__(
        self,
        split: str,
        sample_size: int | None = None,
    ) -> None:

        self.split = split
        self.sample_size = sample_size

    def load(self):

        path = PROCESSED_DATA_DIR / f"{self.split}.parquet"

        print(f"\nLoading {self.split} dataset...")

        df = pl.read_parquet(path)

        if self.sample_size is not None:

            sample_size = min(self.sample_size, len(df))

            df = df.sample(
                n=sample_size,
                seed=42,
                shuffle=True,
            )

            print(f"Using sample size : {sample_size:,}")

        print(f"Samples : {len(df):,}")

        X = df.select(self.FEATURES)

        y = df[self.TARGET]

        return X, y
def main():

    dataset = MLDataset("train")

    X, y = dataset.load()

    print("\nFeatures")

    print(X.head())

    print("\nLabels")

    print(y.head())


if __name__ == "__main__":
    main()
