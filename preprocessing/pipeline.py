"""
Runs the complete preprocessing workflow.
"""

import polars as pl

from preprocessing.dataset_loader import DatasetLoader
from preprocessing.validator import DatasetValidator
from preprocessing.clean_reviews import ReviewCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.sentiment_labels import SentimentLabeler
from configs.config import PROCESSED_DATA_DIR


class PreprocessingPipeline:

    def __init__(self, category: str = "All_Beauty") -> None:

        self.category = category
        self.loader = DatasetLoader(category)
        self.validator = DatasetValidator()
        self.cleaner = ReviewCleaner()
        self.engineer = FeatureEngineer()
        self.labeler = SentimentLabeler()

    def run(self) -> pl.DataFrame:
        print("\n" + "=" * 60)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 60)

        # Step 1
        df = self.loader.load()

        # Step 2
        df = self.validator.validate(df)

        # Step 3
        df = self.cleaner.clean(df)

        # Step 4
        df = self.engineer.engineer(df)

        # Step 5
        df = self.labeler.generate(df)

        print("\nPreprocessing completed successfully.")
        print(f"Final Dataset Shape : {df.shape}")
        print("=" * 60)

        output_path = PROCESSED_DATA_DIR / f"{self.category}.parquet"

        print("\nSaving processed dataset...")
        df.write_parquet(output_path)

        print(f"Saved to: {output_path}")

        return df


def main() -> None:
    pipeline = PreprocessingPipeline()

    df = pipeline.run()

    print("\nProcessed Dataset Preview\n")

    print(
        df.select(
            [
                "rating",
                "sentiment",
                "label",
                "review_length",
                "word_count",
                "helpful_vote",
                "verified_purchase",
            ]
        ).head(10)
    )


if __name__ == "__main__":
    main()