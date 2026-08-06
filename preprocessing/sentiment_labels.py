"""
Converts Amazon star ratings into sentiment labels for machine learning and LLM fine-tuning.
"""

import polars as pl


class SentimentLabeler:

    @staticmethod
    def rating_to_sentiment(rating: float) -> str:
        
        if rating <= 2:
            return "Negative"

        if rating == 3:
            return "Neutral"

        return "Positive"

    @staticmethod
    def rating_to_label(rating: float) -> int:

        if rating <= 2:
            return 0

        if rating == 3:
            return 1

        return 2

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:


        print("\nGenerating Sentiment Labels...\n")

        df = df.with_columns(

            pl.col("rating")
            .map_elements(
                self.rating_to_sentiment,
                return_dtype=pl.String
            )
            .alias("sentiment"),

            pl.col("rating")
            .map_elements(
                self.rating_to_label,
                return_dtype=pl.Int8
            )
            .alias("label"),
        )

        print("Sentiment labels generated.\n")

        return df


if __name__ == "__main__":

    from preprocessing.dataset_loader import DatasetLoader
    from preprocessing.validator import DatasetValidator
    from preprocessing.clean_reviews import ReviewCleaner
    from preprocessing.feature_engineering import FeatureEngineer

    loader = DatasetLoader()
    validator = DatasetValidator()
    cleaner = ReviewCleaner()
    engineer = FeatureEngineer()
    labeler = SentimentLabeler()

    df = loader.load()
    df = validator.validate(df)
    df = cleaner.clean(df)
    df = engineer.engineer(df)
    df = labeler.generate(df)

    print(
        df.select(
            [
                "rating",
                "sentiment",
                "label",
            ]
        ).head(10)
    )