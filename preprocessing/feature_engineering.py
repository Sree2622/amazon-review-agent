"""
Creates additional features for classical machine learning models.
"""

import polars as pl


class FeatureEngineer:
    
    def engineer(self, df: pl.DataFrame) -> pl.DataFrame:
        print("\nEngineering Features...\n")

        df = df.with_columns(

            # Number of characters in review
            pl.col("text")
            .str.len_chars()
            .alias("review_length"),

            # Number of words
            pl.col("text")
            .str.split(" ")
            .list.len()
            .alias("word_count"),

            # Boolean -> Integer
            pl.col("verified_purchase")
            .cast(pl.Int8)
            .alias("verified_purchase"),

        )

        print("Feature engineering completed.\n")

        return df