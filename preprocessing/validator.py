"""
Validates the Amazon Review dataset before preprocessing.
"""

import polars as pl


class DatasetValidator:

    REQUIRED_COLUMNS = [
        "rating",
        "title",
        "text",
        "user_id",
        "parent_asin",
        "timestamp",
    ]

    MIN_RATING = 1.0
    MAX_RATING = 5.0

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:

        print("\nValidating Dataset...\n")
        original_rows = df.height


        # Check  Columns
        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )


        # Remove Missing Title
        missing_title = (
            df.filter(
                pl.col("title").is_null() |
                (pl.col("title").str.strip_chars() == "")
            ).height
        )

        df = df.filter(
            pl.col("title").is_not_null() &
            (pl.col("title").str.strip_chars() != "")
        )


        # Remove Missing Review
        missing_text = (
            df.filter(
                pl.col("text").is_null() |
                (pl.col("text").str.strip_chars() == "")
            ).height
        )

        df = df.filter(
            pl.col("text").is_not_null() &
            (pl.col("text").str.strip_chars() != "")
        )


        # Remove Invalid Ratings
        invalid_rating = (
            df.filter(
                (pl.col("rating") < self.MIN_RATING) |
                (pl.col("rating") > self.MAX_RATING)
            ).height
        )

        df = df.filter(
            (pl.col("rating") >= self.MIN_RATING) &
            (pl.col("rating") <= self.MAX_RATING)
        )


        # Remove Duplicate Reviews
        before_duplicates = df.height

        df = df.unique(
            subset=[
                "user_id",
                "parent_asin",
                "timestamp",
            ],
            keep="first",
        )

        duplicate_reviews = before_duplicates - df.height


        # Report
        print("=" * 45)
        print("Validation Report")
        print("=" * 45)

        print(f"Original Rows        : {original_rows:,}")
        print(f"Missing Titles       : {missing_title:,}")
        print(f"Missing Reviews      : {missing_text:,}")
        print(f"Invalid Ratings      : {invalid_rating:,}")
        print(f"Duplicate Reviews    : {duplicate_reviews:,}")

        print("-" * 45)

        print(f"Remaining Rows       : {df.height:,}")

        print("=" * 45)

        return df
