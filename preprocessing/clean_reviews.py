"""
Responsible for cleaning review text while preserving
its semantic meaning.
"""

import re

import polars as pl


class ReviewCleaner:

    def __init__(self) -> None:

        self.url_pattern = re.compile(r"http\S+|www\S+")

        self.email_pattern = re.compile(
            r"\S+@\S+\.\S+"
        )

        self.html_pattern = re.compile(
            r"<.*?>"
        )

        self.space_pattern = re.compile(
            r"\s+"
        )

    def clean_text(self, text: str) -> str:

        if text is None:
            return ""

        # lowercase
        text = text.lower()

        # remove html
        text = self.html_pattern.sub(" ", text)

        # remove urls
        text = self.url_pattern.sub(" ", text)

        # remove emails
        text = self.email_pattern.sub(" ", text)

        # remove tabs/newlines
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        # remove extra spaces
        text = self.space_pattern.sub(" ", text)

        return text.strip()


    def clean(self, df: pl.DataFrame) -> pl.DataFrame:

        print("\nCleaning Reviews...\n")

        df = df.with_columns(

            pl.col("title")
            .map_elements(
                self.clean_text,
                return_dtype=pl.String,
            )
            .alias("title"),

            pl.col("text")
            .map_elements(
                self.clean_text,
                return_dtype=pl.String,
            )
            .alias("text"),
        )

        print("Review cleaning completed.\n")

        return df

