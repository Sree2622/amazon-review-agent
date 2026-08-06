"""
Convert review text into TF-IDF and numerical features.
"""

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import polars as pl


class ReviewVectorizer:
    """
    Vectorizes review text using TF-IDF and combines it with
    engineered numerical features.
    """

    def __init__(
        self,
        max_features: int = 10000,
    ) -> None:

        self.title_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )

        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )

    def fit_transform(
        self,
        X: pl.DataFrame,
    ):
        """
        Fit TF-IDF vectorizers and transform the dataset.

        Parameters
        ----------
        X : pl.DataFrame

        Returns
        -------
        scipy sparse matrix
        """

        print("\nVectorizing training dataset...\n")

        title_features = self.title_vectorizer.fit_transform(
            X["title"].to_list()
        )

        text_features = self.text_vectorizer.fit_transform(
            X["text"].to_list()
        )

        numeric_features = np.column_stack(
            [
                X["helpful_vote"].to_numpy(),
                X["verified_purchase"].to_numpy(),
                X["review_length"].to_numpy(),
                X["word_count"].to_numpy(),
            ]
        )

        features = hstack(
            [
                title_features,
                text_features,
                numeric_features,
            ]
        )

        print("Vectorization completed.\n")

        return features

    def transform(
        self,
        X: pl.DataFrame,
    ):
        """
        Transform unseen data using trained TF-IDF models.

        Parameters
        ----------
        X : pl.DataFrame

        Returns
        -------
        scipy sparse matrix
        """

        title_features = self.title_vectorizer.transform(
            X["title"].to_list()
        )

        text_features = self.text_vectorizer.transform(
            X["text"].to_list()
        )

        numeric_features = np.column_stack(
            [
                X["helpful_vote"].to_numpy(),
                X["verified_purchase"].to_numpy(),
                X["review_length"].to_numpy(),
                X["word_count"].to_numpy(),
            ]
        )

        return hstack(
            [
                title_features,
                text_features,
                numeric_features,
            ]
        )

if __name__ == "__main__":

    from ml.dataset import MLDataset

    dataset = MLDataset(split="train")

    X, y = dataset.load()

    vectorizer = ReviewVectorizer()

    X_vectorized = vectorizer.fit_transform(X)

    print("Vectorized Shape :", X_vectorized.shape)
