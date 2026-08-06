"""
Evaluate trained machine-learning models.
"""

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from configs.config import ML_MODEL_DIR
from ml.dataset import MLDataset
from ml.vectorizer import ReviewVectorizer


class ModelEvaluator:
    """
    Evaluate trained machine learning models.
    """

    def __init__(self) -> None:
        """
        Initialize evaluator.
        """
        self.test_dataset = MLDataset("test")

    @staticmethod
    def calculate_metrics(
        y_true,
        y_pred,
    ) -> dict:
        """
        Calculate evaluation metrics.

        Parameters
        ----------
        y_true
            Ground truth labels.
        y_pred
            Predicted labels.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """

        return {
            "Accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "Precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "Recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "F1 Score": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate all trained ML models.
        """

        print("\nLoading test dataset...\n")

        X_test, y_test = self.test_dataset.load()

        print("Loading vectorizer...\n")

        vectorizer: ReviewVectorizer = joblib.load(
            ML_MODEL_DIR / "tfidf_vectorizer.joblib"
        )

        X_test = vectorizer.transform(X_test)

        results = []

        model_names = [
            "logistic_regression",
            "random_forest",
            "xgboost",
        ]

        for model_name in model_names:

            print(f"\nEvaluating {model_name}...")

            model = joblib.load(
                ML_MODEL_DIR / f"{model_name}.joblib"
            )

            predictions = model.predict(X_test)

            metrics = self.calculate_metrics(
                y_test,
                predictions,
            )

            print(
                classification_report(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            )

            results.append(
                {
                    "Model": model_name,
                    **metrics,
                }
            )

        return pd.DataFrame(results)


def main() -> None:

    evaluator = ModelEvaluator()

    results = evaluator.evaluate()

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================\n")

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
