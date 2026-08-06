"""
Train classical machine-learning models.
"""

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from configs.config import ML_MODEL_DIR
from ml.dataset import MLDataset
from ml.vectorizer import ReviewVectorizer


class ModelTrainer:
    """
    Train and save classical machine learning models.
    """

    def __init__(self) -> None:
        self.vectorizer = ReviewVectorizer()

        self.models = {
            "logistic_regression": LogisticRegression(
                solver="saga",
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "xgboost": XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=-1,
            ),
        }

    def train(self) -> dict:
        """
        Train all ML models.
        """

        print("\nLoading training dataset...\n")

        train_dataset = MLDataset(
            split="train",
            sample_size=50000,
        )
        X_train, y_train = train_dataset.load()

        print("\nVectorizing training data...\n")

        X_train = self.vectorizer.fit_transform(X_train)

        trained_models = {}

        for name, model in self.models.items():
            print(f"\n{'=' * 60}")
            print(f"Training: {name}")
            print(f"{'=' * 60}\n")

            model.fit(X_train, y_train)

            trained_models[name] = model

            model_path = ML_MODEL_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)

            print(f"Saved model -> {model_path}")

        vectorizer_path = ML_MODEL_DIR / "tfidf_vectorizer.joblib"
        joblib.dump(self.vectorizer, vectorizer_path)

        print(f"\nSaved vectorizer -> {vectorizer_path}")
        print("\nTraining Completed Successfully.\n")

        return trained_models


def main() -> None:
    trainer = ModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
