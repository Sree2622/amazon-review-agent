"""
Evaluate the base and LoRA LLMs using the same test dataset.
"""

from __future__ import annotations

import pandas as pd

from sklearn.metrics import classification_report

from ml.dataset import MLDataset
from ml.evaluator import ModelEvaluator

from llm.base_model import BaseLLM
from llm.lora_model import LoRAModel


class LLMEvaluator:
    """
    Generic evaluator for LLM models.
    """

    def __init__(
        self,
        sample_size: int | None = 500,
    ) -> None:
        """
        Initialize evaluator.

        Parameters
        ----------
        sample_size:
            Number of test samples.
            None uses full dataset.
        """

        self.dataset = MLDataset(
            "test",
            sample_size=sample_size,
        )


    def evaluate(
        self,
        model,
        model_name: str,
    ) -> pd.DataFrame:
        """
        Evaluate an LLM.

        Parameters
        ----------
        model:
            Object implementing predict_batch()

        model_name:
            Display name.
        """

        print(
            "\nLoading test dataset...\n"
        )

        X_test, y_test = (
            self.dataset.load()
        )


        reviews = (
            X_test.to_dicts()
        )


        print(
            f"Running inference on {len(reviews):,} reviews...\n"
        )


        predictions = (
            model.predict_batch(
                reviews
            )
        )


        print(
            "\nClassification Report\n"
        )

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )


        metrics = (
            ModelEvaluator.calculate_metrics(
                y_test,
                predictions,
            )
        )


        return pd.DataFrame(
            [
                {
                    "Model": model_name,
                    **metrics,
                }
            ]
        )



def main() -> None:
    """
    Compare Base LLM and LoRA LLM.
    """


    evaluator = LLMEvaluator(
        sample_size=100
    )


    results = []


    # base Qwen

    base_model = BaseLLM()


    results.append(
        evaluator.evaluate(
            base_model,
            "Qwen2.5-3B-Instruct",
        )
    )


    # LoRA Qwen

    lora_model = LoRAModel()


    results.append(
        evaluator.evaluate(
            lora_model,
            "Qwen2.5-3B-LoRA",
        )
    )


    final_results = pd.concat(
        results,
        ignore_index=True,
    )


    print(
        "\n=============================="
    )

    print(
        "LLM MODEL COMPARISON"
    )

    print(
        "==============================\n"
    )


    print(
        final_results.to_string(
            index=False
        )
    )



if __name__ == "__main__":

    main()
