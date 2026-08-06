"""
Compare classical machine-learning models.
"""

from ml.trainer import ModelTrainer
from ml.evaluator import ModelEvaluator


def main():

    print("\nTraining Classical ML Models...\n")

    trainer = ModelTrainer()

    trainer.train()

    print("\nEvaluating Models...\n")

    evaluator = ModelEvaluator()

    results = evaluator.evaluate()

    print("\n===============================")
    print("MODEL COMPARISON")
    print("===============================\n")

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
