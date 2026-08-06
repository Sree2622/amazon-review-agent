# Machine Learning

This directory contains classical three-class sentiment classifiers trained on TF-IDF review text and engineered numerical features.

## Files

| File | Description |
| --- | --- |
| `dataset.py` | Loads processed train, validation, and test splits. |
| `vectorizer.py` | Builds TF-IDF and numerical features. |
| `trainer.py` | Trains and saves the ML models. |
| `evaluator.py` | Evaluates saved models on the test split. |
| `compare.py` | Runs training and displays the comparison. |

## Results

All classical models were evaluated on the 69,355-review test split.

| Model | Accuracy | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 52.13% | 73.68% | 52.13% | 58.32% |
| Random Forest | **83.37%** | **81.48%** | **83.37%** | **81.73%** |
| XGBoost | 81.86% | 79.88% | 81.86% | 78.55% |

![ML accuracy in the full model comparison](../docs/images/model_accuracy_comparison.png)

Random Forest delivered the best recorded classical-model accuracy and F1 score. The class-level reports also show that neutral reviews are the most difficult class for the evaluated ML models.

## Models

* Logistic Regression
* Random Forest
* XGBoost

## Commands

```bash
python -m ml.trainer
python -m ml.evaluator
python -m ml.compare
```

Saved models and the TF-IDF vectorizer are stored in `models/ml/`.
