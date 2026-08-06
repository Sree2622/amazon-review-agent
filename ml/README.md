# Machine Learning

This directory contains the classical machine-learning sentiment-classification workflow. It uses TF-IDF features from review text together with engineered numerical features.

## Files

| File | Description |
| --- | --- |
| `dataset.py` | Loads processed train, validation, and test dataset splits. |
| `vectorizer.py` | Converts review text and numerical fields into model features. |
| `trainer.py` | Trains and saves logistic regression, random forest, and XGBoost models. |
| `evaluator.py` | Evaluates saved models on the test dataset. |
| `compare.py` | Runs training and displays the model comparison. |

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
