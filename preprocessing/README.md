# Preprocessing

This directory contains the preprocessing modules used to prepare the Amazon Review dataset for model training.

## Files

| File                  | Description                                                                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_loader.py`   | Loads Amazon Review datasets stored as JSONL files. Includes utilities to load, preview, and display dataset information.                                |
| `validator.py`        | Validates the dataset by removing missing values, invalid ratings, and duplicate reviews. Generates a validation report.                                 |
| `cleaner_review.py`   | Cleans review text by converting it to lowercase, removing URLs, tabs, extra spaces, and other unnecessary characters while preserving semantic meaning. |
| `feature_engineer.py` | Creates additional features such as review length and word count for classical machine learning models.                                                  |
| `sentiment_label.py`  | Generates sentiment labels from star ratings (`<3` → Negative, `3` → Neutral, `>3` → Positive).                                                          |
| `pipeline.py`         | Executes the complete preprocessing pipeline in the correct order.                                                                                       |
| `splitter.py`         | Splits the processed dataset into Train, Validation, and Test sets (80:10:10) and saves them as Parquet files in the `processed` directory.              |
| `metadata_loader.py`  | Builds the processed product catalog used by the Review Agent.                                                                                           |

## Pipeline Workflow

```text
Load Dataset
      ↓
Validate Dataset
      ↓
Clean Reviews
      ↓
Engineer Features
      ↓
Generate Sentiment Labels
      ↓
Save Processed Dataset
      ↓
Train / Validation / Test Split
```

## Validation

The validation step performs the following checks:

* Removes missing values.
* Removes invalid ratings.
* Removes duplicate reviews.
* Generates a validation report.

## Example Output

```text
Preprocessing completed successfully.
Final Dataset Shape : (693547, 14)

Saving processed dataset...
Saved to: data/processed/All_Beauty.parquet

Processed Dataset Preview

shape: (10, 7)
┌────────┬───────────┬───────┬───────────────┬────────────┬──────────────┬───────────────────┐
│ rating ┆ sentiment ┆ label ┆ review_length ┆ word_count ┆ helpful_vote ┆ verified_purchase │
├────────┼───────────┼───────┼───────────────┼────────────┼──────────────┼───────────────────┤
│ 5.0    ┆ Positive  ┆ 2     ┆ 135           ┆ 21         ┆ 0            ┆ 1                 │
│ 2.0    ┆ Negative  ┆ 0     ┆ 63            ┆ 12         ┆ 0            ┆ 1                 │
│ 3.0    ┆ Neutral   ┆ 1     ┆ 132           ┆ 27         ┆ 1            ┆ 1                 │
│ ...    ┆ ...       ┆ ...   ┆ ...           ┆ ...        ┆ ...          ┆ ...               │
└────────┴───────────┴───────┴───────────────┴────────────┴──────────────┴───────────────────┘
```
