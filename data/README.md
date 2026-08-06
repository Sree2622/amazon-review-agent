# Data

This directory contains the **Amazon Reviews 2023 – All Beauty** dataset in both its **raw** and **processed** forms.

## Directory Structure

```text
data/
├── raw/
│   ├── All_Beauty.jsonl
│   └── meta_All_Beauty.jsonl
│
└── processed/
    ├── All_Beauty.parquet
    ├── All_Beauty_products.parquet
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

## Raw Data

The `raw/` directory contains the original Amazon Reviews 2023 dataset.

### `All_Beauty.jsonl`

Contains customer reviews for beauty-related products.

Example:

```json
{
  "rating": 5.0,
  "title": "Such a lovely scent but not overpowering.",
  "text": "This spray is really nice...",
  "asin": "B00YQ6X8EO",
  "parent_asin": "B00YQ6X8EO",
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "timestamp": 1588687728923,
  "helpful_vote": 0,
  "verified_purchase": true
}
```

### `meta_All_Beauty.jsonl`

Contains metadata for each product, including product title, store, rating, price, description, categories, and images.

Example:

```json
{
  "title": "Howard LC0008 Leather Conditioner",
  "average_rating": 4.8,
  "rating_number": 10,
  "store": "Howard Products",
  "price": null,
  "parent_asin": "B01CUPMQZE"
}
```

---

## Processed Data

The `processed/` directory contains cleaned datasets stored in **Apache Parquet**, a column-oriented format that is faster and more efficient for analytics and machine learning.

### `All_Beauty.parquet`

Processed version of `All_Beauty.jsonl` with additional features generated during preprocessing.

Example:

```text
rating  title       text   ... word_count sentiment label
5.0     a must buy  ...          11      Positive     2
```

### `All_Beauty_products.parquet`

Processed version of `meta_All_Beauty.jsonl`.

Columns:

```text
parent_asin
title
store
price
average_rating
rating_number
features
description
categories
images
```

### Dataset Splits

The repository also contains pre-generated dataset splits for future sentiment analysis experiments:

* `train.parquet`
* `validation.parquet`
* `test.parquet`
