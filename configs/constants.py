"""
Values of Contants used throughout the repo
"""

from pathlib import Path


# Project Information
PROJECT_NAME: str = "Amazon Review Agentic AI System"
VERSION: str = "1.0.0"


# Dataset
DATASET_NAME: str = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORY = "raw_review_All_Beauty"


# Random Seed
RANDOM_STATE: int = 42


# Train / Validation / Test
TRAIN_SIZE: float = 0.80
VALIDATION_SIZE: float = 0.10
TEST_SIZE: float = 0.10


# Directory Names
RAW_DIR = "raw"
PROCESSED_DIR = "processed"
TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
TEST_DIR = "test"


# Supported ML Models
SUPPORTED_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "XGBoost",
]