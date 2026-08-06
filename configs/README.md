# Configurations

This directory contains all configuration files used across the project. Keeping project settings in one place makes them easier to manage and maintain.

| File              | Description                                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`       | Defines project directory paths (e.g., `DATA_DIR`, `PROCESSED_DATA_DIR`, `MODEL_DIR`).                                                        |
| `constants.py`    | Stores project-wide constants such as directory names, supported ML models, and train/validation/test split ratios.                           |
| `lora_configs.py` | Contains LoRA fine-tuning configurations, including quantization settings, target modules, batch size, epochs, and other training parameters. |
