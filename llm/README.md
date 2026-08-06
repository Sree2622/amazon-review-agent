# LLM

This directory contains the Qwen-based sentiment-classification workflow. It supports base-model inference, LoRA fine-tuning, and evaluation against the processed test dataset.

## Files

| File | Description |
| --- | --- |
| `prompt_builder.py` | Builds sentiment-classification prompts from review features. |
| `base_model.py` | Loads the quantized base Qwen model and predicts sentiment labels. |
| `lora_model.py` | Loads the base Qwen model with the trained LoRA adapter. |
| `train_lora.py` | Prepares the dataset and fine-tunes the Qwen model with QLoRA. |
| `evaluate.py` | Evaluates and compares the base Qwen and LoRA models. |

## Sentiment Labels

```text
0 = Negative
1 = Neutral
2 = Positive
```

## Commands

```bash
python -m llm.train_lora
python -m llm.evaluate
```

LoRA training uses settings defined in `configs/lora_config.py` and writes its adapter to `models/llm/`.
