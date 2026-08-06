"""
LoRA Configuration

"""
from pathlib import Path
from configs.config import MODEL_DIR


# Base Model
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


# Quantization
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
COMPUTE_DTYPE = "bfloat16"


# LoRA

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BIAS = "none"
TASK_TYPE = "CAUSAL_LM"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# Training
MAX_LENGTH = 1024
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 25
SAVE_STEPS = 500
EVALUATION_STEPS = 500
SEED = 42


# Output
LORA_MODEL_DIR: Path = MODEL_DIR / "qwen_lora"
LORA_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
