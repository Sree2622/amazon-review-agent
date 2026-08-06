"""
Fine-tune Qwen2.5-3B-Instruct with QLoRA.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from configs import lora_config as cfg
from llm.prompt_builder import PromptBuilder
from ml.dataset import MLDataset


# logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# dtype helper

def _resolve_compute_dtype() -> torch.dtype:
    """
    Resolves cfg.COMPUTE_DTYPE (a string) into an actual torch dtype,
    so quantization / model dtype / trainer dtype all stay in sync.
    """

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    dtype = mapping.get(cfg.COMPUTE_DTYPE)

    if dtype is None:
        raise ValueError(
            f"Unsupported COMPUTE_DTYPE in config: {cfg.COMPUTE_DTYPE!r}"
        )

    return dtype


# dataset builder

class LoRADatasetBuilder:
    """
    Converts parquet dataset into instruction format.
    """

    def __init__(self) -> None:

        self.dataset = MLDataset(
            "train",
            sample_size=500,
        )


    def build(self) -> Dataset:
        """
        Create HuggingFace dataset for SFT training.

        Uses TRL's prompt-completion format (separate "prompt" and
        "completion" columns) rather than a single concatenated
        "text" field. As of TRL 1.x, SFTTrainer computes loss on
        the completion tokens only by default when the dataset is
        in this format (completion_only_loss=True in SFTConfig) --
        this is what actually restricts training signal to the
        label token instead of the whole prompt.
        (The older DataCollatorForCompletionOnlyLM approach was
        removed in this TRL version.)
        """

        X, y = self.dataset.load()

        rows = X.to_dicts()
        labels = y.to_list()

        samples = []

        for review, label in tqdm(
            zip(rows, labels),
            total=len(rows),
            desc="Building training prompts",
        ):

            prompt = PromptBuilder.build(review)

            samples.append(
                {
                    "prompt": prompt,
                    "completion": f" {label}",
                }
            )


        logger.info(
            "Training samples created: %s",
            len(samples),
        )


        return Dataset.from_list(samples)



# QLoRA model loader

class QLoRAModelLoader:
    """
    Loads Qwen model with 4-bit quantization.
    """


    def load(self):

        logger.info(
            "Loading tokenizer..."
        )


        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_NAME,
            trust_remote_code=True,
        )


        tokenizer.pad_token = tokenizer.eos_token


        logger.info(
            "Loading base model..."
        )

        compute_dtype = _resolve_compute_dtype()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type=(
                cfg.BNB_4BIT_QUANT_TYPE
            ),

            bnb_4bit_use_double_quant=(
                cfg.BNB_4BIT_USE_DOUBLE_QUANT
            ),

        # use the configured compute dtype
            bnb_4bit_compute_dtype=compute_dtype,
        )


        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_NAME,

            quantization_config=(
                quantization_config
            ),

            device_map="auto",

        # use the configured torch dtype
            torch_dtype=compute_dtype,

            trust_remote_code=True,
        )


        model.config.use_cache = False


        model = prepare_model_for_kbit_training(
            model
        )


        lora_config = LoraConfig(

            r=cfg.LORA_R,

            lora_alpha=cfg.LORA_ALPHA,

            lora_dropout=cfg.LORA_DROPOUT,

            bias=cfg.BIAS,

            task_type=TaskType.CAUSAL_LM,

            target_modules=(
                cfg.TARGET_MODULES
            ),
        )


        model = get_peft_model(
            model,
            lora_config,
        )


        model.print_trainable_parameters()


        return model, tokenizer



# trainer

class LoRATrainer:
    """
    Controls LoRA training.
    """


    def __init__(self) -> None:

        self.dataset_builder = (
            LoRADatasetBuilder()
        )

        self.loader = (
            QLoRAModelLoader()
        )



    def train(self) -> None:

        dataset = (
            self.dataset_builder.build()
        )


        model, tokenizer = (
            self.loader.load()
        )


        training_args = SFTConfig(

            output_dir=str(
                cfg.LORA_MODEL_DIR
            ),


            num_train_epochs=(
                cfg.NUM_EPOCHS
            ),


            per_device_train_batch_size=(
                cfg.TRAIN_BATCH_SIZE
            ),


            gradient_accumulation_steps=(
                cfg.GRADIENT_ACCUMULATION_STEPS
            ),


            learning_rate=(
                cfg.LEARNING_RATE
            ),


            warmup_ratio=(
                cfg.WARMUP_RATIO
            ),


            weight_decay=(
                cfg.WEIGHT_DECAY
            ),


            logging_steps=(
                cfg.LOGGING_STEPS
            ),


            save_steps=(
                cfg.SAVE_STEPS
            ),


            max_length=(
                cfg.MAX_LENGTH
            ),

            bf16=True,
            fp16=False,

            report_to="none",


            seed=(
                cfg.SEED
            ),

            # calculate loss from the completion tokens
            completion_only_loss=True,
        )



        trainer = SFTTrainer(

            model=model,

            args=training_args,

            train_dataset=dataset,

            processing_class=tokenizer,
        )



        logger.info(
            "Starting LoRA training..."
        )


        trainer.train()



        logger.info(
            "Saving LoRA adapter..."
        )


        trainer.model.save_pretrained(
            cfg.LORA_MODEL_DIR
        )


        tokenizer.save_pretrained(
            cfg.LORA_MODEL_DIR
        )


        logger.info(
            "Training complete."
        )



# main

def main() -> None:

    trainer = LoRATrainer()

    trainer.train()



if __name__ == "__main__":

    main()
