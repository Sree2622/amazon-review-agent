"""
Load a Qwen base model with a trained LoRA adapter.
"""

from __future__ import annotations

import logging
import re

import torch

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from configs import lora_config as cfg
from llm.prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)


class LoRAModel:
    """
    Qwen model with LoRA adapter for sentiment classification.
    """

    def __init__(self) -> None:
        """
        Load base model and LoRA adapter.
        """

        logger.info(
            "Loading tokenizer..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_NAME,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )


        logger.info(
            "Loading base model..."
        )


        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=(
                cfg.BNB_4BIT_QUANT_TYPE
            ),
            bnb_4bit_use_double_quant=(
                cfg.BNB_4BIT_USE_DOUBLE_QUANT
            ),
            bnb_4bit_compute_dtype=(
                torch.float16
            ),
        )


        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )


        logger.info(
            "Loading LoRA adapter..."
        )


        self.model = PeftModel.from_pretrained(
            base_model,
            cfg.LORA_MODEL_DIR,
        )


        self.model.eval()


        logger.info(
            "LoRA model loaded successfully."
        )


    def predict(
        self,
        review: dict,
    ) -> int:
        """
        Predict sentiment for one review.

        Returns:
            0 = Negative
            1 = Neutral
            2 = Positive
        """

        prompt = PromptBuilder.build(
            review
        )


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.MAX_LENGTH,
        )


        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value in inputs.items()
        }


        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )


        generated_tokens = outputs[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]


        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()


        return self._parse_prediction(
            response
        )


    def predict_batch(
        self,
        reviews: list[dict],
    ) -> list[int]:
        """
        Predict multiple reviews.
        """

        predictions = []

        for review in reviews:

            predictions.append(
                self.predict(review)
            )

        return predictions



    @staticmethod
    def _parse_prediction(
        output: str,
    ) -> int:
        """
        Convert model output into sentiment label.
        """

        match = re.search(
            r"[012]",
            output,
        )

        if match:

            return int(
                match.group()
            )


        logger.warning(
            "Unable to parse output: %s",
            output,
        )

        return 1
