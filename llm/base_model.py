"""
Load the base Qwen model for sentiment classification.
"""

from __future__ import annotations

import logging
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llm.prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)


class BaseLLM:
    """
    Base Qwen sentiment classifier.
    """

    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self) -> None:
        """
        Load tokenizer and quantized model.
        """

        logger.info("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading 4-bit model...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.model.eval()

        logger.info("Model loaded successfully.")

    def predict(self, review: dict) -> int:
        """
        Predict sentiment for a single review.

        Parameters
        ----------
        review : dict
            Review features.

        Returns
        -------
        int
            0 = Negative
            1 = Neutral
            2 = Positive
        """

        prompt = PromptBuilder.build(review)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        return self._parse_prediction(text)

    def predict_batch(
        self,
        reviews: list[dict],
    ) -> list[int]:
        """
        Predict sentiments for multiple reviews.
        """

        return [self.predict(review) for review in reviews]

    def respond(
        self,
        system_message: str,
        user_message: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a grounded chat response with the Qwen instruction model."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_prediction(output: str) -> int:
        """
        Parse the model output.

        Falls back to Neutral (1) if parsing fails.
        """

        match = re.search(r"[012]", output)

        if match:
            return int(match.group())

        logger.warning(
            "Unable to parse prediction: %s",
            output,
        )

        return 1
