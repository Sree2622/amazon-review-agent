"""
Build prompts for LLM-based sentiment classification.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl


class PromptBuilder:
    """
    Builds prompts for sentiment classification.
    """

    SYSTEM_PROMPT = (
        "You are an expert sentiment classification model.\n\n"
        "Your task is to classify Amazon product reviews.\n\n"
        "Return ONLY one integer.\n\n"
        "0 = Negative\n"
        "1 = Neutral\n"
        "2 = Positive\n\n"
        "Do not explain your answer."
    )

    @classmethod
    def build(cls, review: Mapping | pl.Series | dict) -> str:
        """
        Build a prompt from one review.

        Parameters
        ----------
        review:
            Mapping-like object containing review features.

        Returns
        -------
        str
            Formatted prompt.
        """

        title = cls._get(review, "title")
        text = cls._get(review, "text")
        helpful_vote = cls._get(review, "helpful_vote")
        verified_purchase = cls._get(review, "verified_purchase")
        review_length = cls._get(review, "review_length")
        word_count = cls._get(review, "word_count")

        prompt = f"""{cls.SYSTEM_PROMPT}

Review Information

Title:
{title}

Review:
{text}

Helpful Votes:
{helpful_vote}

Verified Purchase:
{verified_purchase}

Review Length:
{review_length}

Word Count:
{word_count}

Answer:
"""

        return prompt

    @staticmethod
    def _get(review: Mapping | pl.Series | dict, key: str) -> str:
        """
        Safely retrieve a value from a review object.
        """

        try:
            value = review[key]
        except Exception:
            value = ""

        if value is None:
            return ""

        return str(value)
