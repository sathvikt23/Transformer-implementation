import re
from typing import Dict, Any


class HeuristicFilter:
    """
    Stage 1: Language ID
    Stage 4: Rule-based Heuristic Quality Filter
    Stage 6: Custom Task Filtering (Syntax & Density)
    """

    def __init__(self, config: Dict[str, Any]):
        self.min_word_count = config.get("min_word_count", 20)
        self.max_word_count = config.get("max_word_count", 5000)
        self.min_alpha_ratio = config.get("min_alpha_ratio", 0.60)
        self.max_punc_ratio = config.get("max_punc_ratio", 0.30)
        self.max_digit_ratio = config.get("max_digit_ratio", 0.40)
        self.max_uppercase_ratio = config.get("max_uppercase_ratio", 0.60)
        self.min_unique_word_ratio = config.get("min_unique_word_ratio", 0.30)
        self.min_sentence_length = config.get("min_sentence_length", 5)

    def is_english(self, text: str) -> bool:
        """
        Stage 1: Simple & robust ASCII/English alphabet ratio heuristic.
        (Can be backed by lingua-py when available).
        """
        if len(text) < 30:
            return False
        english_chars = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isspace() or c in ".,!?'\"-"))
        return (english_chars / len(text)) >= 0.80

    def passes_heuristic_quality(self, text: str) -> bool:
        """Stage 4: Word count, alphabetic ratio, max punctuation checks."""
        words = text.split()
        word_count = len(words)

        # Word count range check (20 - 5000 words)
        if not (self.min_word_count <= word_count <= self.max_word_count):
            return False

        total_chars = len(text)
        if total_chars == 0:
            return False

        alpha_chars = sum(1 for c in text if c.isalpha())
        punc_chars = sum(1 for c in text if c in ".,;:!?\"'()[]{}<>-/*+=@#$%^&*")

        # Alphabetic ratio (>= 60%)
        if (alpha_chars / total_chars) < self.min_alpha_ratio:
            return False

        # Punctuation ratio (<= 30%)
        if (punc_chars / total_chars) > self.max_punc_ratio:
            return False

        return True

    def passes_custom_task_filter(self, text: str) -> bool:
        """Stage 6: Digit ratio, uppercase ratio, vocabulary diversity."""
        words = text.split()
        if not words:
            return False

        total_chars = len(text)
        digits = sum(1 for c in text if c.isdigit())
        uppercases = sum(1 for c in text if c.isupper())

        # Digit ratio (<= 40%)
        if (digits / total_chars) > self.max_digit_ratio:
            return False

        # Uppercase ratio (<= 60%)
        if (uppercases / total_chars) > self.max_uppercase_ratio:
            return False

        # Unique word ratio (vocabulary diversity >= 30%)
        unique_words = set(w.lower() for w in words)
        if (len(unique_words) / len(words)) < self.min_unique_word_ratio:
            return False

        # Average sentence word count (>= 5 words)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if sentences:
            avg_sent_len = len(words) / len(sentences)
            if avg_sent_len < self.min_sentence_length:
                return False

        return True
