"""
Sentences - Text segmentation and tokenization utilities.

A Python package for splitting text into sentences and paragraphs with exact
position tracking and token range extraction for LLM tokenizers.
"""

from .sentence_splitter import split_text_to_sentences
from .para_splitter import split_text_to_paragraphs
from .token_ranges import get_token_ranges, validate_token_ranges

__version__ = "0.1.0"

__all__ = [
    "split_text_to_sentences",
    "split_text_to_paragraphs",
    "get_token_ranges",
    "validate_token_ranges",
]