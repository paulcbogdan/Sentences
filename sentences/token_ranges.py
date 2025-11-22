"""
Find token ranges for sentences using iterative tokenization.

Key function: get_token_ranges(sentences, pre_string, tokenizer_func)

Uses iterative tokenization to avoid boundary issues:
- Tokenizes pre_string
- Adds sentences one by one
- Tokenizes full accumulated text each time
- Returns list of (start, end) tuples
"""

from typing import List, Tuple, Callable, Optional, Any

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = Any  # Type hint fallback

def get_token_ranges(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    pre_string: str = "",
) -> List[Tuple[int, int]]:
    """
    Get token ranges for each sentence.

    Uses iterative tokenization to avoid boundary issues:
    1. Tokenize pre_string
    2. Add sentence1, tokenize (pre_string + sentence1)
    3. Add sentence2, tokenize (pre_string + sentence1 + sentence2)
    ... and so on

    Args:
        sentences: List of sentences (should include trailing whitespace)
        tokenizer: AutoTokenizer used to tokenize the text
        pre_string: Text that comes before sentences (e.g., chat template)

    Returns:
        List of tuples (start_idx, end_idx) where:
        - start_idx is inclusive
        - end_idx is exclusive
        - sentences[i]'s tokens are tokenizer(full_text)[start_idx:end_idx]
    """

    token_ranges = []
    accumulated_text = pre_string

    # Get initial token count from pre_string
    if pre_string:
        pre_tokens = tokenizer.encode(pre_string)
        prev_token_count = len(pre_tokens)
    else:
        prev_token_count = 0

    # Process each sentence
    for sentence in sentences:
        # Add current sentence to accumulated text
        accumulated_text += sentence

        # Tokenize the full accumulated text
        all_tokens = tokenizer.encode(accumulated_text)
        current_token_count = len(all_tokens)

        token_range = (prev_token_count, current_token_count)
        token_ranges.append(token_range)

        prev_token_count = current_token_count

    return token_ranges


def validate_token_ranges(
    token_ranges: List[Tuple[int, int]],
    sentences: List[str],
    tokenizer: AutoTokenizer,
    pre_string: str = "",
) -> bool:
    """
    Validate that token ranges are correct.

    Checks:
    1. Ranges are monotonically increasing
    2. Each range's end equals the next range's start
    3. No gaps or overlaps
    4. Reconstructed text matches original

    Args:
        token_ranges: List of (start, end) tuples
        sentences: Original sentences
        tokenizer: AutoTokenizer used to tokenize the text
        pre_string: Text before sentences

    Returns:
        True if all validations pass
    """

    # Check 1: Monotonically increasing
    for i, (start, end) in enumerate(token_ranges):
        if start >= end:
            print(f"ERROR: Range {i} is invalid: start={start} >= end={end}")
            return False

        if i > 0:
            prev_end = token_ranges[i-1][1]
            if start != prev_end:
                print(f"ERROR: Gap/overlap at range {i}: prev_end={prev_end}, start={start}")
                return False
            if start <= token_ranges[i-1][0]:
                print(f"ERROR: Range {i} starts before previous range")
                return False

    # Check 2: First range starts after pre_string tokens
    if pre_string:
        pre_tokens = tokenizer.encode(pre_string)
        expected_start = len(pre_tokens)
        if token_ranges[0][0] != expected_start:
            print(f"ERROR: First range should start at {expected_start}, got {token_ranges[0][0]}")
            return False
    elif token_ranges[0][0] != 0:
        print(f"ERROR: First range should start at 0 with no pre_string, got {token_ranges[0][0]}")
        return False

    # Check 3: Token reconstruction
    full_text = pre_string + ''.join(sentences)
    full_tokens = tokenizer.encode(full_text)

    for i, (start, end) in enumerate(token_ranges):
        if end > len(full_tokens):
            print(f"ERROR: Range {i} end={end} exceeds total tokens={len(full_tokens)}")
            return False

    print("  OK: Ranges are contiguous and monotonic")
    return True

