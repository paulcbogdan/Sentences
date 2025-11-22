"""
Paragraph Splitter Module

A utility for splitting text into paragraphs with special handling for lists.
Includes position tracking and intelligent merging of related content.
"""

import re
from typing import List, Tuple, Dict


def split_text_to_paragraphs(
    text: str,
    min_paragraph_length: int = 64
) -> Tuple[List[str], List[int]]:
    """
    Split text into paragraphs with special handling for lists.

    Features:
    - Splits on double newlines
    - Merges consecutive list items
    - Handles introduction sentences before lists
    - Filters out short paragraphs

    Args:
        text: Input text to split
        min_paragraph_length: Minimum length for a valid paragraph

    Returns:
        Tuple containing:
            - List of paragraphs
            - List of starting positions for each paragraph
    """
    if not text or not isinstance(text, str):
        return [], []

    # Split on double newlines
    raw_paragraphs = text.split("\n\n")
    paragraph_info = []
    text_position = 0

    # Analyze each paragraph
    for para_idx, para in enumerate(raw_paragraphs):
        # Find paragraph position
        if para_idx > 0:
            text_position = text.find(para, text_position)
            if text_position == -1:
                text_position = len("\n\n".join(raw_paragraphs[:para_idx])) + 2 * para_idx

        # Check if paragraph contains list items
        lines = para.strip().split("\n")
        is_list = _is_list_paragraph(lines)
        has_intro = _has_intro_line(lines, is_list)

        paragraph_info.append({
            "text": para,
            "position": text_position,
            "is_list": is_list,
            "has_intro": has_intro
        })

        text_position += len(para)

    # Merge paragraphs intelligently
    merged_paragraphs, merged_positions = _merge_paragraphs(
        paragraph_info,
        min_paragraph_length
    )

    return merged_paragraphs, merged_positions


def _is_list_paragraph(lines: List[str]) -> bool:
    """
    Check if a paragraph contains list items.

    Args:
        lines: Lines in the paragraph

    Returns:
        True if the paragraph contains list formatting
    """
    for line in lines:
        stripped = line.strip()
        if stripped and (
            re.match(r"^[-*•]\s+", stripped) or
            re.match(r"^\d+\.\s+", stripped) or
            re.match(r"^[a-zA-Z]\.\s+", stripped)
        ):
            return True

    # Check for header followed by list
    if len(lines) > 1 and lines[0].strip().endswith(":"):
        for line in lines[1:]:
            if _is_list_item(line.strip()):
                return True

    return False


def _is_list_item(text: str) -> bool:
    """Check if text is a list item."""
    return bool(
        text and (
            re.match(r"^[-*•]\s+", text) or
            re.match(r"^\d+\.\s+", text) or
            re.match(r"^[a-zA-Z]\.\s+", text)
        )
    )


def _has_intro_line(lines: List[str], is_list: bool) -> bool:
    """
    Check if a list paragraph has an introductory line.

    Args:
        lines: Lines in the paragraph
        is_list: Whether the paragraph is a list

    Returns:
        True if there's an intro line before list items
    """
    if not is_list or not lines:
        return False

    # Check if first line is not a list item
    if not _is_list_item(lines[0].strip()):
        return True

    # Check if list items start after the first line
    for i, line in enumerate(lines):
        if i > 0 and _is_list_item(line.strip()):
            return True

    return False


def _merge_paragraphs(
    paragraph_info: List[Dict],
    min_length: int
) -> Tuple[List[str], List[int]]:
    """
    Merge paragraphs intelligently, combining related content.

    Args:
        paragraph_info: List of paragraph information dictionaries
        min_length: Minimum paragraph length to keep

    Returns:
        Tuple of merged paragraphs and their positions
    """
    merged_paragraphs = []
    merged_positions = []
    i = 0

    while i < len(paragraph_info):
        current = paragraph_info[i]

        # Check if current paragraph introduces a following list
        if _should_merge_with_list(i, paragraph_info):
            merged_text, position = _merge_intro_with_list(i, paragraph_info)
            if len(merged_text) >= min_length:
                merged_paragraphs.append(merged_text)
                merged_positions.append(position)
            # Skip the merged paragraphs
            i = _find_next_non_list(i + 1, paragraph_info)
            continue

        # Handle consecutive list paragraphs
        if current["is_list"]:
            merged_text, position = _merge_consecutive_lists(i, paragraph_info)
            if len(merged_text) >= min_length:
                merged_paragraphs.append(merged_text)
                merged_positions.append(position)
            # Skip the merged list paragraphs
            i = _find_next_non_list(i, paragraph_info)
        else:
            # Regular paragraph
            if len(current["text"]) >= min_length:
                merged_paragraphs.append(current["text"])
                merged_positions.append(current["position"])
            i += 1

    return merged_paragraphs, merged_positions


def _should_merge_with_list(index: int, paragraph_info: List[Dict]) -> bool:
    """Check if paragraph should be merged with following list."""
    if index + 1 >= len(paragraph_info):
        return False

    current = paragraph_info[index]
    next_para = paragraph_info[index + 1]

    if current["is_list"] or not next_para["is_list"]:
        return False

    # Check if current paragraph ends with colon or contains intro language
    text = current["text"].strip()
    return (
        text.endswith(":") or
        "following" in text.lower() or
        "each" in text.lower() or
        "below" in text.lower()
    )


def _merge_intro_with_list(index: int, paragraph_info: List[Dict]) -> Tuple[str, int]:
    """Merge an introductory paragraph with following list paragraphs."""
    parts = [paragraph_info[index]["text"]]
    position = paragraph_info[index]["position"]

    j = index + 1
    while j < len(paragraph_info) and paragraph_info[j]["is_list"]:
        parts.append(paragraph_info[j]["text"])
        j += 1

    return "\n\n".join(parts), position


def _merge_consecutive_lists(index: int, paragraph_info: List[Dict]) -> Tuple[str, int]:
    """Merge consecutive list paragraphs."""
    parts = [paragraph_info[index]["text"]]
    position = paragraph_info[index]["position"]

    j = index + 1
    while j < len(paragraph_info) and paragraph_info[j]["is_list"]:
        parts.append(paragraph_info[j]["text"])
        j += 1

    return "\n\n".join(parts), position


def _find_next_non_list(index: int, paragraph_info: List[Dict]) -> int:
    """Find the index of the next non-list paragraph."""
    while index < len(paragraph_info) and paragraph_info[index]["is_list"]:
        index += 1
    return index


# Convenience function with simpler interface
def split_into_paragraphs(text: str) -> List[str]:
    """
    Simple interface to split text into paragraphs.

    Args:
        text: Input text

    Returns:
        List of paragraphs
    """
    paragraphs, _ = split_text_to_paragraphs(text)
    return paragraphs


if __name__ == "__main__":
    # Example usage
    sample_text = """
    First paragraph here.
    It spans multiple lines.

    Second paragraph with a list:
    - Item 1
    - Item 2

    Final paragraph.
    """

    paragraphs = split_into_paragraphs(sample_text)
    for i, paragraph in enumerate(paragraphs, 1):
        print(f"Paragraph {i}:")
        print(paragraph)
        print()