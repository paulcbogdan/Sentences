"""
Complete sentence splitter with exact position tracking.

Key guarantee: Every character in the text belongs to exactly one sentence.
This ensures perfect reconstruction: ''.join(sentences) == text
"""

import re
from typing import List, Tuple


def split_text_to_sentences(text: str, min_sentence_length: int = 4) -> Tuple[List[str], List[int]]:
    """
    Split text into sentences, ensuring complete coverage.

    Every character in the input text will be included in exactly one sentence.
    Short segments are merged with adjacent sentences.

    Guarantees:
    - ''.join(sentences) == text
    - text[positions[i]:positions[i+1]] == sentences[i] for all valid i

    Args:
        text: Input text
        min_sentence_length: Minimum length for standalone sentence

    Returns:
        (sentences, positions)
    """
    if not text:
        return [], []

    # Protect common abbreviations
    abbrevs = [
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 'Ph.D.', 'M.D.',
        'B.A.', 'M.A.', 'B.S.', 'M.S.', 'i.e.', 'e.g.', 'etc.', 'vs.', 'Inc.',
        'Ltd.', 'Co.', 'Corp.', 'U.S.', 'U.K.', 'E.U.', 'N.Y.'
    ]

    protected = text
    abbrev_map = {}
    for i, abbrev in enumerate(abbrevs):
        if abbrev in protected:
            placeholder = f"_AB{i:02d}_"
            abbrev_map[placeholder] = abbrev
            protected = protected.replace(abbrev, placeholder)

    # Find potential sentence boundaries
    boundaries = [0]  # Always start at beginning

    # After period/exclamation/question + space + capital
    for m in re.finditer(r'[.!?]\s+([A-Z])', protected):
        # Boundary should be at the start of the capital letter
        capital_pos = m.start(1)  # Position of the captured capital letter
        boundaries.append(capital_pos)

    # After colon + space + capital
    for m in re.finditer(r':\s+([A-Z])', protected):
        # Boundary should be at the start of the capital letter
        capital_pos = m.start(1)  # Position of the captured capital letter
        boundaries.append(capital_pos)

    # After newlines - only if previous char is not sentence-ending punctuation
    for m in re.finditer(r'\n', protected):
        # Check what comes before the newline
        prev_pos = m.start() - 1
        if prev_pos >= 0:
            prev_char = protected[prev_pos]
            # Only add boundary if previous char is not sentence-ending punctuation
            if prev_char not in '.!?:':
                pos = m.end()
                while pos < len(protected) and protected[pos] in ' \t':
                    pos += 1
                if pos < len(protected) and protected[pos] != '\n':
                    boundaries.append(pos)

    # Add end of text (use protected length, not original)
    boundaries.append(len(protected))

    # Sort and deduplicate
    boundaries = sorted(set(boundaries))

    # Build initial segments from PROTECTED text
    segments = []
    for i in range(len(boundaries) - 1):
        # Get segment from protected text first
        segment_text = protected[boundaries[i]:boundaries[i+1]]

        # Restore abbreviations in this segment
        for placeholder, abbrev in abbrev_map.items():
            segment_text = segment_text.replace(placeholder, abbrev)

        segments.append({
            'text': segment_text,
            'position': boundaries[i],  # Will be fixed later
            'core_length': len(segment_text.strip())
        })

    # Merge short segments with neighbors
    merged_segments = []
    i = 0
    while i < len(segments):
        current = segments[i]

        # If this segment is too short, merge it with previous or next
        if current['core_length'] < min_sentence_length:
            if merged_segments:
                # Merge with previous
                prev = merged_segments[-1]
                prev['text'] += current['text']
                # Position stays the same (from previous)
            elif i + 1 < len(segments):
                # No previous, merge with next
                next_seg = segments[i + 1]
                next_seg['text'] = current['text'] + next_seg['text']
                next_seg['position'] = current['position']
            else:
                # Only segment and it's short - keep it anyway
                merged_segments.append(current)
        else:
            # Segment is long enough
            merged_segments.append(current)

        i += 1

    for i, seg in enumerate(merged_segments[:-1]):
        if seg['text'].endswith(' '):
            next_seg = merged_segments[i+1]
            next_seg['text'] = ' ' + next_seg['text']
            seg['text'] = seg['text'][:-1]

    # Now fix positions to match the original text
    # We need to recalculate positions based on the actual merged segments
    sentences = []
    positions = []
    current_pos = 0

    for seg in merged_segments:
        seg_text = seg['text']
        # Find where this segment starts in the original text
        # (it should be at current_pos if we've done everything right)
        sentences.append(seg_text)
        positions.append(current_pos)
        current_pos += len(seg_text)

    # Validate our guarantee
    reconstructed = ''.join(sentences)
    if reconstructed != text:
        print(f"Reconstruction failed!")
        print(f"Original length: {len(text)}")
        print(f"Reconstructed length: {len(reconstructed)}")
        for i, (c1, c2) in enumerate(zip(text, reconstructed + ' '*100)):
            if c1 != c2:
                print(f"First diff at position {i}: {repr(c1)} != {repr(c2)}")
                break
        raise AssertionError("Failed to maintain complete coverage")

    # Validate position accuracy
    for i in range(len(sentences)):
        if i < len(sentences) - 1:
            expected = text[positions[i]:positions[i+1]]
        else:
            expected = text[positions[i]:]

        if expected != sentences[i]:
            print(f"Position mismatch at sentence {i}")
            print(f"  Stored: {repr(sentences[i][:50])}")
            print(f"  Expected: {repr(expected[:50])}")
            raise AssertionError(f"Position mismatch at sentence {i}")

    return sentences, positions