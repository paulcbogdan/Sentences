import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentences import split_text_to_sentences
from sentences import get_token_ranges
from transformers import AutoTokenizer
import json


def main():
    """Demonstrate sentence splitter on a COT example."""

    # Load a COT example
    with open("examples/example_cots.json", "r", encoding="utf-8") as f:
        cots = json.load(f)

    example = cots[1]  # Example

    # Extract thinking content
    full_cot = example["full_cot"]
    think_start = full_cot.find("<think>")
    think_end = full_cot.find("</think>")

    if think_start != -1 and think_end != -1:
        think_content = full_cot[think_start + 7:think_end]
    else:
        think_content = full_cot

    print("="*60)
    print(f"SENTENCE SPLITTER EXAMPLE")
    print(f"Problem: {example['problem_id']}")
    print("="*60)

    # Split into sentences
    sentences, positions = split_text_to_sentences(think_content)

    print(f"\nExtracted {len(sentences)} sentences (notice leading spaces, which helps with tokenization)")
    print(f"Original text length: {len(think_content)} characters")

    # Show first 5 sentences with their positions
    print("\nFirst 5 sentences with positions:")
    for i in range(min(5, len(sentences))):
        print(f"  {i}: pos={positions[i]:4d}, '{sentences[i]}'")

    # Verify exact reconstruction
    print("\nVerifying exact reconstruction:")
    for i in range(min(5, len(sentences))):
        if i < len(positions) - 1:
            extracted = think_content[positions[i]:positions[i+1]]
        else:
            extracted = think_content[positions[i]:]

        matches = extracted == sentences[i]
        assert matches, f"Sentence {i} does not match"
    print(f'\tExact reconstruction: OK')

def README_examples():
    text = "Dr. Smith went to the store. They bought some milk. It cost $3.50."
    sentences, positions = split_text_to_sentences(text)

    for i, (sent, pos) in enumerate(zip(sentences, positions)):
        print(f"{i}: {repr(sent)}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

    # Example with Qwen3-32B chat template
    pre_string = """<|im_start|>system
    This system message is just for demonstration purposes.<|im_end|>
    <|im_start|>user
    Solve this math problem step by step.<|im_end|>
    <|im_start|>assistant
    <think>

    """

    sentences = ["Let me think about this problem.", " First, I'll break it down."]
    ranges = get_token_ranges(sentences, tokenizer, pre_string)
    tokens_all = tokenizer.batch_decode(tokenizer.encode(pre_string + ''.join(sentences)))

    for sent, (start, end) in zip(sentences, ranges):
        print(f"Tokens [{start}:{end}] = '{sent}'\n\t{tokens_all[start:end]}")

if __name__ == "__main__":
    # main()
    README_examples()