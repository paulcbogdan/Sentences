import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from sentences import split_text_to_sentences
from sentences import get_token_ranges, validate_token_ranges
from transformers import AutoTokenizer


def main(model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
    """Demonstrate token range extraction with realistic chat template."""

    # Load actual COT example
    with open("examples/example_cots.json", "r", encoding="utf-8") as f:
        cots = json.load(f)

    example = cots[1]

    # Extract and clean the prompt
    prompt = example["prompt"]
    if "<think>" in prompt:
        prompt = prompt.split("<think>")[0].strip()
    if prompt.endswith("Solution:"):
        prompt = prompt[:-len("Solution:")].strip()

    print("="*60)
    print("TOKEN RANGE EXTRACTION WITH CHAT TEMPLATE")
    print("="*60)

    # Create messages list (system message is optional)
    messages = []

    # Optional system message
    system_msg = "This system message is just for demonstration purposes."
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    # User message
    messages.append({"role": "user", "content": prompt})


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pre_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "gpt-oss" in model_name: # gpt-oss doesn't use "<think>" tag
        pre_string += "<|channel|>analysis<|message|>" # Uses this to start CoT
    else:
        pre_string += "<think>\n" # May be different for non-qwen models

    print(f"\nMessages ({len(messages)} total):")
    for msg in messages:
        content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  {msg['role']}: {repr(content_preview)}")

    print(f"\nGenerated pre-string ({len(pre_string)} chars):")
    print("-"*40)
    print(pre_string)
    print("-"*40)

    # Get the actual COT response text
    full_cot = example["full_cot"]
    think_st_str = "<think>\n"
    think_start = full_cot.find(think_st_str)
    think_end = full_cot.find("</think>")

    response_text = full_cot[think_start + len(think_st_str):think_end]

    # Split response into sentences
    sentences, _ = split_text_to_sentences(response_text)
 
    print("First 5 sentences:")
    for i, sent in enumerate(sentences[:5]):
        print(f"  {i}: {repr(sent)}")

    print(f"\nModel response has {len(sentences)} sentences")
    print("Using first 5 sentences for this example.")

    # Define a simple tokenizer (whitespace-based)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get token ranges using iterative tokenization
    ranges = get_token_ranges(sentences[:5], tokenizer, 
                              pre_string=pre_string)

    # Get full text and all tokens
    full_text = pre_string + ''.join(sentences[:5])


    all_tokens = tokenizer.batch_decode(tokenizer.encode(full_text))

    print(f"\nPre-string has {len(all_tokens)} tokens")
    print(f"Total tokens after adding 5 sentences: {len(all_tokens)}")

    print("\nToken ranges for each sentence:")
    print("  Range        Sentence")
    print("  -----------  --------")
    for sent, (token_start, token_end) in zip(sentences[:5], ranges):
        # Get actual tokens for this sentence
        tokens = all_tokens[token_start:token_end]
        n_tokens = len(tokens)

        # Display sentence (truncated)
        sent_display = sent.strip()

        print(f"  [{token_start:3d}, {token_end:3d})  '{sent_display}'")
        print(f"               {n_tokens} tokens: {tokens}")

    # Validate the ranges

    validate_token_ranges(ranges, sentences[:5], tokenizer, pre_string)

    # Show actual tokens for first sentence
    print("\nExample - First sentence tokens:")
    first_tokens = all_tokens[ranges[0][0]:ranges[0][1]]
    print(f"  Tokens {ranges[0]}: {first_tokens}")

if __name__ == "__main__":
    main()