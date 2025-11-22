# sentences

Text segmentation and tokenization utilities for LLM tokenizers. This 

## Features

- Splits a given text into sentences
- Avoids common issues (e.g., "Dr. Fu" shouldn't be a sentnece split)
- Is mindful of standard LLM tokenization patterns - e.g., "I love my cat. It is big." should be split with a leading space rather than a trailing one, ["I love my cat.", " It is big."]
- Given a tokenizer, it will find the token ranges of your sentences' positions in the input text

## Installation

```bash
pip install sentences
```
## Sentence Splitting

```python
from sentences import split_text_to_sentences

text = "Dr. Smith went to the store. They bought some milk. It cost $3.50."
sentences, positions = split_text_to_sentences(text)

for i, (sent, pos) in enumerate(zip(sentences, positions)):
    print(f"{i}: {repr(sent)}")

    # 0: 'Dr. Smith went to the store.'
    # 1: ' They bought some milk.'
    # 2: ' It cost $3.50.'

    # Verify reconstruction
    assert text[pos:positions[i+1] if i+1 < len(positions) else len(text)] == sent
```

## Token Range Extraction

Get exact token ranges for sentences. You can use this to split up a model's chain-of-thought into sentences.

```python
from sentences import get_token_ranges
from transformers import AutoTokenizer

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
    # Tokens [39:46] = 'Let me think about this problem.'
    #   [' Let', ' me', ' think', ' about', ' this', ' problem', '.']
    # Tokens [46:54] = ' First, I'll break it down.'
    #   [' First', ',', ' I', "'ll", ' break', ' it', ' down', '.']
```

Token ranges are calculated by repeatedly appending a new sentence to the `pre_string`, tokenizing the new string, and counting the number of tokens. This helps avoid tokenization oddities. Simply tokenizing each sentence independently can cause problems.  

### Note on CoT pre-filling

gpt-oss models don't use "<think>" tags but instead employee a special format:

```python
# GPT-OSS uses a different format without <think> tags
pre_string = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-11-22<|end|><|start|>user<|message|>Solve this math problem step by step.<|end|><|start|>assistant<|channel|>analysis<|message|>"""
```
