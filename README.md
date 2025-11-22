# Sentences

[![PyPI version](https://img.shields.io/pypi/v/sentences.svg)](https://pypi.org/project/sentences/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.19143-b31b1b.svg)](https://arxiv.org/abs/2506.19143)

Utilities for sentence-level text segmentation and tokenization tailored to LLM tokenizers.  

This package is designed to support sentence-level (“Thought Anchor”) analyses like those in:

> Bogdan, P. C.\*, Macar, U.\*, Nanda, N.°, & Conmy, A.° Thought anchors: Which LLM reasoning steps matter? 2025. https://arxiv.org/abs/2506.19143.

## Features

- Splits a given text into sentences
- Avoids common issues (e.g., "Dr. Fu" shouldn't be split into two sentences)
- Respects standard LLM tokenization patterns (e.g., leading-space tokens)
- Given a tokenizer, returns token ranges for each sentence in the tokenized input text

## Installation

```
pip install sentences
```
## Sentence Splitting

Sentences split here adhere to typical LLM tokenization strategies. For example, this sentence `"I love my cat. It is big."` should be split with a leading space rather than a trailing one, `["I love my cat.", " It is big."]`

```python
from sentences import split_text_to_sentences

text = "Dr. Smith went to the store. They bought some milk. It cost $3.50."
sentences, positions = split_text_to_sentences(text)

for i, (sent, pos) in enumerate(zip(sentences, positions)):
    print(f"{i}: {repr(sent)}")
    # 0: 'Dr. Smith went to the store.'
    # 1: ' They bought some milk.'
    # 2: ' It cost $3.50.'
```

## Token Range Extraction

You can use this package to get the exact token ranges for sentences. You can use this to split up a model's chain-of-thought into sentences. You can include `pre_string`, where you provide a string that will appear before your sentences (e.g., a chat template), and the token ranges will respect that.

Token ranges are calculated by repeatedly appending a new sentence to the `pre_string`, tokenizing the new string, and counting the number of tokens. This helps avoid tokenization oddities. Simply tokenizing each sentence independently can cause problems.  

```python
from sentences import get_token_ranges
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# Example with Qwen-2.5 chat template
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

### Note on CoT pre-filling

gpt-oss models don't use `<think>` tags but instead employ a special format:

```python
pre_string = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-11-22<|end|><|start|>user<|message|>Solve this math problem step by step.<|end|><|start|>assistant<|channel|>analysis<|message|>"""
```
