## Minimal GPT Implementation (from scratch)

This project is a minimal, learning-oriented implementation of a GPT-like
language model built entirely from scratch.

The goal of this project is conceptual and implementation-level understanding
of Transformer-based language models, rather than performance or production use.

### Motivation
Before relying on large pretrained models, this experiment was conducted to:
- Understand the internal structure of GPT-style Transformers
- Implement the full training and generation pipeline manually
- Observe how data scale and generation parameters affect model behavior

### Scope
- Token-based language modeling
- Autoregressive Transformer architecture
- Training loop and checkpointing
- Text generation with configurable parameters

### Usage

#### Training
Train the model on a plain text corpus.

python mini_ja_gpt_all_in_one.py --mode train --data_path <path_to_text_file>

Input:
- A single UTF-8 encoded plain text file

Output:
- A checkpoint file containing trained model parameters

#### Text Generation
Generate text from a trained checkpoint and a given prompt.

python mini_ja_gpt_all_in_one.py --mode generate --prompt "Your prompt here" --ckpt_path <checkpoint_path>

#### Simple Chat Mode
Run a lightweight, terminal-based chat loop using a trained model.

python mini_ja_gpt_all_in_one.py --mode chat --ckpt_path <checkpoint_path>

For a full list of available arguments:

python mini_ja_gpt_all_in_one.py --help

### Limitations and Reflections

- Model and dataset sizes are intentionally small, resulting in limited output quality.
- Training speed and scalability are constrained by the simplicity of the implementation.
- Tokenization, optimization, and training stability are simplified for clarity.
- This experiment highlights that practical LLM performance requires large-scale data,
  substantial compute resources, and extensive engineering.
- In real-world applications, pretrained models and mature frameworks should be used.

This project is intended as a learning artifact, not a reusable production library.