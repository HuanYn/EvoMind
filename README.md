# MiniMind from Scratch

A compact, educational implementation of a decoder-only language model in PyTorch. The project is structured to make the core LLM pipeline inspectable and reproducible:

```text
raw corpus → cleaning → BPE tokenizer → token IDs → MiniMind → training / generation / evaluation
```

## Included

- Decoder-only Transformer with causal self-attention, GQA, pre-norm residual blocks, MLP, token embedding, positional embedding, and LM head.
- Character-level tokenizer baseline and scripts to train a byte-level BPE tokenizer from a corpus.
- Data cleaning, tokenizer inspection, held-out tokenizer evaluation, pretraining, generation, and language-model evaluation scripts.
- Unit tests for the model forward/backward path and character tokenizer persistence.

## Setup

```powershell
conda create -n MLLM python=3.10
conda activate MLLM
python -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your system from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Quick start

Prepare your own text corpus, with one document per line after cleaning:

```powershell
python scripts\train_bpe_tokenizer.py --input data\processed\corpus.txt --output data\tokenizers\minimind_bpe_16k.json --vocab-size 16000
python scripts\inspect_bpe_tokenizer.py --tokenizer data\tokenizers\minimind_bpe_16k.json
python scripts\evaluate_bpe_tokenizer.py --input data\processed\validation.txt --tokenizer data\tokenizers\minimind_bpe_16k.json
```

The repository deliberately does **not** include corpora, trained tokenizer files, checkpoints, local experiment artifacts, or notebooks. Create these locally using the scripts above.

## Tests

```powershell
python -m unittest discover -s test -p "test_*.py" -v
```

## Current scope

The repository currently provides the MiniMind decoder-only baseline and its data/tokenizer pipeline. RoPE, KV cache, MoE, alignment training (SFT/DPO/GRPO), multimodal MiniMind-V modules, and benchmark evaluation will be added as subsequent project stages.
