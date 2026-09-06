# MiniMind from Scratch

A compact, educational implementation of a decoder-only language model in PyTorch. The project is structured to make the core LLM pipeline inspectable and reproducible:

```text
raw corpus → cleaning → BPE tokenizer → token IDs → MiniMind → training / generation / evaluation
```

## Included

- Decoder-only Transformer with causal self-attention, GQA, RoPE, RMSNorm, SwiGLU, KV cache, sparse MoE, token embedding, and LM head.
- Character-level tokenizer baseline and scripts to train a byte-level BPE tokenizer from a corpus.
- Data cleaning, tokenizer inspection, held-out tokenizer evaluation, pretraining, generation, language-model evaluation, and Dense/MoE inference benchmarks.
- Versioned JSONL training histories and automatically refreshed PNG dashboards for loss, perplexity, learning rate, gradient norm, and MoE routing health.
- Chat-template-aware SFT data preparation with assistant-only loss masks and leakage-safe train/validation splitting.
- Native post-training implementations for assistant-only SFT, DPO, and GRPO/RLAIF.
- Unit tests for model forward/backward, KV-cache equivalence, tokenization, SFT/DPO/GRPO objectives, data masking, rewards, and training-history persistence.

## Repository layout

The top-level responsibilities follow the official MiniMind project while the
model itself remains split into small teaching-friendly modules:

```text
model/       model architecture and reusable learning components
trainer/     train_pretrain.py, train_full_sft.py, train_dpo.py, train_grpo.py
dataset/     dataset layout and provenance documentation (large data excluded)
scripts/     data preparation, evaluation, generation, diagnostics, benchmarks
test/        unit and regression tests
eval_assets/ small versioned evaluation prompts
```

This is a structural alignment rather than a source-code copy. In particular,
`model/model_minimind.py` is the model assembly point; attention, RoPE,
RMSNorm, SwiGLU, MoE, chat templates, and alignment losses stay in separate
modules so each component can be inspected independently.

## Data provenance and reproducibility ledger

No raw corpus, tokenizer, checkpoint, or generated artifact is committed to this repository.  Every local training run must be traceable to a source file, a transformation script, a deterministic split, and a report containing SHA-256 hashes.

| Stage | Local input / output | Upstream source | Local processing and split | Status / intended use |
| --- | --- | --- | --- | --- |
| Pretraining | `data/raw/wikipedia_zh_10k.jsonl` + `wikipedia_zh_train_extra_100k.jsonl` → `data/processed/wikipedia_zh_train_110k_clean.txt` | `wikimedia/wikipedia`, config `20231101.zh`, streamed through `scripts/download_wiki_zh_sample.py` | Drops documents shorter than 200 characters at download; cleaning removes blank/short documents and exact duplicates; the two cleaned train files are merged with cross-file deduplication | Completed: 109,215 cleaned documents, used by Dense/MoE pretraining |
| Pretraining validation | `data/raw/wikipedia_zh_val_1k.jsonl` → `data/processed/wikipedia_zh_val_1k_clean.txt` | Same Wikimedia snapshot | Downloader skips the first 10,000 eligible documents before collecting 1,000 validation documents; uses the same cleaner | Completed: 994 cleaned documents; held out from the initial 10K train slice |
| Tokenizer | `data/processed/wikipedia_zh_train_110k_clean.txt` → `data/tokenizers/minimind_bpe_16k_110k.json` | The local cleaned pretraining corpus above | Byte-level BPE, 16,000 vocabulary entries, with `<pad>/<bos>/<eos>/<unk>` IDs 0/1/2/3 | Completed: project-owned tokenizer; do not compare token-level PPL across different tokenizers |
| SFT source | `data/raw/minimind_sft/sft_t2t_mini.jsonl` | Official `jingyaogong/minimind_dataset` release, file `sft_t2t_mini.jsonl` | Official dataset describes this as unified multi-turn instruction, reasoning, and Tool Call data; its published upstream ingredients include public instruction/dialogue corpora and model-distilled synthetic data | Completed source download; source SHA-256 is stored in the preparation report |
| SFT train/validation | `sft_t2t_mini.jsonl` → `sft_train_20k_seed42.jsonl`, `sft_val_1k_seed42.jsonl` | Official MiniMind SFT source above | Canonical JSON SHA-256 deduplication, stable hash split (`val_ratio=0.01`), then deterministic sampling with seed 42; rejects invalid/no-target records and records truncation effects | Completed: 20,000 train / 1,000 held-out validation; report `artifacts/sft_prepare_20k_1k_seed42.json` |
| SFT scale ablation | `sft_train_20k_seed42.jsonl` → `sft_train_2k_seed42.jsonl` | Local 20K SFT training split | Lowest stable SHA-256 priorities with seed 42; validation set unchanged | Completed controlled 2K subset; report `artifacts/sft_prepare_2k_from_20k_seed42.json` |
| DPO | `data/raw/minimind_dpo/dpo.jsonl` → local Chinese filtered train/validation pairs | Official `jingyaogong/minimind_dataset` release, whose README states `dpo.jsonl` is sampled from `llamafactory/DPO-En-Zh-20k` | Validates a shared prompt prefix, distinct chosen/rejected responses, assistant-only targets, and rejects truncation; deterministic Chinese subset/split used for the local run | Completed: 5,000 train / 500 held-out pairs, Dense DPO |
| GRPO exact-reward baseline | `scripts/prepare_grpo_math_data.py` → `data/processed/minimind_grpo/math_*_seed42.jsonl` | Locally generated non-negative addition/subtraction facts; not an external corpus | Three Chinese templates per fact; prompt has no answer; hidden integer answer is used only by a deterministic verifier | Completed as a unit-test / algorithm baseline; **not** the main MiniMind RLAIF experiment |
| GRPO / RLAIF main line | Official `rlaif.jsonl` → 15K train / 500 held-out prompt-only JSONL; frozen `internlm2-1_8b-reward` | Official `jingyaogong/minimind_dataset` and `internlm/internlm2-1_8b-reward` releases | Removes only the empty final assistant placeholder; exact-history deduplication and seed-42 stable ordering; source/tokenizer hashes and left-truncation stats in report | Prepared: 19,502 raw → 19,400 unique; 98 duplicates, 4 invalid; 43/19,400 prompts exceed 768. RM smoke passed: specific answer -1.3584 > vague answer -2.8965; RM-alone peak 3.21 GiB. GRPO training remains pending. |

### Source and claim boundaries

- “Official source” above means the released file is downloaded from the named upstream repository; it does **not** imply every individual example has a fully published one-to-one original-data lineage.
- The official MiniMind documentation lists its SFT ingredients as public instruction/dialogue data and model-distilled synthetic data, including Craftsman, Magpie-Align, R1-Distill-SFT, COIG, and Step-3.5-Flash-SFT. This project does not relabel those examples as independently collected data.
- Each preparation or training command writes a JSON report or JSONL history. Preserve these alongside a checkpoint before making a result claim.
- Before redistributing data or weights, verify the upstream licenses and their transitive restrictions yourself; this repository records provenance but does not grant rights beyond the original licenses.

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

Prepare token streams and start pretraining:

```powershell
python scripts\prepare_pretrain_data.py --input data\processed\corpus.txt --tokenizer data\tokenizers\minimind_bpe_16k.json --output data\prepared\pretrain.pt
python trainer\train_pretrain.py --train-data data\prepared\pretrain.pt --val-data data\prepared\validation.pt --checkpoint checkpoints\minimind_pretrain.pt
```

During pretraining, raw metrics are appended after every optimizer step and the
plot is refreshed at each evaluation point:

```text
checkpoints/<name>.pt
artifacts/training/<name>.metrics.jsonl
artifacts/training/<name>.curves.png
```

Existing histories can be replotted or compared without rerunning training:

```powershell
python scripts\plot_training_curves.py --metrics artifacts\training\run_a.metrics.jsonl artifacts\training\run_b.metrics.jsonl --output artifacts\training\run_a_vs_b.curves.png --smoothing-window 100
```

## Supervised fine-tuning

Prepare a leakage-safe SFT split, then initialize a new SFT run from a
pretraining checkpoint:

```powershell
python scripts\prepare_sft_data.py --input data\raw\minimind_sft\sft_t2t_mini.jsonl --tokenizer data\tokenizers\minimind_bpe_16k_110k.json --train-output data\processed\minimind_sft\sft_train_20k_seed42.jsonl --val-output data\processed\minimind_sft\sft_val_1k_seed42.jsonl --smoke-output data\processed\minimind_sft\sft_smoke_256_seed42.jsonl --report artifacts\sft_prepare_20k_1k_seed42.json --train-size 20000 --val-size 1000 --smoke-size 256 --max-length 768 --seed 42
python trainer\train_full_sft.py --train-data data\processed\minimind_sft\sft_train_20k_seed42.jsonl --val-data data\processed\minimind_sft\sft_val_1k_seed42.jsonl --tokenizer data\tokenizers\minimind_bpe_16k_110k.json --pretrained-checkpoint checkpoints\minimind_dense_v2_20k.pt --checkpoint checkpoints\minimind_dense_sft_2ep.pt --epochs 2 --batch-size 1 --grad-accum-steps 16 --max-length 768 --empty-think-ratio 0.2 --system-prompt-ratio 0.2 --lr 1e-5 --min-lr 1e-6 --amp-dtype bfloat16
```

The trainer computes shifted cross entropy only on assistant outputs. It also
supports exact SFT resume, token-weighted gradient accumulation, deterministic
system-prompt augmentation, validation, raw JSONL metrics, and automatic curve
plots. Use `--pretrained-checkpoint` only to begin SFT; use `--resume` to
continue an existing SFT run with its optimizer and data cursor.

Evaluate on held-out assistant tokens and generate with the exact training chat
template rather than the raw pretraining completion prompt:

```powershell
python scripts\evaluate_sft.py --checkpoint checkpoints\minimind_dense_sft_2ep.pt --data data\processed\minimind_sft\sft_val_1k_seed42.jsonl --tokenizer data\tokenizers\minimind_bpe_16k_110k.json --max-length 768 --batch-size 1 --max-batches 1000
python -X utf8 scripts\chat.py --checkpoint checkpoints\minimind_dense_sft_2ep.pt --tokenizer data\tokenizers\minimind_bpe_16k_110k.json --system "You are a concise, accurate assistant." --prompt "What is SFT?" --max-new-tokens 128 --temperature 0.7 --top-k 40 --top-p 0.9
```

`chat.py` serializes the user message with the same project-owned template as
SFT and appends the assistant generation header. `-X utf8` avoids Windows
console mojibake when printing Chinese text.

### Data-scale ablation

For a controlled 2K vs 20K SFT comparison, make a deterministic subset of
the existing 20K training split. The validation split stays unchanged, and the
JSON report stores input/output SHA-256 hashes and the selection seed:

```powershell
python scripts\create_sft_subset.py --input data\processed\minimind_sft\sft_train_20k_seed42.jsonl --output data\processed\minimind_sft\sft_train_2k_seed42.jsonl --report artifacts\sft_prepare_2k_from_20k_seed42.json --size 2000 --seed 42
python trainer\train_full_sft.py --train-data data\processed\minimind_sft\sft_train_2k_seed42.jsonl --val-data data\processed\minimind_sft\sft_val_1k_seed42.jsonl --tokenizer data\tokenizers\minimind_bpe_16k_110k.json --pretrained-checkpoint checkpoints\minimind_dense_v2_20k.pt --checkpoint checkpoints\minimind_dense_sft_2k_2ep.pt --epochs 2 --batch-size 1 --grad-accum-steps 16 --max-length 768 --empty-think-ratio 0.2 --system-prompt-ratio 0.2 --lr 1e-5 --min-lr 1e-6 --amp-dtype bfloat16
```

In one local reproduction with the same 1K held-out validation set, two
epochs, and the same initialization, 20K SFT achieved assistant-only PPL
18.00 versus 31.65 for the 2K subset. On a fixed 10-prompt repetition suite,
sampled 20K SFT had distinct-2 0.465 and repeated-4-gram fraction 0.407;
the 2K run collapsed to repeated `</think>` tokens (distinct-2 0.055,
repeated-4-gram 0.944). This fixed-epoch comparison also changes the number
of optimizer steps (250 vs 2,500), so it measures the value of the larger SFT
training budget rather than isolating data diversity alone.

## Tests

```powershell
python -m unittest discover -s test -p "test_*.py" -v
```

## Current scope

The repository currently provides the MiniMind decoder-only Dense/MoE baseline,
its project-owned tokenizer and pretraining pipeline, assistant-only SFT,
pairwise DPO, grouped GRPO/RLAIF with a frozen reward model, and corresponding
evaluation/generation/diagnostic tools. MiniMind-V is the next implementation
stage; ORPO and production-scale distributed training are outside the current
verified scope.
