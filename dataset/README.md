# Dataset directory

This directory mirrors the upstream MiniMind repository layout and documents
the expected local datasets. Raw and processed samples remain under the local
`data/` directory and are excluded from Git because they are large and may
carry upstream redistribution restrictions.

Use the preparation utilities in `scripts/` to build these local stages:

```text
data/raw/        downloaded source records
data/processed/  cleaned and deterministically split JSONL/text
data/prepared/   tokenized pretraining streams
data/tokenizers/ trained tokenizer artifacts
```

Every experiment should retain its provenance report under `artifacts/`.
