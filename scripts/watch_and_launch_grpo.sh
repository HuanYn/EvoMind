#!/usr/bin/env bash
# Wait for GPU 0 or 1 to become genuinely idle, then launch the one-epoch
# MiniMind RLAIF-GRPO run exactly once. Intended for a shared server.
set -euo pipefail

project_dir="${PROJECT_DIR:-$HOME/HappyLLM}"
conda_bin="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
poll_seconds="${POLL_SECONDS:-60}"
free_memory_mib="${FREE_MEMORY_MIB:-500}"
log_dir="$project_dir/logs"
model_dir="$project_dir/models/internlm2-1_8b-reward"

mkdir -p "$log_dir"
cd "$project_dir"

gpu_is_free() {
    local gpu="$1"
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
    [[ -n "$used" && "$used" -lt "$free_memory_mib" ]]
}

reward_model_ready() {
    [[ -s "$model_dir/model-00001-of-00002.safetensors" && -s "$model_dir/model-00002-of-00002.safetensors" ]]
}

while true; do
    if reward_model_ready; then
        for gpu in 0 1; do
            if gpu_is_free "$gpu"; then
                timestamp="$(date +%Y%m%dT%H%M%S)"
                log_file="$log_dir/minimind_grpo_rlaif_1ep_${timestamp}_gpu${gpu}.log"
                echo "$(date -Is) launching MiniMind GRPO on GPU $gpu" | tee -a "$log_file"
                CUDA_VISIBLE_DEVICES="$gpu" "$conda_bin" run -n base python -B trainer/train_grpo.py \
                    --policy-checkpoint checkpoints/minimind_dense_sft_2ep.pt \
                    --train-data data/processed/minimind_rlaif/rlaif_train_15k_seed42.jsonl \
                    --val-data data/processed/minimind_rlaif/rlaif_val_500_seed42.jsonl \
                    --tokenizer data/tokenizers/minimind_bpe_16k_110k.json \
                    --reward-mode minimind_rlaif \
                    --reward-model models/internlm2-1_8b-reward \
                    --reward-device cuda \
                    --checkpoint checkpoints/minimind_dense_grpo_rlaif_1ep.pt \
                    --epochs 1 --batch-size 1 --grad-accum-steps 1 --num-generations 6 \
                    --max-length 768 --max-new-tokens 64 \
                    --temperature 0.8 --top-k 40 --top-p 0.9 \
                    --epsilon 0.2 --beta 0.1 --lr 3e-7 --min-lr 3e-8 --warmup-steps 0 \
                    --eval-interval 250 --eval-prompts 4 --save-interval 250 --amp-dtype float16 \
                    2>&1 | tee -a "$log_file"
                exit "${PIPESTATUS[0]}"
            fi
        done
    fi
    sleep "$poll_seconds"
done
