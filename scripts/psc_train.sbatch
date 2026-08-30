#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared            # adjust to your allocation's GPU partition
#SBATCH --gpus=h100-80:1         # adjust the type string to what sinfo reports
#SBATCH -t 12:00:00              # walltime; smoke and timing runs need far less
#SBATCH -o slurm-%j.out

# Runs whatever command follows the script name, from the project root on
# Ocean. Examples:
#   sbatch -t 0:30:00 scripts/psc_train.sbatch \
#       uv run python scripts/train_bt_reward.py \
#       --pairs-path data/pairs.parquet --limit-prompts 32 --max-rounds 2
#   sbatch scripts/psc_train.sbatch \
#       uv run python scripts/train_bt_reward.py --pairs-path data/pairs.parquet

set -euo pipefail
# uv lives in ~/.local/bin; batch jobs inherit the submission shell's PATH,
# which does not always include it
export PATH="$HOME/.local/bin:$PATH"
# The fp32 AdamW fine-tune leaves little headroom on an 80GB H100; expandable
# segments reclaims the ~10GB the default allocator loses to fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Jobs inherit the submitting shell's env, which does not always carry the
# wandb key; fall back to the one-time key file (create with:
#   printf '%s\n' "<key>" > ~/.wandb_key && chmod 600 ~/.wandb_key )
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || true)}"
# Keep model downloads and wandb files on Ocean, not the small home quota
export HF_HOME="${PROJECT:-$PWD}/hf_cache"
export WANDB_DIR="${PROJECT:-$PWD}/wandb_logs"
mkdir -p "$HF_HOME" "$WANDB_DIR"
"$@"
