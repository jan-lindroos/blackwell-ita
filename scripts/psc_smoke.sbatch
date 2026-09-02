#!/bin/bash
#SBATCH -J blackwell-smoke
#SBATCH -N 1
#SBATCH -p GPU-shared            # confirm against `sinfo`; see psc_setup.sh output
#SBATCH --gpus=h100-80:1         # confirm the type string against `sinfo`
#SBATCH -t 01:30:00
#SBATCH -o slurm-smoke-%j.out

# Stage 1 of the cluster workup: does the merged code train, evaluate, save a
# checkpoint and report to wandb on a real GPU? Both reward models run back to
# back on 32 prompts for 2 rounds, which is minutes of compute, not hours.
#
#     sbatch scripts/psc_smoke.sbatch
#     tail -f slurm-smoke-<jobid>.out
#
# Add an allocation with `sbatch -A <grantid> ...` if `projects` shows more
# than one.
set -euo pipefail

# uv lives in ~/.local/bin; batch jobs inherit the submitting shell's PATH,
# which does not always include it
export PATH="$HOME/.local/bin:$PATH"
# The fp32 AdamW fine-tune leaves little headroom on an 80GB H100; expandable
# segments reclaims the ~10GB the default allocator loses to fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Jobs inherit the submitting shell's env, which does not always carry the
# wandb key; fall back to the one-time key file (create with:
#   printf '%s\n' "<key>" > ~/.wandb_key && chmod 600 ~/.wandb_key )
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat "$HOME/.wandb_key" 2>/dev/null || true)}"
# Gated repos (Llama among them) need a token, and HF_HOME points at Ocean
# below, so a token file written by `hf auth login` in a plain shell lands
# somewhere the job does not look. The env var wins over the file either way.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.hf_token" 2>/dev/null || true)}"
# Keep model downloads and wandb files on Ocean, not the small home quota
export HF_HOME="${HF_HOME:-${PROJECT:-$PWD}/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT:-$PWD}/wandb_logs}"
mkdir -p "$HF_HOME" "$WANDB_DIR"

if [ -z "$WANDB_API_KEY" ]; then
    echo "no wandb key found; logging offline. Sync afterwards with:" >&2
    echo "    uv run wandb sync $WANDB_DIR/wandb/offline-run-*" >&2
    export WANDB_MODE=offline
fi

# Smoke-test sizes. Micro-batch 4 with accumulation 3 keeps the effective batch
# at the default 12 while a third of the activations are live at once, so a
# first run cannot fail for memory alone. Stage 2 re-times at the real
# --batch-size 12 --accumulation 1 to check that it fits.
PROMPTS="${PROMPTS:-32}"
ROUNDS="${ROUNDS:-2}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-3}"
TAG="${TAG:-smoke-$SLURM_JOB_ID}"

echo "== node =="
hostname
nvidia-smi
echo
echo "== torch sees =="
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

for script in train_bt_reward train_pairwise_preference; do
    echo
    echo "=============== $script ==============="
    uv run python "scripts/$script.py" \
        --pairs-path data/pairs.parquet \
        --limit-prompts "$PROMPTS" \
        --max-rounds "$ROUNDS" \
        --test-prompts 32 \
        --batch-size "$BATCH" \
        --accumulation "$ACCUM" \
        --inference-batch-size 4 \
        --warmup-steps 10 \
        --eval-every 5 \
        --eval-pairs 32 \
        --output-dir "checkpoints/$TAG" \
        --wandb-run-name "$TAG-$script"
done

echo
echo "smoke test finished; checkpoints in checkpoints/$TAG"
