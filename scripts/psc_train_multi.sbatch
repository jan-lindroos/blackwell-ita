#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:4         # confirm the type string against `sinfo`
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out

# The multi-GPU counterpart of psc_train.sbatch, for encoders whose fp32
# optimiser state does not fit one card: 8B needs 120 GiB of weights,
# gradients and Adam moments before a single activation is allocated.
# FSDP shards those across the requested GPUs.
#
#     sbatch scripts/psc_train_multi.sbatch \
#         scripts/train_pairwise_preference.py \
#         --pairs-path data/pairs.parquet --distributed \
#         --encoder meta-llama/Llama-3.1-8B-Instruct --batch-size 4
#
# Unlike psc_train.sbatch this takes the script path and its flags, not a
# whole command: the launcher has to be `accelerate launch`, not `uv run`.
# GPUS overrides the process count if you change --gpus above.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# NCCL's NVLink SHARP multicast fails to bind on these nodes ("Failed to bind
# NVLink SHARP (NVLS) Multicast memory ... CUDA error 1 'invalid argument'"),
# which the error itself attributes to the Fabric Manager or NVSwitches rather
# than to the job. Collectives fall back to the ring algorithm without it.
export NCCL_NVLS_ENABLE=0
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat "$HOME/.wandb_key" 2>/dev/null || true)}"
# Gated repos (Llama among them) need a token, and HF_HOME points at Ocean
# below, so a token file written by `hf auth login` in a plain shell lands
# somewhere the job does not look. The env var wins over the file either way.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.hf_token" 2>/dev/null || true)}"
export HF_HOME="${HF_HOME:-${PROJECT:-$PWD}/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT:-$PWD}/wandb_logs}"
mkdir -p "$HF_HOME" "$WANDB_DIR"

if [ -z "$WANDB_API_KEY" ]; then
    echo "no wandb key found; logging offline. Sync afterwards with:" >&2
    echo "    uv run wandb sync $WANDB_DIR/wandb/offline-run-*" >&2
    export WANDB_MODE=offline
fi

# One process per allocated GPU. SLURM_GPUS_ON_NODE is what the job actually
# got, which is the number to trust over anything hardcoded above
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
echo "== $GPUS GPU(s) =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# The FSDP settings live in a config file rather than on the command line:
# passed as flags, use_orig_params silently stayed false, which flattens every
# parameter into one 1-D buffer and breaks the embedding lookup with
# "'weight' must be 2-D".
LAUNCH=(--config_file scripts/fsdp_config.yaml --num_processes "$GPUS")
# Which class to shard at. Left out, accelerate reads the model's
# _no_split_modules, which is version-dependent; set WRAP_CLS to pin it
# (LlamaDecoderLayer, Qwen3DecoderLayer) if that lookup comes up empty.
if [ -n "${WRAP_CLS:-}" ]; then
    LAUNCH+=(--fsdp_transformer_layer_cls_to_wrap "$WRAP_CLS")
fi

uv run accelerate launch "${LAUNCH[@]}" "$@"
