#!/bin/bash
# One-time setup on a Bridges-2 LOGIN node, run from the project root on Ocean:
#
#     cd $PROJECT/blackwell-ita && scripts/psc_setup.sh
#
# Installs uv, resolves the environment (the cu126 torch build the H100 nodes'
# driver needs) and verifies the pieces a job cannot recover from: the pairs
# file, the wandb key, and the import path into notebooks/.
set -euo pipefail

: "${PROJECT:?PROJECT is unset - run this on a Bridges-2 login node}"

# Every large cache goes to Ocean. $HOME is capped at 25 GB and the venv alone
# (torch + CUDA libraries) is several GB before a single model is downloaded.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.uv_cache}"
export HF_HOME="${HF_HOME:-$PROJECT/hf_cache}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== resolving the environment (first run downloads ~3 GB of wheels) =="
uv sync

echo "== checks =="
uv run python - <<'PY'
import sys
from pathlib import Path

import torch

root = Path.cwd()
sys.path.insert(0, str(root / "notebooks"))
import train_prefs  # noqa: F401  the scripts import it the same way

pairs = root / "data" / "pairs.parquet"
print(f"torch {torch.__version__}")
assert "+cu" in torch.__version__, (
    f"{torch.__version__} is not a CUDA build; check [tool.uv.sources] in"
    " pyproject.toml resolved the cu126 index"
)
print(f"data/pairs.parquet present: {pairs.exists()}")
assert pairs.exists(), "upload data/pairs.parquet before submitting jobs"
print("notebooks/train_prefs.py imports cleanly")
PY

if [ -s "$HOME/.wandb_key" ]; then
    echo "wandb key file present at ~/.wandb_key"
else
    echo
    echo "!! ~/.wandb_key is missing. Get your key from https://wandb.ai/authorize, then:"
    echo "     printf '%s\\n' '<your-key>' > ~/.wandb_key && chmod 600 ~/.wandb_key"
fi

echo
echo "== GPU partitions this cluster actually offers =="
echo "   (check the type string before submitting; psc_smoke.sbatch assumes h100-80)"
sinfo -o "%20P %10D %30G" | grep -i gpu || true

echo
echo "== prefetching the Qwen3-4B backbone into \$HF_HOME =="
echo "   (~8 GB; doing it here rather than inside a job saves GPU-hours)"
uv run python - <<'PY'
from transformers import AutoModel, AutoTokenizer

name = "Qwen/Qwen3-4B-Instruct-2507"
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print(f"cached {name}")
PY

echo
echo "setup complete. Submit the smoke test with:"
echo "    sbatch scripts/psc_smoke.sbatch"
