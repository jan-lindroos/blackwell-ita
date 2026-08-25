# blackwell-ita

Multi-preference inference-time alignment on HelpSteer2, built on the Blackwell winner (Bhatia et al. 2021, [arXiv:2105.01850](https://arxiv.org/abs/2105.01850)). Everything lives in self-contained marimo notebooks that hand artifacts to each other through the HF Hub:

1. `notebooks/train_prefs.py` — build preference pairs (5 attributes + overall), split prompts, train the pairwise 6-head model per half, score preference tensors (GPU).
2. `notebooks/generate_candidates.py` — anchors and N = 128 base-policy candidates per evaluation prompt (GPU).
3. `notebooks/experiments.py` — best-of-Blackwell vs best-of-Nash vs base, held-out worst-criterion win rates, and a Claude-judged overall win rate (local).

## Running

```sh
uv sync
uv run marimo edit notebooks/
```

The GPU notebooks carry inline script dependencies and run on molab; sign in with `hf auth login` first.
