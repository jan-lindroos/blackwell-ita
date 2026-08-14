# blackwell-ita

A principled approach to multi-preference, inference-time alignment, built on the **Blackwell winner** (Bhatia et al. 2021, [arXiv:2105.01850](https://arxiv.org/abs/2105.01850)).

## Experiment design

**Datasets**

- **HelpSteer2**: criteria = helpfulness, correctness, coherence, plus overall preference from HelpSteer2-Preference. Verbosity and complexity are dropped: they are descriptive rather than monotone quality criteria, and a max-min winner would chase them.
- **Community Alignment** (English subset): the annotator pool spans the US and India, so groups = country × age, k = 4: {US, India} × {18–34, 35+}. Each annotation is a bare choice, so per-annotation targets are binary and a pair's target is the mean over its annotators — annotator disagreement survives as fractional targets.

**Methods** compared over a pool of N base-policy samples (N swept in powers of two): base policy, Bradley–Terry best-of-N, mean- and worst-criterion best-of-N, Best-of-Nash (the k = 1 Blackwell winner), and the Blackwell winner. The two extra baselines isolate what mixing over candidates adds (worst-criterion best-of-N vs Blackwell) and what per-criterion information adds (Best-of-Nash vs Blackwell).

**Base policy**: RLHFlow LLaMA3-SFT 8B. **Preference models**: multi-head pairwise reward models on a Qwen2.5-3B backbone, trained here on the human labels.

**Metrics**: primary = worst-criterion / worst-group win rate from the held-out evaluation model; secondary = overall expected win rate (win + draw/2) judged by Claude (`claude-sonnet-5`, order-swap debiased) against two anchors: the dataset reference answer and a reserved base-policy sample. Policies are scored in expectation over their support atoms (no sampling).

## Layout

```
src/
  human_prefs.py       preference pairs from HelpSteer2 / Community Alignment
  train_rms.py         multi-head reward-model training
  generate.py          candidate sampling from the base policy
  winners.py           Blackwell / Nash / best-of-N policies (LP solve)
  judge.py             Claude pairwise judging via the CLI
  artifacts.py         HF Hub locations and transfer helpers
notebooks/                 numbered in run order
  01_prep-human-prefs      dataset preparation
  02_train-rms             reward-model training          (GPU, molab)
  03_generate-candidates   65 samples per eval prompt     (GPU, molab)
  04_score-preferences     cache all RM inference to Hub  (GPU, molab)
  05_local-experiments     LP solves + Claude judging     (laptop, no GPU)
  06_results               figures and summary tables     (laptop)
tests/               pytest suite
```

## Running

```sh
uv sync
uv run marimo edit notebooks/
```

GPU notebooks carry their own inline dependencies and run in marimo's sandbox on molab (CUDA). Judging requires the `claude` CLI on PATH. CI (GitHub Actions) runs ruff, basedpyright, `marimo check`, and the fast tests.
