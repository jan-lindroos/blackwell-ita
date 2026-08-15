# blackwell-ita

A principled approach to multi-preference, inference-time alignment, built on the Blackwell winner (Bhatia et al. 2021, [arXiv:2105.01850](https://arxiv.org/abs/2105.01850)).

## Experiment design

### Datasets

`HelpSteer2` supplies the quality criteria: helpfulness, correctness and coherence, plus overall preference from `HelpSteer2-Preference`.

`Community-Alignment` supplies the groups. The annotator pool spans the US and India (the English subset of the dataset), so the k=4 groups are the annotator subpopulations formed by crossing country with age: US or India, aged 18-34 or 35+. A pair's target is the mean of its annotators' binary choices.

Between them, the datasets cover both combining multiple qualities and serving annotator subpopulations, echoing the notion of group fairness.

### Methods

All methods select from a pool of N base-policy samples: 

- the base policy
- Bradley-Terry best-of-N
- mean- and worst-criterion best-of-N (pairwise and Bradley-Terry variants)
- Best-of-Nash (special case of k=1 Blackwell winner)
- and the Blackwell winner. 

The extra baselines isolate what mixing over candidates contributes (worst-criterion best-of-N vs Blackwell), what per-criterion information contributes (Best-of-Nash vs Blackwell) and what the pairwise representation of intransitive preferences contributes (the Bradley-Terry variants vs their pairwise counterparts).

### Models

The base policy is `RLHFlow LLaMA3-SFT 8B`. The preference models are multi-head pairwise reward models on a `Qwen3-4B-Instruct` backbone, trained here on the human labels.

### Metrics

The primary metric is the worst-criterion or worst-group win rate from the held-out evaluation model. The secondary metric is the overall expected win rate (win + draw/2) against the dataset reference answer, judged by Claude (`claude-sonnet-5`, order-swap debiased) at N = 1, 4, 16 and 64. Policies are scored in expectation over their support atoms, never by sampling.

## Running

```sh
uv sync
uv run marimo edit notebooks/
```

Notebooks 2, 3 and 4 are set up to run on molab with GPU.
