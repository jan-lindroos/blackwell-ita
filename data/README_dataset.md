---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
tags:
  - preference-learning
  - reward-modeling
  - helpsteer2
---

# HelpSteer2 preference pairs, split for the Blackwell ITA experiment

One file, `pairs.parquet`: 10,681 preference pairs built from
[nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), carrying
the split labels the whole project keys on. Everything downstream filters this
file rather than re-deriving a split, so the labels here are the single source
of truth.

## Splits

| `split` | pairs | source | role |
|---|---|---|---|
| `train` | 10,162 | HelpSteer2 `train` | trains the preference model — nothing is carved out of it |
| `evaluation` | 419 | HelpSteer2 `validation` | drives early stopping and the reported per-criterion metrics |
| `ita_holdout` | 100 | HelpSteer2 `validation` | the inference-time-alignment prompts; never trained or scored on |

Split by prompt, never by row, so no prompt's pairs straddle two sets.
HelpSteer2's own train and validation halves share no prompt at all, so
`train` is disjoint from the other two by construction rather than by sampling.

`ita_holdout` is drawn only from pairs carrying a **decisive** overall
preference — neither missing nor tied at 0.5 — because the anchor selection
downstream skips those, and sampling from all 519 validation pairs would
silently yield fewer than 100 anchors.

## Columns

| column | meaning |
|---|---|
| `prompt` | the instruction |
| `response_a`, `response_b` | the two responses being compared |
| `helpfulness`, `correctness`, `coherence`, `complexity`, `verbosity` | graded win probability for response A, from the rating margin |
| `overall` | graded overall preference; **missing on 1,556 rows**, masked out in training |
| `split` | one of the three labels above |

Targets are graded, not binary: a rating margin of ±2 or more maps to 1.0/0.0,
±1 to 0.75/0.25, and a tie to 0.5. The tie rate varies a lot by criterion —
74% for `complexity` against 29% for `helpfulness` — which is why per-criterion
accuracy is reported over decisive pairs only, and why those counts differ.

## Building a candidate pool from `ita_holdout`

For each of the 100 prompts, the anchor is the human-preferred response of its
pair, and the pool is 128 samples from the base policy plus that anchor at
index 128. From the project repo:

```bash
python scripts/build_candidates.py --pairs-path pairs.parquet
python scripts/score_candidates.py \
    --checkpoint <pairwise_lr5e-6_bf16.pt> --dtype bfloat16 --batch-size 128
```

The first samples `RLHFlow/LLaMA3-SFT-v2` at temperature 1.0 and
`max_new_tokens=1024`; the second turns each pool into the skew-symmetrised
preference tensor the Blackwell winner is solved over. Both resume from their
own output, so a job that runs out of walltime can simply be resubmitted.

Scoring is `n(n-1)` forward passes per prompt — the LP constrains the policy
against every pure opponent column, so no entry of the matrix is spare — which
is roughly 1.65M forwards over the 100 prompts. Budget it accordingly, and
check the reported `pairs/min` on the first prompt before queuing the rest.
