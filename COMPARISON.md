# BT reward model vs. pairwise preference model

A controlled comparison of two ways to learn human preferences from the same
data: a **Bradley-Terry (BT) reward model** that scores one response at a
time, and a **pairwise preference model** that judges two responses jointly.
Everything about their training is identical except the model structure.

## Dataset: HelpSteer2

[nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
provides pairs of responses to the same prompt, annotated two ways:

- **Per-attribute ratings** (0-4) on each response for five criteria:
  helpfulness, correctness, coherence, complexity, verbosity.
- **An overall preference** between the two responses
  (`preference_strength`, -3..3).

We turn each annotated pair into one training example with **six graded
targets** — the probability that response A beats response B on each
criterion — by mapping rating margins onto a five-level grid:

| margin (A - B) | <= -2 | -1 | 0 | +1 | >= +2 |
|---|---|---|---|---|---|
| target P(A > B) | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |

The sixth ("overall") target comes from `preference_strength` the same way;
pairs without a preference annotation have it masked out of the loss.

**Splits** (`data/pairs.parquet`, seed 1810): 100 prompts held out for
evaluation; the rest split into two halves ("inference": 5,031 pairs,
"evaluate": 5,031 pairs). Both models train on the **same half** (default:
inference). Within it, 10% of prompts form the validation set — split by
prompt, not by row, so no prompt leaks between train and validation.

## Shared backbone

Both models are the same network except for their input format:

```
text -> Qwen3-4B-Instruct (decoder, fp32, bf16 autocast, grad checkpointing)
     -> hidden state of the last non-padding token   (size 2560)
     -> one linear layer                             (2560 -> 6)
     -> six logits, one per criterion
```

## The two models

**BT reward model** (`scripts/train_bt_reward.py`). Input is a prompt and
**one** response:

```
{prompt}\n\n[RESPONSE]\n{response}   ->   r(prompt, response) in R^6
```

Each response gets six absolute scores. The preference prediction for a pair
is the score difference, squashed to a probability — the Bradley-Terry model:

```
P(A > B on criterion g) = sigmoid( r_g(A) - r_g(B) )
```

Two forward passes per pair. After training you have a standalone scorer:
"how good is this single response?" needs no comparison partner.

**Pairwise preference model** (`scripts/train_pairwise_preference.py`).
Input is a prompt and **both** responses in one sequence:

```
{prompt}\n\n[RESPONSE 1]\n{A}\n\n[RESPONSE 2]\n{B}   ->   six logits
P(A > B on criterion g) = sigmoid( logit_g )
```

One forward pass per pair. Its attention can compare the responses
token-by-token (which BT structurally cannot), but it only answers relative
questions — it has no notion of a single response's absolute quality.

## Identical training protocol

Both scripts share the notebook's training code (`notebooks/train_prefs.py`):

- Loss: binary cross-entropy of each criterion's logit against its graded
  target, masked where the target is missing.
- Augmentation: every pair also appears with responses swapped and targets
  flipped (for BT this is mathematically a no-op, kept so both models see
  the identical example stream).
- Optimizer: AdamW, lr 1e-5, linear warmup 100 steps, batch of 8 pairs
  (`--batch-size`; the notebook default of 12 exceeds 80GB H100 memory).
- Schedule: validate every 1/3 epoch ("round"); stop after 2 rounds without
  a new best validation loss; restore the best checkpoint.
- Same seed (1810): same head initialization, same batch order, same
  validation pairs.

The only degree of freedom left is the model structure — so any difference
in the reported metrics is attributable to it.

## Evaluation sets

- **Validation** (10% of the training half's prompts, ~503 pairs): drives
  early stopping and checkpoint selection; its metrics are reported but
  mildly optimistic, since it picked the checkpoint.
- **Test** (the *other* prompt half, 5,031 pairs): never touched by
  training or checkpoint selection — the headline numbers. Both models are
  tested on the identical pairs (per-criterion loss and decisive accuracy;
  skip with `--skip-test`, subsample with `--test-prompts N`).

## What the comparison asks

Does reading both responses jointly predict human preferences better than
scoring each response in isolation? The pairwise model is strictly more
expressive; BT pays ~2x compute per training pair but yields reusable
absolute scores and needs only n (not n^2) forwards to rank n candidates.

## Running (PSC Bridges-2)

```
sbatch --gpus=h100-80:1 -t 0:30:00 scripts/psc_train.sbatch \
  uv run python scripts/train_bt_reward.py \
  --pairs-path data/pairs.parquet --limit-prompts 32 --max-rounds 2   # smoke
```

- `--limit-prompts N` — tiny subsample for smoke tests.
- `--max-rounds N` — cap rounds; prints min/round for walltime estimates.
- No flags — the full experiment.

Outputs: `checkpoints/bt_<half>.pt` / `checkpoints/pairwise_<half>.pt`,
live curves in the `blackwell-ita-rm-comparison` wandb project, and final
per-criterion metrics in the job log.
