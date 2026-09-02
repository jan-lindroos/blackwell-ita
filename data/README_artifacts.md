---
license: cc-by-4.0
tags:
  - inference-time-alignment
  - reward-modeling
  - helpsteer2
---

# ITA candidate pools and preference tensors

Derived artifacts for the Blackwell inference-time-alignment experiment, over
the 100 `ita_holdout` prompts of
[blackwell-ita/helpsteer2-splits](https://huggingface.co/datasets/blackwell-ita/helpsteer2-splits).
Those prompts come from HelpSteer2's validation half and are disjoint from
everything the reward models trained on.

| file | contents |
|---|---|
| `helpsteer2/anchors.parquet` | one anchor per prompt: the human-preferred response of its pair |
| `helpsteer2/candidates.parquet` | 129 rows per prompt — 128 samples at `sample_index` 0..127 plus the anchor at 128 |
| `helpsteer2/preference_tensors.npz` | win probabilities from the **pairwise** model, `tensor_{i}` shaped (head, 129, 129) |
| `helpsteer2/preference_tensors_bt.npz` | the same from a **Bradley-Terry** model |

## How the pools were sampled

`RLHFlow/LLaMA3-SFT-v2` at temperature 1.0, `max_new_tokens=1024`, 128 samples
per prompt. The responses are far shorter than that cap in practice: a median
of 239 tokens, which is why sampling took under an hour rather than the day a
cap-length estimate predicts.

The anchor rides at index 128 so a tensor covers it in the same pass. Pool
prefixes slice past it, so the LP never sees it: `tensor[..., :n, :n]` for
n up to 128 is candidates only.

## The two tensors

Both are win probabilities per criterion head, and both are skew-symmetric
with a 0.5 diagonal — but they are computed differently, which is the point of
having both.

The pairwise model reads both responses in one input, so every ordered pair
needs its own forward: 129x128 = 16,512 per prompt, 1.65M over the hold-out,
and the result has to be skew-symmetrised afterwards because the model is not
order-invariant. That run took 9h45m on one H100.

The Bradley-Terry model scores each response alone, so 129 forwards per prompt
suffice and `sigmoid(r_i - r_j)` gives the whole matrix — exactly
skew-symmetric and exactly 0.5 on the diagonal by construction. That run took
3m20s: the same 100 pools, 175 times faster.

On the 419-pair evaluation split the pairwise model is the better of the two
(overall loss 0.585 against 0.600), so the tensors are not interchangeable.
Solving the winner on both is a check on how much the alignment result depends
on which reward model produced it.

## Loading

```python
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

REPO = "blackwell-ita/artifacts"
pools = pd.read_parquet(hf_hub_download(REPO, "helpsteer2/candidates.parquet", repo_type="dataset"))
tensors = np.load(hf_hub_download(REPO, "helpsteer2/preference_tensors.npz", repo_type="dataset"))
tensors["prompts"]        # the canonical order tensor_{i} is keyed on
tensors["criteria"]       # head order: 5 attributes then overall
tensors["tensor_0"].shape # (6, 129, 129)
```

`tensor_{i}` belongs to the i-th entry of `prompts`; the key is positional, so
that order is what makes the artifact readable.
