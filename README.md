# blackwell-ita

Multi-preference inference-time alignment on HelpSteer2, built on the Blackwell winner (Bhatia et al. 2021, [arXiv:2105.01850](https://arxiv.org/abs/2105.01850)). Everything lives in self-contained marimo notebooks that hand artifacts to each other through the HF Hub:

1. `notebooks/train_prefs.py` — build preference pairs (5 attributes + overall), split prompts, train the pairwise 6-head model per half, score the inference-half preference tensors (GPU).
2. `notebooks/generate_candidates.py` — anchors and N = 128 base-policy candidates per evaluation prompt (GPU).
3. `notebooks/experiments.py` — best-of-Blackwell (exact and entropic, plus a verbosity-ablation arm over 4 criteria) vs best-of-Nash vs base, held-out worst-criterion win rates (the evaluate-half model scores only policy support atoms against the anchor, on demand; GPU), token efficiency (expected response tokens per policy, the check on verbosity chasing), and a Claude-judged overall win rate (local).

## Entropic best-of-Blackwell

The experiments notebook solves the Blackwell winner in two variants through one function, `blackwell_winner(tensor, thresholds, beta)`. Both minimise the worst clipped shortfall against pure opponents on the orthant target `S = {z : z_g >= tau_g}` with `tau = 1/2` per head, written in epigraph form over the simplex: variables `(pi, t)` with constraints `sum_i pi_i P_g[i, j] + t >= tau_g` for every head `g` and pool column `j`, `sum pi = 1`, `pi >= 0` and `t >= 0`. The `t >= 0` bound implements the clipping.

At `beta = 0` the objective is `t` and the problem is the exact linear programme. At `beta > 0` the objective becomes `t + beta * sum_i pi_i log pi_i`, which equals `t + beta * KL(pi || uniform)` up to the additive constant `beta * log n`. That is the entropically regularised objective of the theory, an exponential-cone programme. Both are modelled in CVXPY (the entropy via `cp.entr`) and solved by Clarabel, an open-source interior-point cone solver that ships with CVXPY. This replaced the previous `scipy.optimize.linprog` path entirely, so one constraint construction serves both variants. A non-optimal status or a solver exception raises rather than returning a policy.

The notebook runs `entropic_blackwell` at `beta = 0.05` over the 5 criterion heads at every pool size, alongside the exact `best_of_blackwell` and `best_of_nash` (the `beta = 0` solver on the overall head alone). The entropic policy flows into the selections artifact and the held-out worst-criterion plot but is excluded from Claude judging. The KL term forces strictly positive mass on every candidate, so judging its support atoms would cost roughly `n` calls per prompt, and the tensor metric already scores it exactly as an expectation. The accepted optimisation error is `beta * log n`, about 0.24 at `n = 128`.

Tests pin the solver from independent angles: the exact variant against a from-scratch von Neumann max-min LP and a grid search on the definition, and the entropic variant on its fixed points (a preference cycle stays uniform at every `beta`), full support, equal mass on duplicated candidates, the excess-value bound `v(pi_beta) <= v(pi_lp) + beta * log n`, and interpolation towards the LP vertex as `beta` shrinks and towards uniform as it grows. Solution tolerances are 1e-6 rather than the old 1e-9 because interior-point solutions land near, not on, the optimal vertex.

## Running

```sh
uv sync
uv run marimo edit notebooks/
```

The GPU notebooks carry inline script dependencies and run on molab; sign in with `hf auth login` first.
