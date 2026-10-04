import numpy as np
import pandas as pd
import pytest

from blackwell_ita import helpsteer
from blackwell_ita.helpsteer import (
    HELPSTEER_POOL_SIZE,
    candidate_pools,
    fit_scalar_weights,
    helpsteer_prompts,
    pairwise_normals,
    selection_inputs,
    selection_policies,
    validate_prompts,
)
from blackwell_ita.helpsteer_cli import evaluate_policies, input_fingerprint
from blackwell_ita.scoring import rewards_to_preferences
from blackwell_ita.selection import best_of_n, borda


def test_target_refinement_reaches_values_between_coarse_grid_points(monkeypatch):
    optimum = np.r_[np.full(6, 0.375), np.full(6, 0.625)]

    def policy(game, normals, thresholds):
        weights = np.array(
            [
                normals[i, first]
                for i, (first, _) in enumerate(helpsteer.CRITERION_PAIRS)
            ]
        )
        score = 1 - np.mean((np.r_[weights, thresholds] - optimum) ** 2)
        return np.array([score, 1 - score])

    monkeypatch.setattr(helpsteer, "target_set_winner", policy)
    result = helpsteer.fit_target([np.full((4, 2, 2), 0.5)], [np.array([1.0, 0.0])])
    np.testing.assert_array_equal(result["pair_weights"], optimum[:6])
    np.testing.assert_array_equal(result["thresholds"], optimum[6:])
    assert result["tuning_version"] == helpsteer.TUNING_VERSION


def source_pairs():
    return pd.DataFrame(
        {
            "prompt": [f"test {i}" for i in range(300)]
            + [f"train {i}" for i in range(30)],
            "response_a": "a",
            "response_b": "b",
            "overall": [0.25, 0.75] * 165,
            "split": ["test"] * 300 + ["train"] * 30,
        }
    )


def test_manifest_has_exact_split_and_uses_only_rm_test():
    pairs = source_pairs()
    prompts = helpsteer_prompts(pairs)
    assert prompts.split.value_counts().to_dict() == {"evaluation": 200, "tuning": 50}
    assert set(prompts.prompt) <= set(pairs.loc[pairs.split == "test", "prompt"])
    expected_anchor = pairs.set_index("prompt").overall.map(
        lambda y: "a" if y > 0.5 else "b"
    )
    assert prompts.anchor.tolist() == prompts.prompt.map(expected_anchor).tolist()
    validate_prompts(prompts, pairs)
    changed = prompts.copy()
    changed.loc[0, "split"] = "evaluation"
    with pytest.raises(ValueError):
        validate_prompts(changed, pairs)


def test_manifest_refuses_insufficient_or_cross_split_prompts():
    with pytest.raises(ValueError, match="Need 250"):
        helpsteer_prompts(source_pairs().iloc[:100])
    pairs = pd.concat([source_pairs(), source_pairs().iloc[:1].assign(split="train")])
    with pytest.raises(ValueError, match="crosses"):
        helpsteer_prompts(pairs)


def candidate_frame():
    return pd.DataFrame(
        {
            "prompt_id": "p",
            "sample_index": range(64),
            "response": "r",
            "temperature": 1.2,
            "max_new_tokens": 1024,
            "seed": 1810,
            "backbone": "generator",
        }
    )


def test_pool_rejects_missing_indices_and_stale_generation_settings():
    prompts = pd.DataFrame({"prompt_id": ["p"]})
    assert len(candidate_pools(prompts, candidate_frame())["p"]) == 64
    with pytest.raises(ValueError, match="indices"):
        candidate_pools(prompts, candidate_frame().iloc[:-1])
    with pytest.raises(ValueError, match="temperature"):
        candidate_pools(prompts, candidate_frame().assign(temperature=1.0))


def test_generation_uses_fixed_size_temperature_and_records_settings(monkeypatch):
    called = {}

    def generate(model, prompts, count, device, **kwargs):
        called.update(model=model, prompts=prompts, count=count, **kwargs)
        return pd.DataFrame(
            {"prompt_index": 0, "sample_index": range(count), "response": "r"}
        )

    monkeypatch.setattr(helpsteer, "generate_responses", generate)
    prompts = pd.DataFrame({"prompt_id": ["p"], "prompt": ["instruction"]})
    frame = helpsteer.generate_helpsteer_candidates("generator", prompts, "cpu")
    assert called["count"] == 64 and called["temperature"] == 1.2
    assert called["max_new_tokens"] == 1024
    assert len(candidate_pools(prompts, frame)["p"]) == 64


def test_scalar_best_of_n_is_not_borda_of_bt_probabilities():
    rewards = np.array([[100.0, 1.0, 0.0], [0.0, 5.0, 10.0]])
    assert best_of_n(rewards).argmax() == 0
    assert borda(rewards_to_preferences(rewards)).argmax() == 2
    with pytest.raises(ValueError, match="scalar rewards"):
        best_of_n(rewards_to_preferences(rewards))


def test_anchor_excluded_and_every_blackwell_runs_for_both_scorers():
    rewards = np.tile(np.arange(HELPSTEER_POOL_SIZE + 1, dtype=float), (4, 1))
    rewards[:, -1] = 10000  # An overwhelming anchor must still be unselectable.
    parameters = {
        "normals": pairwise_normals(np.full(6, 0.5)).tolist(),
        "thresholds": [0.5] * 6,
        "scalar_weights": [0.25] * 4,
    }
    for kind, scores in [
        ("bradley_terry", rewards),
        ("pairwise", rewards_to_preferences(rewards)),
    ]:
        policies = selection_policies(scores, kind, parameters)
        assert {
            "blackwell_fixed",
            "blackwell_learned",
            "blackwell_overall",
        } <= policies.keys()
        assert all(len(p) == 64 and np.isclose(p.sum(), 1) for p in policies.values())
        assert policies["blackwell_overall"].argmax() == 63
        if kind == "bradley_terry":
            np.testing.assert_allclose(
                policies["blackwell_overall"], policies["best_of_n_overall"], atol=1e-5
            )
        else:
            assert not any(name.startswith("best_of_n") for name in policies)
    with pytest.raises(ValueError):
        selection_inputs(rewards[:, :64], "bradley_terry")


def test_scalar_weights_optimise_pool_selections_against_claude():
    rewards = [np.array([[3.0, 0], [3.0, 0], [3.0, 0], [0, 1.0]])]
    scores = [np.array([0.0, 1.0])]
    weights = np.array(fit_scalar_weights(rewards, scores))
    assert best_of_n(rewards[0], weights).argmax() == 1
    assert weights.sum() == pytest.approx(1)


def test_missing_selected_judgment_fails_instead_of_renormalising():
    policy = {"p": {"method": np.array([0.5, 0.5])}}
    outcomes = {"p": np.array([[1.0, np.nan]] * 4)}
    with pytest.raises(ValueError, match="Missing Claude"):
        evaluate_policies(policy, outcomes)
    outcomes["p"][:, 1] = 0
    result = evaluate_policies(policy, outcomes).iloc[0]
    assert result.overall == 0.5
    assert result.rawlsian_welfare == 0.5


def test_fingerprint_changes_on_scores_candidates_or_split():
    prompts = pd.DataFrame({"prompt_id": ["p"], "split": ["tuning"]})
    candidates = candidate_frame()
    scores = {"p": np.zeros((4, 65))}
    original = input_fingerprint(prompts, candidates, scores)
    assert original != input_fingerprint(
        prompts.assign(split="evaluation"), candidates, scores
    )
    assert original != input_fingerprint(
        prompts, candidates.assign(response="changed"), scores
    )
    assert original != input_fingerprint(prompts, candidates, {"p": np.ones((4, 65))})
