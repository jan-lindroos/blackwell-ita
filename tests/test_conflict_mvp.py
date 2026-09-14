import conflict_mvp
import numpy as np
import pandas as pd
import pytest
from conflict_mvp import (
    comparison_table,
    evaluate_policy,
    objectives,
    select_policies,
    winner,
)


def test_flip_copies_and_preserves_reciprocity():
    tensor = np.full((6, 2, 2), 0.5)
    tensor[3] = [[0.5, 0.8], [0.2, 0.5]]
    flipped = objectives(tensor)
    np.testing.assert_allclose(flipped[3], [[0.5, 0.2], [0.8, 0.5]])
    np.testing.assert_allclose(flipped + flipped.transpose(0, 2, 1), 1)
    assert tensor[3, 0, 1] == 0.8


def test_same_policies_evaluated_by_both_graders(monkeypatch):
    selector_tensor = np.full((6, 4, 4), 0.5)
    evaluator_tensor = selector_tensor.copy()
    selector_tensor[:3, :2, -1] = 0.8
    evaluator_tensor[:3, :2, -1] = 0.2
    metadata = {"prompts": np.array(["p"]), "criteria": np.array(conflict_mvp.HEADS)}
    calls = []

    def select(tensor, scenario, norm):
        calls.append(tensor)
        return {"blackwell": np.array([0.25, 0.75])}

    monkeypatch.setattr(conflict_mvp, "select_policies", select)
    candidates = pd.DataFrame(
        {"prompt": ["p", "p"], "sample_index": [0, 1], "tokens": [100, 200]}
    )
    results = conflict_mvp.run_experiment(
        {**metadata, "tensor_0": selector_tensor},
        {**metadata, "tensor_0": evaluator_tensor},
        candidates,
        n=2,
    ).set_index("grader")
    assert len(calls) == 1
    assert calls[0].shape == (6, 2, 2)
    assert results["n"].tolist() == [2, 2]
    assert results.loc["Phi (independent)", "minimum"] == pytest.approx(0.2)
    assert results.loc["Qwen (self-evaluation)", "minimum"] == pytest.approx(0.5)
    assert results["tokens"].tolist() == [175, 175]


def test_shared_objectives_select_dominant_candidate():
    preferred = np.array([[0.5, 0.9], [0.1, 0.5]])
    tensor = np.stack([preferred] * 6)
    tensor[3] = 1 - preferred
    # Opposite overall preferences must not affect any of the three methods.
    tensor[5] = 1 - preferred
    for policy in select_policies(tensor).values():
        np.testing.assert_allclose(policy, [1, 0], atol=1e-6)


def test_conflicting_heads_require_mixture():
    head = np.array([[0.5, 0.9], [0.1, 0.5]])
    policy = winner(np.stack([head, 1 - head]))
    np.testing.assert_allclose(policy, [0.5, 0.5], atol=1e-6)
    assert min((head.T @ policy).min(), ((1 - head).T @ policy).min()) == pytest.approx(
        0.3
    )


def test_evaluation_uses_expectation_before_aggregation():
    anchors = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.8], [0.6, 0.6], [0, 0], [0, 0]])
    metrics = evaluate_policy(np.array([0.5, 0.5]), anchors, [100, 300])
    assert metrics["minimum"] == pytest.approx(0.4)
    assert metrics["arithmetic"] == pytest.approx(0.55)
    assert metrics["geometric"] == pytest.approx((0.5 * 0.5 * 0.8 * 0.4) ** 0.25)
    assert metrics["tokens"] == 200
    anchors[0] = 0
    assert evaluate_policy(np.array([0.5, 0.5]), anchors, [100, 300])["geometric"] == 0


def test_paired_bootstrap_aligns_prompts_and_averages_prompt_minima():
    rows = []
    for method, delta in [("blackwell", 0.1), ("scalarized_nash", 0), ("best_of_n", 0)]:
        for prompt, score in [("a", 0.2), ("b", 0.6)]:
            rows.append(
                {
                    "prompt": prompt,
                    "method": method,
                    "minimum": score + delta,
                    "geometric": score + delta,
                    "arithmetic": score + delta,
                    "tokens": 100,
                }
            )
    result = comparison_table(
        pd.DataFrame(rows).sample(frac=1, random_state=2)
    ).set_index("metric")
    assert result.loc["minimum", "blackwell"] == pytest.approx(0.5)
    assert (
        result.loc["minimum", "Blackwell − scalarized_nash [95% CI]"]
        == "+0.1000 [+0.1000, +0.1000]"
    )


def test_flipped_correctness_drops_complexity_for_selection_and_evaluation():
    preferred = np.array([[0.5, 0.9], [0.1, 0.5]])
    tensor = np.stack([preferred] * 6)
    tensor[1] = 1 - preferred
    tensor[3:] = 1 - preferred
    for policy in select_policies(tensor, "flipped_correctness").values():
        np.testing.assert_allclose(policy, [1, 0], atol=1e-6)
    anchor = np.array([[0.8], [0.2], [0.8], [0.0], [0.0], [0.0]])
    result = evaluate_policy(np.array([1.0]), anchor, [100], "flipped_correctness")
    assert result["minimum"] == pytest.approx(0.8)
    assert result["geometric"] == pytest.approx(0.8)
    assert result["arithmetic"] == pytest.approx(0.8)
    assert "simplicity" not in result and "correctness" not in result
    assert result["incorrectness"] == pytest.approx(0.8)


def test_helpful_incorrect_ignores_coherence_and_complexity():
    preferred = np.array([[0.5, 0.9], [0.1, 0.5]])
    tensor = np.stack(
        [preferred, 1 - preferred, 1 - preferred, preferred, preferred, preferred]
    )
    for policy in select_policies(tensor, "helpful_incorrect").values():
        np.testing.assert_allclose(policy, [1, 0], atol=1e-6)
    anchors = np.array([[0.8], [0.4], [0.0], [1.0], [0.0], [0.0]])
    result = evaluate_policy(np.array([1.0]), anchors, [100], "helpful_incorrect")
    assert result["minimum"] == pytest.approx(0.6)
    assert result["arithmetic"] == pytest.approx(0.7)
    assert result["geometric"] == pytest.approx(np.sqrt(0.48))
    assert "coherence" not in result and "simplicity" not in result


@pytest.mark.parametrize("norm", ["1", "2", "inf"])
def test_norm_solver_matches_dense_two_candidate_search(norm):
    tensor = np.array(
        [[[0.5, 0.95], [0.05, 0.5]], [[0.5, 0.3], [0.7, 0.5]], [[0.5, 0.6], [0.4, 0.5]]]
    )
    order = np.inf if norm == "inf" else int(norm)
    grid = np.linspace(0, 1, 10001)
    policies = np.stack([grid, 1 - grid], axis=1)
    deficits = np.maximum(0.5 - np.einsum("bi,gij->bgj", policies, tensor), 0)
    optimum = np.linalg.norm(deficits, ord=order, axis=1).max(axis=1).min()
    policy = winner(tensor, norm)
    actual = np.linalg.norm(
        np.maximum(0.5 - np.einsum("i,gij->gj", policy, tensor), 0), ord=order, axis=0
    ).max()
    assert actual == pytest.approx(optimum, abs=3e-5)
