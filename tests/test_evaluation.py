import numpy as np
import pandas as pd
import pytest

from blackwell_ita.evaluation import (
    BASELINES,
    MAIN_METHODS,
    METHOD_TARGETS,
    METHOD_WELFARES,
    WELFARE_METHODS,
    WELFARES,
    amplified_error,
    blackwell_loss,
    bootstrap_summary,
    budget_sweep,
    decisive_accuracy,
    deployed_policy,
    error_sweep,
    kind_subsets,
    method_results,
    model_error,
    optimal_losses,
    paired_gaps,
    pool_size_results,
    robust_welfare,
    sampling_results,
    selection_policies,
    targeting_results,
    welfare_optima,
)
from blackwell_ita.scoring import rewards_to_preferences
from blackwell_ita.selection import blackwell_winner, robust_nash_winner

WHOLE_GROUP = [(0, 1, 2, 3)]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def random_tensor(seed: int, group_count: int = 3, size: int = 5) -> np.ndarray:
    return rewards_to_preferences(
        np.random.default_rng(seed).normal(size=(group_count, size))
    )


def evaluation_inputs() -> tuple[pd.DataFrame, dict, dict]:
    random_generator = np.random.default_rng(1)
    selector_tensors, gold_tensors = (
        {
            question_id: rewards_to_preferences(random_generator.normal(size=(4, 6)))
            for question_id in ["q1", "q2"]
        }
        for _ in range(2)
    )
    return pd.DataFrame({"question_id": ["q1", "q2"]}), selector_tensors, gold_tensors


def test_robust_welfare_takes_each_index_at_its_worst_comparator():
    tensor = np.array([[[0.5, 0.9], [0.1, 0.5]], [[0.5, 0.2], [0.8, 0.5]]])
    # The first candidate ties itself and the groups see 0.9 and 0.2 against the other
    welfare = robust_welfare(np.array([1.0, 0.0]), tensor)
    assert welfare["egalitarian"] == pytest.approx(0.2)
    assert welfare["utilitarian"] == pytest.approx(0.5)
    assert welfare["nash"] == pytest.approx(np.sqrt(0.9 * 0.2))


@pytest.mark.parametrize("seed", range(3))
def test_welfare_optima_are_not_beaten_by_random_policies(seed):
    tensor = random_tensor(seed)
    optima = welfare_optima(tensor)
    assert optima["utilitarian"] == pytest.approx(0.5, abs=1e-5)
    random_generator = np.random.default_rng(seed)
    for _ in range(30):
        welfare = robust_welfare(random_generator.dirichlet(np.ones(5)), tensor)
        for name in WELFARES:
            assert welfare[name] <= optima[name] + 1e-5


def test_blackwell_loss_is_the_shortfall_below_each_target():
    tensor = random_tensor(0)
    policy = np.full(5, 0.2)
    welfare = robust_welfare(policy, tensor)
    optima = welfare_optima(tensor)
    assert blackwell_loss(policy, tensor, "orthant", optima["nash"]) == pytest.approx(
        0.5 - welfare["egalitarian"]
    )
    assert blackwell_loss(
        policy, tensor, "half_space", optima["nash"]
    ) == pytest.approx(0.5 - welfare["utilitarian"])
    assert blackwell_loss(policy, tensor, "nash", welfare["nash"]) == 0.0
    assert blackwell_loss(policy, tensor, "nash", optima["nash"]) > 0.0
    assert blackwell_loss(
        blackwell_winner(tensor), tensor, "orthant", optima["nash"]
    ) == pytest.approx(optimal_losses(optima)["orthant"], abs=1e-6)


def test_selection_policies_add_best_of_n_only_for_bradley_terry_selectors():
    tensor = rewards_to_preferences(
        np.array([[2.0, 1.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]])
    )
    policies = selection_policies(tensor, True, 0.4, 0.05)
    assert "best_of_n" in policies
    assert "best_of_n" not in selection_policies(tensor, False, 0.4, 0.05)
    assert set(policies) >= set(METHOD_WELFARES)
    for policy in policies.values():
        assert policy.shape == (4,)
        assert policy.sum() == pytest.approx(1.0)
    assert policies["best_of_n"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_method_results_grade_each_subgroup_with_the_gold():
    groups, selector_tensors, gold_tensors = evaluation_inputs()
    results = method_results(
        groups, selector_tensors, gold_tensors, "s", WHOLE_GROUP + PAIRS, True
    )
    assert len(results) == 2 * 9 * 7
    assert set(results["group_size"]) == {2, 4}
    assert set(results["selector"]) == {"s"}
    regrets = results[[f"{name}_regret" for name in WELFARES]]
    assert (regrets >= -1e-6).all().all()
    blackwell = results[results["method"] == "blackwell"]
    assert blackwell["blackwell_regret"].to_numpy() == pytest.approx(  # pyright: ignore[reportAttributeAccessIssue]
        blackwell["egalitarian_regret"].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    )
    assert results.loc[results["method"] == "uniform", "blackwell_regret"].isna().all()


def test_method_results_with_the_gold_selecting_have_no_regret_on_each_own_welfare():
    groups, _, gold_tensors = evaluation_inputs()
    exact = method_results(groups, gold_tensors, gold_tensors, "gold", WHOLE_GROUP)
    for method_name, welfare in METHOD_WELFARES.items():
        own_regret = exact.loc[exact["method"] == method_name, f"{welfare}_regret"]
        assert own_regret.abs().max() < 1e-4
    nash_target = exact[exact["method"] == "blackwell_nash"]
    assert nash_target["blackwell_regret"].abs().max() < 1e-4  # pyright: ignore[reportAttributeAccessIssue]
    assert set(METHOD_TARGETS) <= set(exact["method"])


def test_method_results_pair_selection_ignores_the_other_participants():
    groups, selector_tensors, gold_tensors = evaluation_inputs()
    pair_only = method_results(groups, selector_tensors, gold_tensors, "s", [(0, 2)])
    changed = {
        question_id: tensor.copy() for question_id, tensor in selector_tensors.items()
    }
    for tensor in changed.values():
        tensor[[1, 3]] = 0.5
    pd.testing.assert_frame_equal(
        pair_only, method_results(groups, changed, gold_tensors, "s", [(0, 2)])
    )


def test_model_error_and_amplified_error():
    gold = random_tensor(1)
    selector = random_tensor(2)
    assert model_error(gold, gold).tolist() == [0.0, 0.0, 0.0]
    assert amplified_error(gold, selector, 1.0) == pytest.approx(selector)
    amplified = amplified_error(gold, selector, 4.0)
    assert amplified.min() >= 0.0 and amplified.max() <= 1.0
    assert amplified + amplified.swapaxes(1, 2) == pytest.approx(
        np.ones_like(amplified)
    )
    assert (model_error(amplified, gold) >= model_error(selector, gold) - 1e-9).all()


def test_error_sweep_records_the_factor_and_its_model_error():
    groups, selector_tensors, gold_tensors = evaluation_inputs()
    sweep = error_sweep(groups, selector_tensors, gold_tensors, [0.0, 2.0])
    assert set(sweep["factor"]) == {0.0, 2.0}
    assert set(sweep["group_size"]) == {4}
    assert (sweep.loc[sweep["factor"] == 0.0, "model_error"] < 1e-12).all()
    assert sweep.loc[sweep["factor"] == 2.0, "model_error"].min() > 0.0


def test_budget_sweep_starts_from_the_level_free_programme():
    groups, selector_tensors, gold_tensors = evaluation_inputs()
    sweep = budget_sweep(groups, selector_tensors, gold_tensors, [0.0, 2.0])
    assert sweep.columns.tolist() == [
        "question_id",
        "budget_factor",
        "model_error",
        "nash_regret",
    ]
    assert len(sweep) == 4
    for question_id in ["q1", "q2"]:
        level_free = robust_welfare(
            robust_nash_winner(selector_tensors[question_id]), gold_tensors[question_id]
        )["nash"]
        zero_budget = sweep[
            (sweep["question_id"] == question_id) & (sweep["budget_factor"] == 0.0)
        ]["nash_regret"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]
        optimum = welfare_optima(gold_tensors[question_id])["nash"]
        assert zero_budget == pytest.approx(optimum - level_free, abs=1e-4)


def test_deployed_policy_puts_mass_on_a_dominant_candidate_when_it_is_drawn():
    tensor = rewards_to_preferences(np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    deployed = deployed_policy(
        tensor, 2, 50, blackwell_winner, np.random.default_rng(0)
    )
    replay = np.random.default_rng(0)
    drawn_pools = [replay.integers(3, size=2) for _ in range(50)]
    assert deployed[0] == pytest.approx(np.mean([0 in pool for pool in drawn_pools]))
    assert deployed.sum() == pytest.approx(1.0)


def test_sampling_results_cover_every_pool_size_and_target():
    groups, _, gold_tensors = evaluation_inputs()
    sampling = sampling_results(groups, gold_tensors, [1, 4], draw_count=3)
    assert len(sampling) == 2 * 2 * 3
    assert set(sampling["target"]) == {"orthant", "half_space", "nash"}
    assert (sampling["regret"] > -1e-6).all()
    # A pool of 200 draws holds all six candidates, so the pool game is the full game
    exhausted = sampling_results(groups, gold_tensors, [200], draw_count=1)
    assert exhausted["regret"].abs().max() < 1e-4


def test_bootstrap_summary_brackets_each_group_mean():
    frame = pd.DataFrame(
        {
            "g": ["a"] * 4 + ["b"] * 4,
            "question_id": [f"q{index}" for index in range(8)],
            "value": [1.0, 2.0, 3.0, 4.0] + [10.0] * 4,
        }
    )
    summary = bootstrap_summary(frame, ["g"], "value").set_index("g")
    assert summary.columns.tolist() == [
        "value",
        "interval_low",
        "interval_high",
        "instances",
    ]
    assert summary.loc["b"].tolist() == [10.0, 10.0, 10.0, 4]
    assert summary.loc["a", "interval_low"] <= 2.5 <= summary.loc["a", "interval_high"]


def test_bootstrap_summary_counts_rows_of_one_question_once():
    frame = pd.DataFrame(
        {"g": ["a"] * 3, "question_id": ["q1", "q1", "q2"], "value": [0.0, 2.0, 4.0]}
    )
    summary = bootstrap_summary(frame, ["g"], "value").iloc[0]
    assert summary["value"] == pytest.approx(2.5)
    assert summary["instances"] == 2


def test_selection_policies_run_only_the_named_methods():
    policies = selection_policies(
        random_tensor(0), True, 0.4, method_names=MAIN_METHODS
    )
    assert set(policies) == set(MAIN_METHODS)


def test_method_results_record_welfare_optimum_and_regret_together():
    groups, selector_tensors, gold_tensors = evaluation_inputs()
    results = method_results(
        groups, selector_tensors, gold_tensors, "s", WHOLE_GROUP, True, MAIN_METHODS
    )
    assert set(results["method"]) == set(MAIN_METHODS)
    for name in WELFARES:
        assert (
            results[f"{name}_optimum"] - results[f"{name}_welfare"]
        ).to_numpy() == pytest.approx(results[f"{name}_regret"].to_numpy())


def test_kind_subsets_hold_the_requested_count_of_each_kind():
    kinds = np.array(["consensus"] * 4 + ["participant"] * 4)
    subsets = kind_subsets(
        kinds, {"consensus": 2, "participant": 1}, 5, np.random.default_rng(0)
    )
    assert len(subsets) == 5
    for subset in subsets:
        assert sorted(kinds[subset].tolist()) == ["consensus"] * 2 + ["participant"]
        assert subset.tolist() == sorted(set(subset.tolist()))
    whole = kind_subsets(kinds, {"participant": 4}, 5, np.random.default_rng(0))
    assert [subset.tolist() for subset in whole] == [[4, 5, 6, 7]]


def kinds_by_question(size: int = 6) -> dict[str, np.ndarray]:
    kinds = np.array(["consensus", "participant"] * (size // 2))
    return {"q1": kinds, "q2": kinds}


def test_pool_size_results_with_the_whole_pool_and_the_gold_reach_each_optimum():
    groups, selector_tensors, gold_tensors = evaluation_inputs()
    results = pool_size_results(
        groups, gold_tensors, gold_tensors, kinds_by_question(), [2, 6], 3, False
    )
    assert len(results) == 2 * 2 * (len(MAIN_METHODS) - 1)
    whole_pool = results[results["pool_size"] == 6]
    for welfare, method_name in WELFARE_METHODS.items():
        own = whole_pool.loc[whole_pool["method"] == method_name, f"{welfare}_regret"]
        assert own.abs().max() < 1e-4
    small_pools = pool_size_results(
        groups, selector_tensors, gold_tensors, kinds_by_question(), [2], 3
    )
    assert (small_pools[[f"{name}_regret" for name in WELFARES]] > -1e-6).all().all()


def test_targeting_results_grade_each_level_inside_its_own_candidates():
    groups, _, gold_tensors = evaluation_inputs()
    levels = {
        "consensus": {"consensus": 3},
        "mixed": {"consensus": 1, "participant": 2},
    }
    results = targeting_results(
        groups, gold_tensors, gold_tensors, kinds_by_question(), levels, 4, False
    )
    assert set(results["level"]) == set(levels)
    assert len(results) == 2 * 2 * (len(MAIN_METHODS) - 1)
    for welfare, method_name in WELFARE_METHODS.items():
        own = results.loc[results["method"] == method_name, f"{welfare}_regret"]
        assert own.abs().max() < 1e-4
    consensus_only = gold_tensors["q1"][:, [0, 2, 4]][:, :, [0, 2, 4]]
    expected_loss = 0.5 - welfare_optima(consensus_only)["egalitarian"]
    observed = results[
        (results["question_id"] == "q1") & (results["level"] == "consensus")
    ]
    assert observed["orthant_optimal_loss"].to_numpy() == pytest.approx(  # pyright: ignore[reportAttributeAccessIssue]
        expected_loss
    )


def test_paired_gaps_are_never_negative_with_the_gold_selecting():
    groups, _, gold_tensors = evaluation_inputs()
    results = method_results(
        groups, gold_tensors, gold_tensors, "gold", WHOLE_GROUP, True, MAIN_METHODS
    )
    gaps = paired_gaps(results, ["selector"])
    assert set(gaps["baseline"]) == set(BASELINES)
    assert set(gaps["welfare"]) == set(WELFARES)
    assert (gaps["value"] > -1e-4).all()
    assert (gaps["instances"] == 2).all()


def test_decisive_accuracy_counts_agreement_on_the_gold_decisive_pairs():
    gold = rewards_to_preferences(np.array([[0.0, 2.0, 4.0]]))
    assert decisive_accuracy(gold, gold) == 1.0
    assert decisive_accuracy(1.0 - gold, gold) == 0.0
