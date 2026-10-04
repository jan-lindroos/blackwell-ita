import json

import numpy as np
import pandas as pd
import pytest

from blackwell_ita import helpsteer_cli as cli
from blackwell_ita.helpsteer import HELPSTEER_PREFIX, pairwise_normals, pool_prefix
from blackwell_ita.judging import comparison_id


def test_cross_scorer_best_of_n_comparison_matches_prompts():
    results = pd.DataFrame(
        [
            {
                "backbone": "generator",
                "selector": selector,
                "method": method,
                "prompt_id": prompt,
                **dict.fromkeys(cli.RESULT_METRICS, value),
            }
            for selector, method, values in [
                ("pairwise", "blackwell_learned", [("a", 0.4), ("b", 0.8)]),
                ("bt", "best_of_n_weighted", [("b", 0.7), ("a", 0.3)]),
            ]
            for prompt, value in values
        ]
    )
    summary = cli.compare_best_of_n(results)
    assert len(summary) == len(cli.RESULT_METRICS)
    assert set(summary.selector) == {"pairwise"}
    assert set(summary.baseline_selector) == {"bt"}
    np.testing.assert_allclose(
        summary[["difference", "interval_low", "interval_high"]], 0.1
    )
    with pytest.raises(ValueError, match="identical unique prompts"):
        cli.compare_best_of_n(results.iloc[:-1])


def test_local_workflow_freezes_tuning_and_only_judges_evaluation_support(
    monkeypatch, tmp_path
):
    backbone, selector, model = (
        "RLHFlow/LLaMA3-SFT-v2",
        "qwen3_4b_pairwise",
        "explicit-model",
    )
    prompts = pd.DataFrame(
        {
            "prompt_id": ["t", "e"],
            "prompt": ["tuning prompt", "evaluation prompt"],
            "anchor": ["anchor t", "anchor e"],
            "split": ["tuning", "evaluation"],
        }
    )
    candidates = pd.DataFrame(
        [
            {
                "prompt_id": key,
                "sample_index": i,
                "response": f"{key}-{i}",
                "temperature": 1.2,
                "max_new_tokens": 1024,
                "seed": 1810,
                "backbone": backbone,
            }
            for key in ["t", "e"]
            for i in range(64)
        ]
    )
    scores = {key: np.full((4, 65, 65), 0.5) for key in ["t", "e"]}
    storage = {}

    def read_frame(repository, filename, prefix):
        if filename == "prompts.parquet":
            assert prefix == HELPSTEER_PREFIX
            return prompts.copy()
        if filename == "pairs.parquet":
            return pd.DataFrame()
        assert filename == "candidates.parquet" and prefix == pool_prefix(backbone)
        return candidates.copy()

    monkeypatch.setattr(cli, "read_hub_dataframe", read_frame)
    # Full 50/200 manifest validation is covered separately; this is a tiny I/O fixture.
    monkeypatch.setattr(cli, "validate_prompts", lambda *args: None)
    monkeypatch.setattr(cli, "download_tensors", lambda *args: scores)
    monkeypatch.setattr(
        cli, "hub_file_exists", lambda repo, name, prefix: (prefix, name) in storage
    )

    def upload_file(repo, path, prefix):
        storage[(prefix, path.name)] = path.read_bytes()

    monkeypatch.setattr(cli, "upload_hub_file", upload_file)

    def download_file(repo, name, prefix):
        path = tmp_path / ("remote_" + name)
        path.write_bytes(storage[(prefix, name)])
        return path

    monkeypatch.setattr(cli, "download_hub_file", download_file)
    monkeypatch.setattr(
        cli,
        "upload_dataframe",
        lambda repo, name, frame, prefix: storage.update(
            {(prefix, name): frame.copy()}
        ),
    )
    judged_prompts, fitted = [], []

    def judge(comparisons, model, prefix, directory):
        judged_prompts.append(set(comparisons.prompt))
        return pd.DataFrame(
            [
                {
                    "comparison_id": comparison_id(
                        row.prompt, row.response, row.anchor, model
                    ),
                    **dict.fromkeys(cli.HELPSTEER_CRITERIA, 1.0),
                }
                for row in comparisons.itertuples()
            ]
        )

    monkeypatch.setattr(cli, "sync_judgments", judge)

    def fit(games, outcomes):
        fitted.append(len(games))
        assert len(games) == len(outcomes) == 1
        np.testing.assert_array_equal(outcomes[0], np.ones(64))
        return {
            "normals": pairwise_normals(np.full(6, 0.5)).tolist(),
            "thresholds": [0.5] * 6,
        }

    monkeypatch.setattr(cli, "fit_target", fit)
    with pytest.raises(ValueError, match="before evaluation"):
        cli.run("evaluate", backbone, selector, model, tmp_path)
    assert judged_prompts == []
    cli.run("tune", backbone, selector, model, tmp_path)
    parameters_key = (cli.judge_prefix(backbone, model), f"{selector}_parameters.json")
    frozen = storage[parameters_key]
    assert json.loads(frozen)["tuning_prompt_ids"] == ["t"]
    cli.run("tune", backbone, selector, model, tmp_path)
    assert fitted == [1] and judged_prompts == [{"tuning prompt"}]
    cli.run("evaluate", backbone, selector, model, tmp_path)
    assert judged_prompts == [{"tuning prompt"}, {"evaluation prompt"}]
    assert storage[parameters_key] == frozen
    results = storage[
        (cli.judge_prefix(backbone, model), f"{selector}_results.parquet")
    ]
    assert set(results.prompt_id) == {"e"}
    assert set(results.method) == {
        "blackwell_fixed",
        "blackwell_learned",
        "blackwell_overall",
    }
    candidates.loc[0, "response"] = "changed input"
    with pytest.raises(ValueError, match="Frozen parameters"):
        cli.run("evaluate", backbone, selector, model, tmp_path)
    assert len(judged_prompts) == 2
