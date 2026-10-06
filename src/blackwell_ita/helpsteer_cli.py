"""Local Claude judging, frozen tuning, and evaluation for the HelpSteer notebook."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from blackwell_ita.helpsteer import (
    GENERATION_BACKBONES,
    GENERATION_TEMPERATURE,
    HELPSTEER_CRITERIA,
    HELPSTEER_POOL_SIZE,
    HELPSTEER_PREFIX,
    TUNING_REGULARIZATION,
    TUNING_VERSION,
    candidate_pools,
    fit_scalar_weights,
    fit_target,
    pool_prefix,
    selection_inputs,
    selection_policies,
    validate_prompts,
)
from blackwell_ita.hub import (
    ARTIFACTS_REPOSITORY,
    PAIRS_FILENAME,
    SPLITS_REPOSITORY,
    download_hub_file,
    hub_file_exists,
    read_hub_dataframe,
    upload_dataframe,
    upload_hub_file,
)
from blackwell_ita.judging import JUDGE_VERSION, comparison_id, judge_comparisons
from blackwell_ita.scoring import download_tensors
from blackwell_ita.selection import paired_bootstrap_interval, welfare
from blackwell_ita.training import REWARD_MODEL_CONFIGURATIONS

RESULT_METRICS = [
    *HELPSTEER_CRITERIA,
    "egalitarian_welfare",
    "nash_welfare",
    "utilitarian_welfare",
]


def input_fingerprint(
    prompts: pd.DataFrame, candidates: pd.DataFrame, scores: dict[str, np.ndarray]
) -> str:
    """Bind parameters to the exact pools, split, scores and experiment settings."""
    digest = hashlib.sha256()
    for frame in [
        prompts.sort_values("prompt_id"),
        candidates.sort_values(["prompt_id", "sample_index"]),
    ]:
        digest.update(json.dumps(frame.to_dict("records"), sort_keys=True).encode())
    for key in sorted(scores):
        array = np.ascontiguousarray(scores[key])
        digest.update(key.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            [
                HELPSTEER_CRITERIA,
                HELPSTEER_POOL_SIZE,
                GENERATION_TEMPERATURE,
                TUNING_REGULARIZATION,
                TUNING_VERSION,
                JUDGE_VERSION,
            ]
        ).encode()
    )
    return digest.hexdigest()


def judge_prefix(backbone: str, model: str) -> str:
    """All selectors share one model/rubric-specific cache per generation pool."""
    tag = hashlib.sha256(f"{model}:{JUDGE_VERSION}".encode()).hexdigest()[:16]
    return f"{pool_prefix(backbone)}/claude-{tag}"


def sync_judgments(
    comparisons: pd.DataFrame, model: str, prefix: str, directory: Path
) -> pd.DataFrame:
    """Merge local unfinished work with the Hub cache, then resume comparisons."""
    path = directory / "judgments.parquet"
    directory.mkdir(parents=True, exist_ok=True)
    if hub_file_exists(ARTIFACTS_REPOSITORY, path.name, prefix):
        remote = read_hub_dataframe(ARTIFACTS_REPOSITORY, path.name, prefix)
        if path.exists():
            remote = pd.concat(
                [remote, pd.read_parquet(path)], ignore_index=True
            ).drop_duplicates("comparison_id", keep="last")
        remote.to_parquet(path, index=False)
    return judge_comparisons(
        comparisons,
        model,
        path,
        checkpoint=lambda file: upload_hub_file(ARTIFACTS_REPOSITORY, file, prefix),
    )


def atom_comparisons(
    prompts: pd.DataFrame,
    pools: dict[str, list[str]],
    supports: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    """All tuning atoms or the union of held-out policy supports, with anchors."""
    return pd.DataFrame(
        [
            {"prompt": row["prompt"], "response": response, "anchor": row["anchor"]}
            for row in prompts.to_dict("records")
            for index, response in enumerate(pools[row["prompt_id"]])
            if supports is None or index in supports[row["prompt_id"]]
        ]
    )


def outcome_arrays(
    prompts: pd.DataFrame,
    pools: dict[str, list[str]],
    judgments: pd.DataFrame,
    model: str,
) -> dict[str, np.ndarray]:
    """NaN marks genuinely unjudged candidates, never a fabricated tie."""
    lookup = judgments.drop_duplicates("comparison_id").set_index("comparison_id")
    results = {}
    for row in prompts.to_dict("records"):
        ids = [
            comparison_id(row["prompt"], response, row["anchor"], model)
            for response in pools[row["prompt_id"]]
        ]
        results[row["prompt_id"]] = (
            lookup.reindex(ids)[HELPSTEER_CRITERIA].to_numpy(dtype=float).T
        )
    return results


def evaluate_policies(
    policies: dict[str, dict[str, np.ndarray]], outcomes: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Expected Claude scores and welfare; require complete selected support."""
    rows = []
    for prompt_id, methods in policies.items():
        for method, policy in methods.items():
            support = policy > 0
            scores = outcomes[prompt_id][:, support]
            if not np.isfinite(scores).all():
                raise ValueError(
                    "Missing Claude judgment on selected support; resume evaluation"
                )
            values = scores @ policy[support]
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "method": method,
                    **dict(zip(HELPSTEER_CRITERIA, values.tolist(), strict=True)),
                    **welfare(values),
                }
            )
    return pd.DataFrame(rows)


def result_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Prompt means with paired learned-minus-baseline bootstrap intervals."""
    rows = []
    for metric in RESULT_METRICS:
        table = results.pivot(index="prompt_id", columns="method", values=metric)
        if table.isna().any().any():
            raise ValueError("Every method must have the same evaluation prompts")
        for method in table:
            difference = (table["blackwell_learned"] - table[method]).to_numpy()
            low, high = paired_bootstrap_interval(difference)
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "prompts": len(table),
                    "mean": table[method].mean(),
                    "learned_minus_method": difference.mean(),
                    "interval_low": low,
                    "interval_high": high,
                }
            )
    return pd.DataFrame(rows)


def compare_best_of_n(results: pd.DataFrame) -> pd.DataFrame:
    """Paired Blackwell-minus-BT-best-of-N comparisons across scorer families.

    Input rows must come from one judge/model version, as in the notebook.
    Match complete prompt sets within each generator, never unpaired means.
    """
    rows = []
    for backbone, group in results.groupby("backbone"):
        baselines = group[
            group.method.isin(["best_of_n_overall", "best_of_n_weighted"])
        ]
        blackwells = group[group.method.str.startswith("blackwell_")]
        for _, selected in blackwells.groupby(["selector", "method"]):
            selector, method = selected.iloc[0][["selector", "method"]]
            selected = selected.set_index("prompt_id")
            for _, baseline in baselines.groupby(["selector", "method"]):
                baseline_selector, baseline_method = baseline.iloc[0][
                    ["selector", "method"]
                ]
                baseline = baseline.set_index("prompt_id")
                if (
                    not selected.index.is_unique
                    or not baseline.index.is_unique
                    or set(selected.index) != set(baseline.index)
                ):
                    raise ValueError(
                        "Cross-scorer comparisons require identical unique prompts"
                    )
                baseline = baseline.reindex(selected.index)
                for metric in RESULT_METRICS:
                    difference = (selected[metric] - baseline[metric]).to_numpy()
                    if not np.isfinite(difference).all():
                        raise ValueError(
                            "Cross-scorer comparisons require complete scores"
                        )
                    low, high = paired_bootstrap_interval(difference)
                    rows.append(
                        {
                            "backbone": backbone,
                            "selector": selector,
                            "method": method,
                            "baseline_selector": baseline_selector,
                            "baseline_method": baseline_method,
                            "metric": metric,
                            "prompts": len(difference),
                            "difference": difference.mean(),
                            "interval_low": low,
                            "interval_high": high,
                        }
                    )
    return pd.DataFrame(rows)


def run(stage: str, backbone: str, selector: str, model: str, work_dir: Path) -> None:
    """No generation or model loading here; consume notebook-produced Hub artifacts."""
    configuration = next(c for c in REWARD_MODEL_CONFIGURATIONS if c.name == selector)
    prefix = pool_prefix(backbone)
    prompts = read_hub_dataframe(
        ARTIFACTS_REPOSITORY, "prompts.parquet", HELPSTEER_PREFIX
    )
    pairs = read_hub_dataframe(SPLITS_REPOSITORY, PAIRS_FILENAME, HELPSTEER_PREFIX)
    validate_prompts(prompts, pairs)
    candidates = read_hub_dataframe(ARTIFACTS_REPOSITORY, "candidates.parquet", prefix)
    pools = candidate_pools(prompts, candidates)
    if not (candidates.backbone == backbone).all():
        raise ValueError(
            "Candidate generation backbone does not match the requested pool"
        )
    scores = download_tensors(ARTIFACTS_REPOSITORY, f"{selector}_scores.npz", prefix)
    if set(scores) != set(prompts.prompt_id):
        raise ValueError("Score all 250 prompts in the notebook before local tuning")
    for values in scores.values():
        selection_inputs(values, configuration.model_type)
    fingerprint = input_fingerprint(prompts, candidates, scores)
    output_prefix = judge_prefix(backbone, model)
    directory = work_dir / output_prefix
    directory.mkdir(parents=True, exist_ok=True)
    parameter_path = directory / f"{selector}_parameters.json"
    if hub_file_exists(ARTIFACTS_REPOSITORY, parameter_path.name, output_prefix):
        parameters = json.loads(
            download_hub_file(
                ARTIFACTS_REPOSITORY, parameter_path.name, output_prefix
            ).read_text()
        )
    elif parameter_path.exists():
        parameters = json.loads(parameter_path.read_text())
    else:
        parameters = None
    if parameters is not None and (
        parameters["input_fingerprint"] != fingerprint
        or parameters["judge_model"] != model
        or parameters["judge_version"] != JUDGE_VERSION
        or parameters["selector"] != selector
    ):
        raise ValueError(
            "Frozen parameters do not match these inputs; use a new experiment prefix"
        )
    if parameters is not None:
        # A validated local freeze survives an interrupted Hub upload.
        parameter_path.write_text(json.dumps(parameters, indent=2) + "\n")
        if not hub_file_exists(
            ARTIFACTS_REPOSITORY, parameter_path.name, output_prefix
        ):
            upload_hub_file(ARTIFACTS_REPOSITORY, parameter_path, output_prefix)
    if stage == "tune":
        if parameters is not None:
            print("Parameters are already frozen; no evaluation data or retuning used.")
            return
        tuning = pd.DataFrame(prompts[prompts.split == "tuning"])
        comparisons = atom_comparisons(tuning, pools)
        print(
            f"Tuning: {len(tuning)} prompts, {len(comparisons.drop_duplicates())} unique comparisons; two Claude calls per missing comparison.",
            flush=True,
        )
        judgments = sync_judgments(comparisons, model, output_prefix, directory)
        outcomes = outcome_arrays(tuning, pools, judgments, model)
        inputs = [
            selection_inputs(scores[key], configuration.model_type)
            for key in tuning.prompt_id
        ]
        overall = [outcomes[key][3] for key in tuning.prompt_id]
        parameters = fit_target([game for game, _ in inputs], overall)
        if configuration.model_type == "bradley_terry":
            parameters["scalar_weights"] = fit_scalar_weights(
                [r for _, r in inputs if r is not None], overall
            )
        parameters.update(
            {
                "input_fingerprint": fingerprint,
                "judge_model": model,
                "judge_version": JUDGE_VERSION,
                "selector": selector,
                "criteria": HELPSTEER_CRITERIA,
                "tuning_prompt_ids": tuning.prompt_id.tolist(),
                "tuning_judgments_sha256": hashlib.sha256(
                    json.dumps(
                        judgments.sort_values("comparison_id").to_dict("records"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
        )
        temporary = parameter_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(parameters, indent=2) + "\n")
        temporary.replace(parameter_path)
        upload_hub_file(ARTIFACTS_REPOSITORY, parameter_path, output_prefix)
        print(f"Frozen parameters: {parameter_path}")
        return
    if parameters is None:
        raise ValueError("Run make helpsteer-tune before evaluation")
    evaluation = pd.DataFrame(prompts[prompts.split == "evaluation"])
    if set(evaluation.prompt_id) & set(parameters["tuning_prompt_ids"]):
        raise ValueError("Tuning and evaluation prompts overlap")
    policies = {
        key: selection_policies(scores[key], configuration.model_type, parameters)
        for key in evaluation.prompt_id
    }
    support_rows = [
        {
            "prompt_id": key,
            "method": method,
            "sample_index": int(index),
            "weight": float(policy[index]),
        }
        for key, methods in policies.items()
        for method, policy in methods.items()
        for index in np.flatnonzero(policy)
    ]
    supports = pd.DataFrame(support_rows)
    supports.to_parquet(directory / f"{selector}_selections.parquet", index=False)
    upload_dataframe(
        ARTIFACTS_REPOSITORY, f"{selector}_selections.parquet", supports, output_prefix
    )
    union = {
        key: {int(i) for policy in methods.values() for i in np.flatnonzero(policy)}
        for key, methods in policies.items()
    }
    comparisons = atom_comparisons(evaluation, pools, union)
    print(
        f"Evaluation: {len(evaluation)} prompts, {len(comparisons.drop_duplicates())} unique support comparisons.",
        flush=True,
    )
    judgments = sync_judgments(comparisons, model, output_prefix, directory)
    results = evaluate_policies(
        policies, outcome_arrays(evaluation, pools, judgments, model)
    )
    summary = result_summary(results)
    for filename, frame in [
        (f"{selector}_results.parquet", results),
        (f"{selector}_summary.parquet", summary),
    ]:
        frame.to_parquet(directory / filename, index=False)
        upload_dataframe(ARTIFACTS_REPOSITORY, filename, frame, output_prefix)
    print(
        summary.pivot(index="metric", columns="method", values="mean")
        .round(4)
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tune", "evaluate"])
    parser.add_argument(
        "--backbone", choices=GENERATION_BACKBONES, default="RLHFlow/LLaMA3-SFT-v2"
    )
    parser.add_argument(
        "--selector",
        choices=[c.name for c in REWARD_MODEL_CONFIGURATIONS],
        default="qwen3_4b_pairwise",
    )
    parser.add_argument("--judge-model", default=os.environ.get("CLAUDE_MODEL", ""))
    parser.add_argument("--work-dir", type=Path, default=Path(".helpsteer"))
    args = parser.parse_args()
    if not args.judge_model.strip():
        parser.error("Set CLAUDE_MODEL to an explicit Claude model identifier")
    run(args.stage, args.backbone, args.selector, args.judge_model, args.work_dir)


if __name__ == "__main__":
    main()
