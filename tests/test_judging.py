import json

import pandas as pd
import pytest

from blackwell_ita import judging
from blackwell_ita.judging import (
    JUDGE_CRITERIA,
    compare_with_claude,
    comparison_id,
    judge_comparisons,
    parse_verdict,
)


def verdict(value):
    return dict.fromkeys(JUDGE_CRITERIA, value)


def test_two_order_scores_preserve_disagreement_and_explicit_ties(monkeypatch):
    replies = iter([verdict("FIRST"), verdict("SECOND")])
    monkeypatch.setattr(judging, "claude_verdict", lambda *args: next(replies))
    row = compare_with_claude("p", "r", "a", "model")
    assert all(row[c] == 1 for c in JUDGE_CRITERIA)
    replies = iter([verdict("FIRST"), verdict("FIRST")])
    row = compare_with_claude("p", "r", "a", "model")
    assert all(row[c] == 0.5 for c in JUDGE_CRITERIA)


@pytest.mark.parametrize(
    "text", ["not json", "{}", json.dumps(verdict("INVALID")), json.dumps(verdict([]))]
)
def test_bad_verdicts_are_errors_not_ties(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_cache_identity_includes_model_anchor_and_rubric(monkeypatch):
    original = comparison_id("p", "r", "a", "m")
    assert original != comparison_id("p", "r", "other anchor", "m")
    assert original != comparison_id("p", "r", "a", "other model")
    monkeypatch.setattr(judging, "JUDGE_VERSION", "new-version")
    assert original != comparison_id("p", "r", "a", "m")


def test_judging_resumes_after_failure_and_deduplicates(monkeypatch, tmp_path):
    comparisons = pd.DataFrame(
        {"prompt": ["p", "p", "p"], "response": ["r1", "r2", "r1"], "anchor": "a"}
    )
    cache_path = tmp_path / "judgments.parquet"
    calls = []

    def fail_second(prompt, first, second, model):
        calls.append((first, second))
        if "r2" in (first, second):
            raise RuntimeError("interrupted")
        return verdict("FIRST")

    monkeypatch.setattr(judging, "claude_verdict", fail_second)
    with pytest.raises(RuntimeError):
        judge_comparisons(comparisons, "m", cache_path)
    assert len(pd.read_parquet(cache_path)) == 1
    calls.clear()

    def succeed(prompt, first, second, model):
        calls.append((first, second))
        return verdict("TIE")

    monkeypatch.setattr(judging, "claude_verdict", succeed)
    result = judge_comparisons(comparisons, "m", cache_path)
    assert len(result) == 2 and len(calls) == 2
    assert all("r2" in pair for pair in calls)
    calls.clear()
    judge_comparisons(comparisons.iloc[:1], "m", cache_path)
    assert calls == []
