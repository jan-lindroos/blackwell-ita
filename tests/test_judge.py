"""Tests for the Claude CLI judge."""

import json
from types import SimpleNamespace

import pytest

import judge


def order_sensitive_pick(preferred: str):
    """Stand-in for claude_pick preferring whichever slot holds ``preferred``."""

    def pick(prompt: str, model: str) -> str:
        first_slot = prompt.index("First response:")
        second_slot = prompt.index("Second response:")
        position = prompt.index(preferred)
        return "FIRST" if first_slot < position < second_slot else "SECOND"

    return pick


def test_outcome_swaps_orders_and_maps_scores(monkeypatch):
    """A consistent winner scores 1 or 0 through the order-swapped prompts."""
    # The stub inspects which slot holds the preferred text, so a missing or
    # mislabelled order swap in outcome cannot pass
    monkeypatch.setattr(judge, "claude_pick", order_sensitive_pick("the response"))
    assert judge.outcome("instruction", "the response", "the anchor")["score"] == 1.0
    monkeypatch.setattr(judge, "claude_pick", order_sensitive_pick("the anchor"))
    assert judge.outcome("instruction", "the response", "the anchor")["score"] == 0.0


def fixed_picks(replies):
    """Stand-in for claude_pick returning scripted replies in order."""
    iterator = iter(replies)
    return lambda prompt, model: next(iterator)


def test_outcome_draws_on_disagreement_or_parse_failure(monkeypatch):
    """Order-swap disagreement and unparseable verdicts both draw at 0.5."""
    monkeypatch.setattr(judge, "claude_pick", fixed_picks(["FIRST", "FIRST"]))
    assert judge.outcome("instruction", "response", "anchor")["score"] == 0.5
    monkeypatch.setattr(judge, "claude_pick", fixed_picks([None, "SECOND"]))
    judgement = judge.outcome("instruction", "response", "anchor")
    assert judgement["score"] == 0.5
    assert judgement["forward"] is None
    assert judgement["backward"] == "SECOND"


def fixed_run(result, returncode=0):
    """Stand-in for subprocess.run returning a fixed CLI reply."""
    return lambda command, **kwargs: SimpleNamespace(
        stdout=json.dumps({"result": result}), stderr="", returncode=returncode
    )


def test_claude_pick_invokes_cli_and_parses_reply(monkeypatch):
    """The CLI is called with the prompt and model, and the reply is parsed."""
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return fixed_run(" first choice")(command, **kwargs)

    monkeypatch.setattr(judge.subprocess, "run", fake_run)
    assert judge.claude_pick("the prompt", "claude-sonnet-5") == "FIRST"
    assert captured["command"][0] == "claude"
    assert captured["command"][-1] == "the prompt"
    assert "claude-sonnet-5" in captured["command"]


def test_claude_pick_handles_second_and_unparseable_replies(monkeypatch):
    """SECOND replies parse; unparseable replies return None."""
    monkeypatch.setattr(judge.subprocess, "run", fixed_run("Second."))
    assert judge.claude_pick("prompt", "claude-sonnet-5") == "SECOND"
    monkeypatch.setattr(judge.subprocess, "run", fixed_run("I cannot decide"))
    assert judge.claude_pick("prompt", "claude-sonnet-5") is None


def test_claude_pick_retries_transient_failures(monkeypatch):
    """A failed CLI call is retried and the retry's reply is used."""
    calls = {"count": 0}

    def flaky_run(command, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(stdout="", stderr="boom", returncode=1)
        return fixed_run("FIRST")(command, **kwargs)

    monkeypatch.setattr(judge.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(judge.subprocess, "run", flaky_run)
    assert judge.claude_pick("prompt", "claude-sonnet-5") == "FIRST"
    assert calls["count"] == 2


def test_claude_pick_raises_after_exhausting_attempts(monkeypatch):
    """Persistent CLI failures raise once the attempts run out."""
    calls = {"count": 0}

    def failing_run(command, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(judge.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(judge.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match=r"3 attempts.*boom"):
        judge.claude_pick("prompt", "claude-sonnet-5")
    assert calls["count"] == 3


def test_outcomes_preserves_order_and_forwards_arguments(monkeypatch):
    """Concurrent judging keeps input order and forwards keyword arguments."""

    def fake_outcome(instruction, response, anchor, model=None):
        assert model == "haiku"
        return {"score": float(len(response))}

    monkeypatch.setattr(judge, "outcome", fake_outcome)
    comparisons = [
        {"instruction": "i", "response": "r" * n, "anchor": "a"} for n in (3, 1, 2)
    ]
    scores = [j["score"] for j in judge.outcomes(comparisons, model="haiku", workers=2)]
    assert scores == [3.0, 1.0, 2.0]
