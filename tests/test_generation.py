from types import SimpleNamespace

import pytest

from blackwell_ita import generation


class FakeBatches:
    """Each batch finishes on its second poll and returns results in reverse order."""

    def __init__(self) -> None:
        self.polls: dict[str, int] = {}
        self.requests: dict[str, list] = {}

    def create(self, requests):
        batch_id = f"batch{len(self.requests)}"
        self.requests[batch_id] = requests
        self.polls[batch_id] = 0
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls[batch_id] += 1
        return SimpleNamespace(
            id=batch_id,
            processing_status="ended" if self.polls[batch_id] == 2 else "in_progress",
        )

    def results(self, batch_id):
        for request in reversed(self.requests[batch_id]):
            prompt = request["params"]["messages"][0]["content"]
            yield SimpleNamespace(
                custom_id=request["custom_id"],
                result=SimpleNamespace(
                    type="succeeded",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=f" {prompt}! ")]
                    ),
                ),
            )


def fake_client(batches: FakeBatches):
    return lambda: SimpleNamespace(messages=SimpleNamespace(batches=batches))


def test_generate_responses_returns_one_response_per_prompt_in_order(monkeypatch):
    batches = FakeBatches()
    monkeypatch.setattr(generation.anthropic, "Anthropic", fake_client(batches))
    assert generation.generate_responses(["a", "b", "c"], poll_seconds=0) == [
        "a!",
        "b!",
        "c!",
    ]
    params = batches.requests["batch0"][0]["params"]
    assert params["model"] == generation.GENERATION_MODEL
    assert params["thinking"] == {"type": "disabled"}


def test_generate_responses_fails_on_an_unsuccessful_request(monkeypatch):
    batches = FakeBatches()
    original_results = batches.results

    def results_with_an_error(batch_id):
        results = list(original_results(batch_id))
        results[0].result.type = "errored"
        return results

    batches.results = results_with_an_error  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setattr(generation.anthropic, "Anthropic", fake_client(batches))
    with pytest.raises(RuntimeError, match="errored"):
        generation.generate_responses(["a", "b"], poll_seconds=0)


def test_generate_responses_splits_prompts_across_batches(monkeypatch):
    batches = FakeBatches()
    monkeypatch.setattr(generation.anthropic, "Anthropic", fake_client(batches))
    responses = generation.generate_responses(
        ["a", "b", "c", "d", "e"], poll_seconds=0, batch_size=2
    )
    assert responses == ["a!", "b!", "c!", "d!", "e!"]
    assert [len(requests) for requests in batches.requests.values()] == [2, 2, 1]
