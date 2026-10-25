import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

GENERATION_MODEL = "claude-haiku-5-5"


def statement_request(prompt: str) -> MessageCreateParamsNonStreaming:
    """One generation request, with thinking off and default sampling."""
    return MessageCreateParamsNonStreaming(
        model=GENERATION_MODEL,
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )


def response_text(message: anthropic.types.Message) -> str:
    """The text blocks of a response, joined and stripped."""
    return "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()


def generate_responses(
    prompts: list[str], poll_seconds: int = 30, batch_size: int = 10000
) -> list[str]:
    """One response per prompt through the Message Batches API, in prompt order.

    A batch holds at most 256 MB of requests, so prompts go out in batches of
    ``batch_size``, all submitted before any is polled.
    """
    client = anthropic.Anthropic()
    batch_ids = [
        client.messages.batches.create(
            requests=[
                Request(custom_id=str(index), params=statement_request(prompts[index]))
                for index in range(start, min(start + batch_size, len(prompts)))
            ]
        ).id
        for start in range(0, len(prompts), batch_size)
    ]
    responses: dict[int, str] = {}
    for batch_id in batch_ids:
        batch = client.messages.batches.retrieve(batch_id)
        while batch.processing_status != "ended":
            time.sleep(poll_seconds)
            batch = client.messages.batches.retrieve(batch_id)
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                raise RuntimeError(f"Request {result.custom_id} {result.result.type}")
            responses[int(result.custom_id)] = response_text(result.result.message)
    return [responses[index] for index in range(len(prompts))]
