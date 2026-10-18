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


def generate_responses(prompts: list[str], poll_seconds: int = 30) -> list[str]:
    """One response per prompt through the Message Batches API, in prompt order."""
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(
        requests=[
            Request(custom_id=str(index), params=statement_request(prompt))
            for index, prompt in enumerate(prompts)
        ]
    )
    while batch.processing_status != "ended":
        time.sleep(poll_seconds)
        batch = client.messages.batches.retrieve(batch.id)
    responses: dict[int, str] = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            raise RuntimeError(f"Request {result.custom_id} {result.result.type}")
        responses[int(result.custom_id)] = response_text(result.result.message)
    return [responses[index] for index in range(len(prompts))]
