"""Candidate response generation with a local causal language model."""

import marimo as mo
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blackwell_ita.train_rms import default_device


def generate_candidates(
    model_name: str,
    prompts: list[str],
    samples_per_prompt: int,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    batch_size: int = 8,
    seed: int = 1810,
    device: str | None = None,
) -> pd.DataFrame:
    """Sample responses per prompt; returns a (prompt, response) dataframe."""
    if device is None:
        device = default_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # transformers 5 annotates generate()'s self with a ty-only Protocol that
    # pyright rejects, hence the targeted ignores here and on generate()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)  # pyright: ignore[reportArgumentType]
    torch.manual_seed(seed)
    rows = []
    for prompt in mo.status.progress_bar(prompts, title="prompts"):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        for start in range(0, samples_per_prompt, batch_size):
            outputs = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=min(batch_size, samples_per_prompt - start),
                pad_token_id=tokenizer.eos_token_id,
            )
            rows.extend(
                {
                    "prompt": prompt,
                    "response": tokenizer.decode(
                        output[inputs.shape[1] :], skip_special_tokens=True
                    ),
                }
                for output in outputs
            )
    return pd.DataFrame(rows)
