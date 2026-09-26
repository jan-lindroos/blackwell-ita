import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def batch_sizes(total: int, batch_size: int) -> list[int]:
    """Split a sample count into generate() batch sizes."""
    return [min(batch_size, total - start) for start in range(0, total, batch_size)]


def generate_responses(
    model_name: str,
    prompts: list[str],
    samples_per_prompt: int,
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    batch_size: int = 32,
    seed: int = 1810,
) -> pd.DataFrame:
    """Sample responses per prompt; one (prompt_index, sample_index, response) row each."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)  # pyright: ignore[reportArgumentType]
    torch.manual_seed(seed)
    response_rows = []
    for prompt_index, prompt in enumerate(tqdm(prompts, desc=model_name)):
        chat_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        prompt_length = chat_inputs["input_ids"].shape[1]
        sample_index = 0
        for sample_count in batch_sizes(samples_per_prompt, batch_size):
            generated_outputs = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                **chat_inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=sample_count,
                pad_token_id=tokenizer.eos_token_id,
            )
            for generated_output in generated_outputs:
                response_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "response": tokenizer.decode(
                            generated_output[prompt_length:], skip_special_tokens=True
                        ).strip(),
                    }
                )
                sample_index += 1
    model.to("cpu")  # pyright: ignore[reportArgumentType]
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pd.DataFrame(response_rows)
