import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from blackwell_ita.hub import download_hub_file, hub_file_exists, upload_hub_file
from blackwell_ita.models import RewardModel, pointwise_text, truncated_pairwise_text


def pairwise_preference_tensor(
    model: RewardModel,
    prompt: str,
    responses: list[str],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Skew-symmetrised win probabilities, shape (head, response, response).

    Both presentation orders of every pair are scored, then
    P <- (P + 1 - P^T) / 2 with 1/2 on the diagonal. A Bradley-Terry model
    scores each response once instead.
    """
    if model.model_type == "bradley_terry":
        return bradley_terry_preference_tensor(
            model, prompt, responses, device, batch_size
        )
    response_count = len(responses)
    index_pairs = [
        (first_index, second_index)
        for first_index in range(response_count)
        for second_index in range(response_count)
        if first_index != second_index
    ]
    pair_texts = [
        truncated_pairwise_text(
            prompt,
            responses[first_index],
            responses[second_index],
            model.tokenizer,
            model.max_tokens,
        )
        for first_index, second_index in index_pairs
    ]
    # Length-sorted batches pad to their own longest member
    length_order = sorted(
        range(len(pair_texts)), key=lambda index: len(pair_texts[index])
    )
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(length_order), batch_size):
            logits = model.score(
                [
                    pair_texts[index]
                    for index in length_order[start : start + batch_size]
                ],
                device,
            )
            probability_batches.append(torch.sigmoid(logits).cpu())
    probabilities = (
        torch.cat(probability_batches).float().numpy()[np.argsort(length_order)]
    )
    preference_tensor = np.full(
        (probabilities.shape[1], response_count, response_count), 0.5
    )
    for (first_index, second_index), pair_probabilities in zip(
        index_pairs, probabilities, strict=True
    ):
        preference_tensor[:, first_index, second_index] = pair_probabilities
    return (preference_tensor + 1.0 - preference_tensor.transpose(0, 2, 1)) / 2.0


def bradley_terry_preference_tensor(
    model: RewardModel,
    prompt: str,
    responses: list[str],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Win probabilities sigmoid(r_i - r_j), shape (head, response, response)."""
    rewards = bradley_terry_rewards(model, prompt, responses, device, batch_size)
    return rewards_to_preferences(rewards)


def rewards_to_preferences(rewards: np.ndarray) -> np.ndarray:
    """Convert raw (head, response) BT rewards into complementary probabilities."""
    values = torch.as_tensor(rewards)
    return torch.sigmoid(values[:, :, None] - values[:, None, :]).numpy()


def bradley_terry_rewards(
    model: RewardModel,
    prompt: str,
    responses: list[str],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Raw scalar rewards, shape (head, response), retained for best-of-N."""
    response_texts = [pointwise_text(prompt, response) for response in responses]
    with torch.no_grad():
        rewards = torch.cat(
            [
                model.score(response_texts[start : start + batch_size], device).cpu()
                for start in range(0, len(response_texts), batch_size)
            ]
        ).T
    return rewards.float().numpy()


def upload_tensors(
    repository_id: str, filename: str, tensors: dict[str, np.ndarray], prefix: str
) -> None:
    """Upload keyed tensors as one npz under ``prefix``."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / filename
        np.savez(local_path, **tensors)  # pyright: ignore[reportArgumentType]
        upload_hub_file(repository_id, local_path, prefix)


def download_tensors(
    repository_id: str, filename: str, prefix: str
) -> dict[str, np.ndarray]:
    """Keyed tensors from an npz under ``prefix``, empty when it does not exist."""
    if not hub_file_exists(repository_id, filename, prefix):
        return {}
    with np.load(download_hub_file(repository_id, filename, prefix)) as saved_tensors:
        return {key: saved_tensors[key] for key in saved_tensors.files}


def score_with_resume[Inputs](
    repository_id: str,
    filename: str,
    prefix: str,
    keyed_inputs: dict[str, Inputs],
    score_one: Callable[[Inputs], np.ndarray],
    checkpoint_every: int = 10,
) -> dict[str, np.ndarray]:
    """Score every key missing from the hub file, checkpointing as it goes."""
    tensors = download_tensors(repository_id, filename, prefix)
    missing_keys = [key for key in keyed_inputs if key not in tensors]
    for scored_count, key in enumerate(tqdm(missing_keys, desc=filename), start=1):
        tensors[key] = score_one(keyed_inputs[key])
        if scored_count % checkpoint_every == 0 or scored_count == len(missing_keys):
            upload_tensors(repository_id, filename, tensors, prefix)
    return {key: tensors[key] for key in keyed_inputs}
