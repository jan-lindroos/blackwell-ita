import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from blackwell_ita import models
from blackwell_ita.models import (
    BradleyTerryRewardModel,
    PairwisePreferenceModel,
    last_token_indices,
    load_reward_model,
    masked_binary_cross_entropy,
    pairwise_text,
    save_reward_model,
    trainable_state_dict,
    truncated_pairwise_text,
)


class CharacterTokenizer:
    """One token per character, right padded with id 0, no BOS token."""

    bos_token = None
    padding_side = "right"

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 97 + 1 for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 1 + 97) for token_id in token_ids)

    def __call__(self, texts, truncation, max_length, padding, return_tensors):
        encoded_texts = [self.encode(text)[:max_length] for text in texts]
        longest_length = max(len(token_ids) for token_ids in encoded_texts)
        return {
            "input_ids": torch.tensor(
                [
                    token_ids + [0] * (longest_length - len(token_ids))
                    for token_ids in encoded_texts
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1] * len(token_ids) + [0] * (longest_length - len(token_ids))
                    for token_ids in encoded_texts
                ]
            ),
        }


class NestedLanguageModel(torch.nn.Module):
    """Stands in for a multimodal checkpoint that nests its text decoder."""

    def __init__(self, language_model: torch.nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.vision_embedder = torch.nn.Linear(4, 4)


def tiny_qwen3(encoder_name: str, dtype: torch.dtype) -> torch.nn.Module:
    torch.manual_seed(0)
    decoder = Qwen3Model(
        Qwen3Config(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).to(dtype)  # pyright: ignore[reportArgumentType]
    return NestedLanguageModel(decoder) if encoder_name == "nested" else decoder


@pytest.fixture(autouse=True)
def tiny_backbone(monkeypatch):
    monkeypatch.setattr(
        models.AutoTokenizer,
        "from_pretrained",
        lambda encoder_name: CharacterTokenizer(),
    )
    monkeypatch.setattr(
        models.AutoModel,
        "from_pretrained",
        lambda encoder_name, dtype: tiny_qwen3(encoder_name, dtype),
    )


def tiny_batch() -> dict:
    return {
        "prompt": ["hello", "hi"],
        "first_response": ["short", "a much longer response"],
        "second_response": ["other", "b"],
        "target": torch.tensor([[1.0, 0.5], [0.0, 0.25]]),
        "mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
    }


def test_last_token_indices_handles_both_padding_sides():
    attention_mask = torch.tensor([[1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    assert last_token_indices(attention_mask).tolist() == [2, 3, 3]


def test_masked_binary_cross_entropy_ignores_masked_entries():
    logits = torch.tensor([[0.0, 100.0]])
    loss = masked_binary_cross_entropy(
        logits, torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])
    )
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_truncated_pairwise_text_leaves_short_pairs_alone():
    assert truncated_pairwise_text(
        "p",
        "aa",
        "bb",
        CharacterTokenizer(),  # pyright: ignore[reportArgumentType]
        100,
    ) == pairwise_text("p", "aa", "bb")


def test_truncated_pairwise_text_splits_budget_equally():
    marker_length = len(pairwise_text("p", "", ""))
    truncated_text = truncated_pairwise_text(
        "p",
        "a" * 50,
        "b" * 50,
        CharacterTokenizer(),  # pyright: ignore[reportArgumentType]
        marker_length + 20,
    )
    assert truncated_text == pairwise_text("p", "a" * 10, "b" * 10)


def test_truncated_pairwise_text_donates_surplus_to_longer_response():
    marker_length = len(pairwise_text("p", "", ""))
    truncated_text = truncated_pairwise_text(
        "p",
        "a" * 50,
        "b" * 4,
        CharacterTokenizer(),  # pyright: ignore[reportArgumentType]
        marker_length + 20,
    )
    assert truncated_text == pairwise_text("p", "a" * 16, "b" * 4)


@pytest.mark.parametrize(
    "model_class", [PairwisePreferenceModel, BradleyTerryRewardModel]
)
def test_reward_models_return_one_logit_per_criterion(model_class):
    model = model_class("tiny", ["x", "y"], 64, None)
    logits = model.batch_logits(tiny_batch(), "cpu")  # pyright: ignore[reportArgumentType]
    assert logits.shape == (2, 2)
    model.compute_loss(tiny_batch(), "cpu").backward()  # pyright: ignore[reportArgumentType]


def test_bradley_terry_logits_are_antisymmetric_in_presentation_order():
    model = BradleyTerryRewardModel("tiny", ["x", "y"], 64, None).eval()
    batch = tiny_batch()
    swapped_batch = batch | {
        "first_response": batch["second_response"],
        "second_response": batch["first_response"],
    }
    with torch.no_grad():
        logits = model.batch_logits(batch, "cpu")  # pyright: ignore[reportArgumentType]
        swapped_logits = model.batch_logits(swapped_batch, "cpu")  # pyright: ignore[reportArgumentType]
    assert torch.allclose(logits, -swapped_logits, atol=1e-6)


def test_lora_freezes_backbone_and_unwraps_nested_decoder():
    model = PairwisePreferenceModel("nested", ["x", "y"], 64, lora_rank=4)
    trainable_names = set(trainable_state_dict(model))
    assert trainable_names
    assert all("lora_" in name or name.startswith("head.") for name in trainable_names)
    assert not any("vision_embedder" in name for name, _ in model.named_parameters())
    model.compute_loss(tiny_batch(), "cpu").backward()  # pyright: ignore[reportArgumentType]
    assert model.scorer.head.weight.grad is not None


@pytest.mark.parametrize("lora_rank", [None, 4])
def test_save_and_load_round_trip(tmp_path, lora_rank):
    model = PairwisePreferenceModel("tiny", ["x", "y"], 64, lora_rank)
    with torch.no_grad():
        for parameter in trainable_state_dict(model).values():
            parameter.add_(0.5)
    checkpoint_path = tmp_path / "model.pt"
    save_reward_model(model, checkpoint_path)
    loaded_model = load_reward_model(checkpoint_path, "cpu")
    assert isinstance(loaded_model, PairwisePreferenceModel)
    assert loaded_model.lora_rank == lora_rank
    for name, tensor in trainable_state_dict(loaded_model).items():
        assert torch.equal(
            tensor,
            trainable_state_dict(model)[name].to(torch.bfloat16).to(tensor.dtype),
        )
