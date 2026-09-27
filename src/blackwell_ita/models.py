from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from blackwell_ita.data import Batch

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of each row's last attended token, whichever side is padded."""
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
    return (attention_mask * positions).argmax(dim=1)


def masked_binary_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean binary cross-entropy over the unmasked criterion entries."""
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return (losses * mask).sum() / mask.sum()


def pairwise_text(prompt: str, first_response: str, second_response: str) -> str:
    """Format a prompt with both responses for joint pairwise scoring."""
    return (
        f"{prompt}\n\n[RESPONSE 1]\n{first_response}\n\n[RESPONSE 2]\n{second_response}"
    )


def truncated_pairwise_text(
    prompt: str,
    first_response: str,
    second_response: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> str:
    """Pairwise text whose responses are truncated to fit within ``max_tokens``.

    Right truncation would consume [RESPONSE 2]'s tail first, so the budget
    left after the prompt and markers is split equally between the responses,
    a response shorter than its half donating the surplus to the other.
    """
    first_token_ids = tokenizer.encode(first_response)
    second_token_ids = tokenizer.encode(second_response)
    response_budget = max_tokens - len(tokenizer.encode(pairwise_text(prompt, "", "")))
    if len(first_token_ids) + len(second_token_ids) <= response_budget:
        return pairwise_text(prompt, first_response, second_response)
    first_kept_count = max(
        0,
        min(
            len(first_token_ids),
            max(response_budget // 2, response_budget - len(second_token_ids)),
        ),
    )
    second_kept_count = max(0, response_budget - first_kept_count)
    return pairwise_text(
        prompt,
        tokenizer.decode(first_token_ids[:first_kept_count]),  # pyright: ignore[reportArgumentType]
        tokenizer.decode(second_token_ids[:second_kept_count]),  # pyright: ignore[reportArgumentType]
    )


def pointwise_text(prompt: str, response: str) -> str:
    """Format a prompt with a single response for pointwise reward scoring."""
    return f"{prompt}\n\n[RESPONSE]\n{response}"


class MultiHeadEncoder(torch.nn.Module):
    """Pretrained decoder with a linear head giving one logit per criterion.

    Without ``lora_rank`` every parameter trains and stays fp32 (AdamW at lr
    around 1e-5 underflows pure bf16 weights). With it the backbone is frozen
    in bf16 and only the fp32 LoRA adapters and the head train.
    """

    def __init__(
        self,
        encoder_name: str,
        head_count: int,
        lora_rank: int | None,
        gradient_checkpointing: bool,
    ) -> None:
        """Load the decoder, optionally wrap it in LoRA, attach a fresh head."""
        super().__init__()
        pretrained_model = AutoModel.from_pretrained(
            encoder_name,
            dtype=torch.float32 if lora_rank is None else torch.bfloat16,
        )
        # Multimodal checkpoints (Gemma 4) nest the text decoder; the vision
        # and audio embedders are never used
        self.encoder = getattr(pretrained_model, "language_model", pretrained_model)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if lora_rank is not None:
            self.encoder = get_peft_model(
                self.encoder,  # pyright: ignore[reportArgumentType]
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=2 * lora_rank,
                    lora_dropout=0.05,
                    target_modules=LORA_TARGET_MODULES,
                ),
            )
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, head_count)

    def forward(self, tokenized: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score tokenized inputs from the last non-padding token's hidden state."""
        hidden_states = self.encoder(**tokenized, use_cache=False).last_hidden_state
        last_indices = last_token_indices(tokenized["attention_mask"])
        pooled_states = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            last_indices,
        ]
        return self.head(pooled_states.to(self.head.weight.dtype)).float()


class RewardModel(torch.nn.Module):
    """Multi-head encoder with its tokenizer and truncation limit."""

    model_type: str

    def __init__(
        self,
        encoder_name: str,
        criteria: list[str],
        max_tokens: int,
        lora_rank: int | None,
        gradient_checkpointing: bool = True,
    ) -> None:
        """Build the encoder and load the tokenizer."""
        super().__init__()
        self.encoder_name = encoder_name
        self.criteria = criteria
        self.max_tokens = max_tokens
        self.lora_rank = lora_rank
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.tokenizer.padding_side = "right"
        # Gemma defines <bos> but its tokenizer does not add it
        bos_token = self.tokenizer.bos_token
        adds_bos_token = (
            bos_token is not None
            and self.tokenizer("x")["input_ids"][0] == self.tokenizer.bos_token_id
        )
        self.text_prefix = (
            bos_token if bos_token is not None and not adds_bos_token else ""
        )
        self.scorer = MultiHeadEncoder(
            encoder_name, len(criteria), lora_rank, gradient_checkpointing
        )

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        """Tokenize texts and return per-criterion logits."""
        tokenized = self.tokenizer(
            [self.text_prefix + text for text in texts],
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in tokenized.items()}
        if device.startswith("cuda"):
            with torch.autocast("cuda", torch.bfloat16):
                return self.scorer(inputs)
        return self.scorer(inputs)

    def batch_logits(self, batch: Batch, device: str) -> torch.Tensor:
        """Per-criterion logits that the first response beats the second."""
        raise NotImplementedError

    def compute_loss(self, batch: Batch, device: str) -> torch.Tensor:
        """Masked binary cross-entropy between the batch logits and targets."""
        return masked_binary_cross_entropy(
            self.batch_logits(batch, device),
            batch["target"].to(device),
            batch["mask"].to(device),
        )


class PairwisePreferenceModel(RewardModel):
    """Preference model scoring both responses jointly in one input."""

    model_type = "pairwise"

    def batch_logits(self, batch: Batch, device: str) -> torch.Tensor:
        """Joint logits over both responses, truncated symmetrically if overlong."""
        joint_texts = [
            truncated_pairwise_text(
                prompt, first_response, second_response, self.tokenizer, self.max_tokens
            )
            for prompt, first_response, second_response in zip(
                batch["prompt"],
                batch["first_response"],
                batch["second_response"],
                strict=True,
            )
        ]
        return self.score(joint_texts, device)


class BradleyTerryRewardModel(RewardModel):
    """Pointwise reward model whose pair logit is a reward difference."""

    model_type = "bradley_terry"

    def batch_logits(self, batch: Batch, device: str) -> torch.Tensor:
        """Pointwise reward difference per criterion, one forward per side."""
        first_rewards = self.score(
            [
                pointwise_text(prompt, response)
                for prompt, response in zip(
                    batch["prompt"], batch["first_response"], strict=True
                )
            ],
            device,
        )
        second_rewards = self.score(
            [
                pointwise_text(prompt, response)
                for prompt, response in zip(
                    batch["prompt"], batch["second_response"], strict=True
                )
            ],
            device,
        )
        return first_rewards - second_rewards


REWARD_MODEL_CLASSES: dict[str, type[RewardModel]] = {
    model_class.model_type: model_class
    for model_class in (PairwisePreferenceModel, BradleyTerryRewardModel)
}


def trainable_state_dict(model: RewardModel) -> dict[str, torch.Tensor]:
    """Parameters that training changes: everything, or LoRA adapters and head."""
    return {
        name: parameter.detach()
        for name, parameter in model.scorer.named_parameters()
        if parameter.requires_grad
    }


def save_reward_model(model: RewardModel, path: Path) -> None:
    """Save the trained parameters in bf16 with everything needed to rebuild."""
    torch.save(
        {
            "model_type": model.model_type,
            "encoder_name": model.encoder_name,
            "criteria": model.criteria,
            "max_tokens": model.max_tokens,
            "lora_rank": model.lora_rank,
            "state_dict": {
                name: tensor.to("cpu", torch.bfloat16)
                for name, tensor in trainable_state_dict(model).items()
            },
        },
        path,
    )


def load_reward_model(path: Path, device: str) -> RewardModel:
    """Rebuild a saved reward model in eval mode on ``device``."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = REWARD_MODEL_CLASSES[checkpoint["model_type"]](
        checkpoint["encoder_name"],
        checkpoint["criteria"],
        checkpoint["max_tokens"],
        checkpoint["lora_rank"],
    )
    assert set(trainable_state_dict(model)) == set(checkpoint["state_dict"])
    model.scorer.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model
