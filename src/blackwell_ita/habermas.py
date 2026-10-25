from itertools import combinations, permutations

import numpy as np
import pandas as pd

from blackwell_ita.data import assign_prompt_splits
from blackwell_ita.generation import generate_responses
from blackwell_ita.hub import ARTIFACTS_REPOSITORY, ensure_hub_dataframe
from blackwell_ita.models import RewardModel
from blackwell_ita.scoring import pairwise_preference_tensor
from blackwell_ita.training import RewardModelConfiguration

COMPARISONS_URL = "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet"
# The data's stand-in for a participant who wrote nothing
PLACEHOLDER_OPINION = "No opinion was provided."
COMPARISON_COLUMNS = {
    "question.id": "question_id",
    "question.text": "question",
    "round_id": "round_id",
    "iteration_index": "iteration_index",
    "metadata.participant_id": "participant_id",
    "own_opinion.text": "opinion",
    "candidates.text": "statements",
    "rankings.numerical_ranks": "ranks",
}
HABERMAS_CRITERIA = ["preference"]
HABERMAS_PREFIX = "habermas-gold"
# The selecting proxies keep their checkpoints and runs apart from the gold's
SELECTOR_PREFIX = "habermas-selector"
# The gold model is trained on human rankings and stands in for the
# participants. The proxies select and only ever see gold labels
GOLD_CONFIGURATION = RewardModelConfiguration(
    name="gemma4_31b_pairwise_lora",
    model_type="pairwise",
    encoder_name="google/gemma-4-31B-it",
    lora_rank=16,
    learning_rate=1e-4,
    evaluation_interval_steps=1000,
    max_tokens=2048,
    batch_size=8,
)
PROXY_ENCODERS = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
}
PROXY_CONFIGURATIONS = [
    RewardModelConfiguration(
        name=f"{size_name}_bradley_terry",
        model_type="bradley_terry",
        encoder_name=encoder_name,
        lora_rank=None,
        learning_rate=5e-6,
        evaluation_interval_steps=1000,
        max_tokens=2048,
        max_epochs=1,
    )
    for size_name, encoder_name in PROXY_ENCODERS.items()
]
# The paper's questions are withheld from every split, test included
PAPER_QUESTION_COUNT = 200
# None takes every eligible question
GROUP_COUNTS = {"train": None, "validation": 100, "test": 100, "paper": 100}
# Pairs the gold scores per participant, every pair in the other splits
SCORED_PAIR_COUNTS = {"train": 60, "validation": 120}
# The even orderings of the four opinions show each participant in each
# position three times
CONSENSUS_ORDERS = [
    order
    for order in permutations(range(1, 5))
    if sum(first > second for first, second in combinations(order, 2)) % 2 == 0
]
# Each slot is a kind and the participants whose opinions it shows, in order
CANDIDATE_SLOTS = [("consensus", order) for order in CONSENSUS_ORDERS] + [
    ("participant", (participant,)) for participant in range(1, 5) for _ in range(3)
]
SLOT_INSTRUCTIONS = {
    "consensus": "Write a statement the whole group could endorse.",
    "participant": "Write the statement as if the whole group shared this opinion.",
}
EXAMPLES_PER_PROMPT = 8
WHOLE_GROUP = [(0, 1, 2, 3)]


def usable_rankings(comparisons: pd.DataFrame) -> pd.DataFrame:
    """One row per human participant and candidate set they fully ranked.

    Rank 0 is best and ties share a rank.
    """
    completed = comparisons[
        (comparisons["metadata.provenance"] == "HUMAN_CITIZEN")
        & (comparisons["rankings.metadata.status"] == "COMPLETED")
    ]
    rankings = completed[list(COMPARISON_COLUMNS)].rename(columns=COMPARISON_COLUMNS)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
    fully_ranked = [
        len(ranks) >= 2 and len(ranks) == len(statements) and min(ranks) >= 0
        for ranks, statements in zip(
            rankings["ranks"], rankings["statements"], strict=True
        )
    ]
    rankings = rankings[fully_ranked]
    return rankings.assign(
        statements=[list(statements) for statements in rankings["statements"]],
        ranks=[[int(rank) for rank in ranks] for ranks in rankings["ranks"]],
    ).reset_index(drop=True)


def load_habermas_rankings() -> pd.DataFrame:
    """Usable human rankings from the Habermas Machine candidate comparisons."""
    return usable_rankings(
        pd.read_parquet(
            COMPARISONS_URL,
            columns=[
                *COMPARISON_COLUMNS,
                "metadata.provenance",
                "rankings.metadata.status",
            ],
        )
    )


def rank_preference(first_rank: int, second_rank: int) -> float:
    """Probability target that the first statement is preferred, 0.5 on ties."""
    if first_rank < second_rank:
        return 1.0
    if first_rank > second_rank:
        return 0.0
    return 0.5


def habermas_prompt(question: str, opinion: str) -> str:
    """The context a pairwise model conditions on: question and one opinion."""
    return f"Question: {question}\n\nParticipant opinion: {opinion}"


def habermas_pairs(rankings: pd.DataFrame) -> pd.DataFrame:
    """Every unordered statement pair each participant ranked, with its target."""
    pair_rows = [
        {
            "question_id": ranking["question_id"],
            "split": ranking["split"],
            "prompt": habermas_prompt(ranking["question"], ranking["opinion"]),
            "response_a": ranking["statements"][first_index],
            "response_b": ranking["statements"][second_index],
            "preference": rank_preference(
                ranking["ranks"][first_index], ranking["ranks"][second_index]
            ),
        }
        for ranking in rankings.to_dict("records")
        for first_index in range(len(ranking["statements"]))
        for second_index in range(first_index + 1, len(ranking["statements"]))
    ]
    return pd.DataFrame(pair_rows)


def candidate_sets(rankings: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """One row per candidate set in ``split_name`` that two or more participants ranked."""
    split_rankings = rankings[rankings["split"] == split_name]
    set_rows = []
    for (round_id, iteration_index), set_rankings in split_rankings.groupby(  # pyright: ignore[reportGeneralTypeIssues]
        ["round_id", "iteration_index"], sort=True
    ):
        if len(set_rankings) < 2:
            continue
        set_rows.append(
            {
                "set_id": f"{round_id}_{iteration_index}",
                "question_id": set_rankings["question_id"].iloc[0],  # pyright: ignore[reportAttributeAccessIssue]
                "question": set_rankings["question"].iloc[0],  # pyright: ignore[reportAttributeAccessIssue]
                "statements": set_rankings["statements"].iloc[0],  # pyright: ignore[reportAttributeAccessIssue]
                "opinions": set_rankings["opinion"].tolist(),
                "participant_ranks": set_rankings["ranks"].tolist(),
            }
        )
    return pd.DataFrame(set_rows)


def deliberation_groups(
    rankings: pd.DataFrame,
    split_name: str,
    group_count: int | None,
    participant_count: int = 4,
    seed: int = 1810,
) -> pd.DataFrame:
    """Four real participants per question in ``split_name``.

    Each question contributes its first opening candidate set that at least
    ``participant_count`` participants with a written opinion ranked, and only
    those participants are chosen. A ``group_count`` of None takes every
    eligible question.
    """
    random_generator = np.random.default_rng(seed)
    opening_sets = candidate_sets(
        rankings[rankings["iteration_index"] == 0],  # pyright: ignore[reportArgumentType]
        split_name,
    )
    opening_sets["opinions"] = [
        [opinion for opinion in opinions if opinion != PLACEHOLDER_OPINION]
        for opinions in opening_sets["opinions"]
    ]
    eligible_sets = (
        opening_sets[opening_sets["opinions"].map(len) >= participant_count]  # pyright: ignore[reportCallIssue]
        .drop_duplicates("question_id")
        .reset_index(drop=True)
    )
    chosen_positions = (
        range(len(eligible_sets))
        if group_count is None
        else sorted(
            random_generator.choice(len(eligible_sets), size=group_count, replace=False)
        )
    )
    group_rows = []
    for set_row in eligible_sets.iloc[chosen_positions].to_dict("records"):
        participant_positions = sorted(
            random_generator.choice(
                len(set_row["opinions"]),
                size=participant_count,
                replace=False,
            )
        )
        group_rows.append(
            {
                "question_id": set_row["question_id"],
                "question": set_row["question"],
                "opinions": [
                    set_row["opinions"][position] for position in participant_positions
                ],
                "split": split_name,
            }
        )
    return pd.DataFrame(group_rows)


def assign_habermas_splits(rankings: pd.DataFrame, seed: int = 1810) -> pd.DataFrame:
    """Label the paper's questions "paper", then split the rest 70/15/15 by question."""
    paper_question_ids = (
        rankings["question_id"]
        .drop_duplicates()
        .sample(n=PAPER_QUESTION_COUNT, random_state=seed)
    )
    is_paper = rankings["question_id"].isin(paper_question_ids)
    return pd.concat(
        [
            assign_prompt_splits(rankings[~is_paper], "question_id"),  # pyright: ignore[reportArgumentType]
            rankings[is_paper].assign(split="paper"),
        ]
    ).sort_index()


def ensure_rankings() -> pd.DataFrame:
    """Human rankings split by question, built and uploaded unless the hub has them."""
    return ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "rankings.parquet",
        HABERMAS_PREFIX,
        lambda: assign_habermas_splits(load_habermas_rankings()),
    )


def ensure_groups(rankings: pd.DataFrame) -> pd.DataFrame:
    """Deliberation groups for every split, built and uploaded unless the hub has them."""
    return ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "groups.parquet",
        HABERMAS_PREFIX,
        lambda: pd.concat(
            [
                deliberation_groups(rankings, split_name, group_count)
                for split_name, group_count in GROUP_COUNTS.items()
            ],
            ignore_index=True,
        ),
    )


def style_examples(
    rankings: pd.DataFrame, example_count: int = 400, seed: int = 1810
) -> pd.DataFrame:
    """Habermas Machine statements from train-split candidate sets, both rounds.

    Only statements of 30 to 220 words that open with a letter are kept,
    without the "Final consensus statement:" label some revisions carry.
    """
    statements = candidate_sets(rankings, "train")[
        ["question_id", "question", "statements"]
    ].explode(  # pyright: ignore[reportCallIssue]
        "statements"  # pyright: ignore[reportArgumentType]
    )
    word_counts = statements["statements"].str.split().str.len()  # pyright: ignore[reportAttributeAccessIssue]
    eligible = statements[
        word_counts.between(30, 220)
        & statements["statements"].str[0].str.isalpha()  # pyright: ignore[reportAttributeAccessIssue]
        & ~statements["statements"].str.startswith("Final consensus")  # pyright: ignore[reportAttributeAccessIssue]
    ]
    return (
        eligible.rename(columns={"statements": "statement"})  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        .sample(n=example_count, random_state=seed)
        .reset_index(drop=True)
    )


def statement_prompt(
    question: str,
    shown_opinions: list[str],
    instruction: str,
    example_text: str,
    target_words: int,
) -> str:
    """The generation prompt for one candidate statement."""
    opinion_text = "\n\n".join(f"Opinion: {opinion}" for opinion in shown_opinions)
    return (
        "Groups of ordinary UK citizens discuss a policy question, and a statement"
        " of the group's view is written for them in plain everyday language."
        f" Here are statements written for other groups.\n\n{example_text}\n\n"
        f"Write the statement for a new group.\n\nQuestion: {question}\n\n"
        f"{opinion_text}\n\n{instruction} Match the example statements closely"
        " in wording, phrasing habits and tone: plain everyday language, written"
        " as the group's view, and reusing the opinions' own words and phrases"
        " wherever possible, as the examples do. State the position plainly and"
        " confidently rather than hedging, and vary the opening as the examples"
        " do. Aim for about"
        f" {target_words} words in one paragraph, with no headings, lists or"
        " titles and no mention of participants or opinions. Reply with the"
        " statement only."
    )


def candidate_prompts(
    groups: pd.DataFrame, examples: pd.DataFrame, seed: int = 1810
) -> pd.DataFrame:
    """The generation prompts for every group's pool.

    Each pool covers the 24 slots once, twelve consensus slots showing every
    opinion in each even ordering and three single-participant slots per
    participant showing only that opinion. Every prompt shows statements from
    other questions as style examples.
    """
    random_generator = np.random.default_rng(seed)
    rows = []
    for group in groups.to_dict("records"):
        opinions = list(group["opinions"])
        if len(opinions) != 4:
            raise ValueError("The candidate pool requires exactly four participants")
        other_examples = examples[examples["question_id"] != group["question_id"]]
        example_lengths = examples["statement"].str.split().str.len().to_numpy()
        for slot_index, (kind, order) in enumerate(CANDIDATE_SLOTS):
            shown_examples = other_examples.iloc[
                random_generator.choice(
                    len(other_examples), size=EXAMPLES_PER_PROMPT, replace=False
                )
            ]
            example_text = "\n\n".join(
                f"Question: {example['question']}\nStatement: {example['statement']}"
                for example in shown_examples.to_dict("records")
            )
            rows.append(
                {
                    "question_id": group["question_id"],
                    "split": group["split"],
                    "sample_index": slot_index,
                    "candidate_kind": kind,
                    "target_participants": list(order) if kind == "participant" else [],
                    "opinion_order": list(order),
                    "prompt": statement_prompt(
                        group["question"],
                        [opinions[participant - 1] for participant in order],
                        SLOT_INSTRUCTIONS[kind],
                        example_text,
                        # Lengths follow the Habermas statements
                        int(round(random_generator.choice(example_lengths), -1)),
                    ),
                }
            )
    return pd.DataFrame(rows)


def generate_candidates(prompts: pd.DataFrame) -> pd.DataFrame:
    """One generated statement per prompt row."""
    return prompts.assign(response=generate_responses(prompts["prompt"].tolist()))


def ensure_candidates(rankings: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Every group's generated pool, generated and uploaded unless the hub has it."""
    return ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "candidates.parquet",
        HABERMAS_PREFIX,
        lambda: generate_candidates(
            candidate_prompts(groups, style_examples(rankings))
        ),
    )


def pool_statements(candidates: pd.DataFrame, question_id: str) -> list[str]:
    """A group's generated statements in sample order."""
    return (
        candidates[candidates["question_id"] == question_id]
        .sort_values("sample_index")["response"]  # pyright: ignore[reportCallIssue]
        .tolist()
    )


def complete_groups(groups: pd.DataFrame) -> pd.DataFrame:
    """Groups in which every participant wrote an opinion."""
    return groups[  # pyright: ignore[reportReturnType]
        [PLACEHOLDER_OPINION not in list(opinions) for opinions in groups["opinions"]]
    ]


def participant_preference_tensor(
    model: RewardModel,
    question: str,
    opinions: list[str],
    statements: list[str],
    device: str,
    scored_pairs: list[list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """Predicted preferences, shape (participant, statement, statement).

    ``scored_pairs`` gives each participant's pairs to score, leaving the rest NaN.
    """
    return np.stack(
        [
            pairwise_preference_tensor(
                model,
                habermas_prompt(question, opinion),
                statements,
                device,
                scored_pairs=None if scored_pairs is None else scored_pairs[position],
            )[0]
            for position, opinion in enumerate(opinions)
        ]
    )


def scoring_inputs(
    groups: pd.DataFrame, candidates: pd.DataFrame, seed: int = 1810
) -> dict[str, dict[str, dict]]:
    """``participant_preference_tensor`` inputs per split and question.

    Splits in ``SCORED_PAIR_COUNTS`` score a random sample of pairs per participant.
    """
    random_generator = np.random.default_rng(seed)
    split_inputs = {}
    for split_name, split_groups in groups.groupby("split", sort=False):
        split_inputs[split_name] = {}
        for group in split_groups.to_dict("records"):
            statements = pool_statements(candidates, group["question_id"])
            inputs = {
                "question": group["question"],
                "opinions": list(group["opinions"]),
                "statements": statements,
            }
            if split_name in SCORED_PAIR_COUNTS:
                every_pair = list(combinations(range(len(statements)), 2))
                inputs["scored_pairs"] = [
                    [
                        every_pair[index]
                        for index in sorted(
                            random_generator.choice(
                                len(every_pair),
                                size=SCORED_PAIR_COUNTS[split_name],  # pyright: ignore[reportArgumentType]
                                replace=False,
                            )
                        )
                    ]
                    for _ in inputs["opinions"]
                ]
            split_inputs[split_name][group["question_id"]] = inputs
    return split_inputs


def gold_pairs(
    groups: pd.DataFrame, candidates: pd.DataFrame, gold_tensors: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Every gold-scored unordered pool pair per participant, targeted at the gold preference."""
    pair_rows = []
    for group in groups.to_dict("records"):
        statements = pool_statements(candidates, group["question_id"])
        gold_tensor = gold_tensors[group["question_id"]]
        for participant, opinion in enumerate(group["opinions"]):
            prompt = habermas_prompt(group["question"], opinion)
            for first_index, second_index in combinations(range(len(statements)), 2):
                if np.isnan(gold_tensor[participant, first_index, second_index]):
                    continue
                pair_rows.append(
                    {
                        "question_id": group["question_id"],
                        "split": group["split"],
                        "prompt": prompt,
                        "response_a": statements[first_index],
                        "response_b": statements[second_index],
                        "preference": float(
                            gold_tensor[participant, first_index, second_index]
                        ),
                    }
                )
    return pd.DataFrame(pair_rows)
