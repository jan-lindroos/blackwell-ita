"""Preference pair construction from HelpSteer2 and Community Alignment data."""

from collections.abc import Iterator

import pandas as pd

# Verbosity and complexity are excluded: they are descriptive, not monotone
# quality criteria, and a max-min winner over them would chase verbose responses
HELPSTEER2_ATTRIBUTES = [
    "helpfulness",
    "correctness",
    "coherence",
]


def graded_target(margin: float) -> float:
    """Map an ordinal rating margin to a graded win probability."""
    # Ordinal margin -> win probability on a symmetric five-level grid
    # {1, 0.75, 0.5, 0.25, 0}. Applied per annotation; pair-level targets are
    # means over annotators, so multi-annotator pairs can fall off-grid — that
    # is the disagreement signal, not noise
    if margin >= 2:
        return 1.0
    if margin >= 1:
        return 0.75
    if margin <= -2:
        return 0.0
    if margin <= -1:
        return 0.25
    return 0.5


def helpsteer2_response_pairs(
    responses: pd.DataFrame,
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """Yield response row pairs, paired consecutively within each prompt group."""
    for prompt, group in responses.groupby("prompt", sort=False):
        # Pairs sit consecutively within a prompt group (one train prompt
        # carries two separate pairs, i.e. four rows); an odd group breaks the
        # pairing assumption, so fail loudly instead of corrupting the pairs
        assert len(group) % 2 == 0, f"odd response group for prompt {prompt!r}"
        for start in range(0, len(group), 2):
            yield group.iloc[start], group.iloc[start + 1]


def helpsteer2_pairs(
    responses: pd.DataFrame, preferences: pd.DataFrame
) -> pd.DataFrame:
    """Build preference pairs with per-attribute and overall targets."""
    rows = []
    for first, second in helpsteer2_response_pairs(responses):
        row = {
            "prompt": first["prompt"],
            "response_a": first["response"],
            "response_b": second["response"],
        }
        for attribute in HELPSTEER2_ATTRIBUTES:
            row[attribute] = graded_target(first[attribute] - second[attribute])
        rows.append(row)
    pairs = pd.DataFrame(rows)
    # Positive preference_strength means response_2 is preferred. The
    # (prompt, response_1, response_2) key is unique within the train split
    # (verified against the source file, Aug 2026), so the merge cannot fan out
    overall = preferences.assign(
        overall=[
            graded_target(-strength) for strength in preferences["preference_strength"]
        ]
    )
    return pairs.merge(
        overall[["prompt", "response_1", "response_2", "overall"]],
        how="left",
        left_on=["prompt", "response_a", "response_b"],
        right_on=["prompt", "response_1", "response_2"],
    ).drop(columns=["response_1", "response_2"])


def helpsteer2_anchors(
    validation_dataframe: pd.DataFrame,
    preferences: pd.DataFrame,
    count: int = 100,
) -> dict[str, str]:
    """Anchor per prompt: the overall-preferred response, for the first ``count``.

    Pairs without a preference annotation or tied on overall preference carry
    no strict human overall winner, so their prompts are skipped.
    """
    strengths = preferences.set_index(["prompt", "response_1", "response_2"])[
        "preference_strength"
    ]
    anchors: dict[str, str] = {}
    for first, second in helpsteer2_response_pairs(validation_dataframe):
        prompt = first["prompt"]
        if prompt in anchors:
            continue
        # Positive preference_strength means response_2 (``second``) is preferred
        strength = strengths.get((prompt, first["response"], second["response"]), 0)
        if strength == 0:
            continue
        anchors[prompt] = (  # pyright: ignore[reportArgumentType]
            second["response"] if strength > 0 else first["response"]  # pyright: ignore[reportOptionalOperand]
        )
        if len(anchors) == count:
            break
    return anchors


def prompt_splits(
    prompts: pd.Series, evaluation_count: int = 100, seed: int = 1810
) -> tuple[list[str], list[str], list[str]]:
    """Split unique prompts into a held-out evaluation set and two halves."""
    shuffled = prompts.drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    held_out = shuffled[:evaluation_count]
    remaining = shuffled[evaluation_count:]
    half = len(remaining) // 2
    return held_out, remaining[:half], remaining[half:]


def annotator_group(country: str, age: str) -> str:
    """Name the annotator group from country and age bucket."""
    # Fail loudly if the dataset ever grows a country or age bucket the
    # mapping would silently fold into the k=4 {US, India} x {18-34, 35+}
    # group design; both sets verified against the source CSV, Aug 2026
    assert country in {"india", "united states"}, country
    assert age in {"18-34", "35-45", "46-54", "55+"}, age
    age_bucket = "18_34" if age == "18-34" else "35_plus"
    return f"{country.replace(' ', '_')}_{age_bucket}"


def community_alignment_pairs(
    conversations: pd.DataFrame, language: str = "en"
) -> pd.DataFrame:
    """Build preference pairs with per-annotator-group and overall targets."""
    first_turns = conversations[
        (conversations["assigned_lang"] == language)
        & conversations["first_turn_preferred_response"].notna()
        & conversations["annotator_age"].notna()
    ]
    rows = []
    for _, conversation in first_turns.iterrows():
        preferred_letter = conversation["first_turn_preferred_response"].removeprefix(  # pyright: ignore[reportAttributeAccessIssue]
            "response_"
        )
        preferred = conversation[f"first_turn_response_{preferred_letter}"]
        group = annotator_group(
            conversation["annotator_country"], conversation["annotator_age"]  # pyright: ignore[reportArgumentType]
        )
        for letter in "abcd":
            if letter == preferred_letter:
                continue
            other = conversation[f"first_turn_response_{letter}"]
            # A few source rows duplicate the preferred text verbatim; a pair
            # against itself carries no preference signal and would train the
            # model on contradictory labels for one input
            if pd.isna(other) or other == preferred:  # pyright: ignore[reportGeneralTypeIssues]
                continue
            response_a, response_b = sorted([preferred, other])
            rows.append(
                {
                    "prompt": conversation["first_turn_prompt"],
                    "response_a": response_a,
                    "response_b": response_b,
                    group: float(response_a == preferred),
                    "overall": float(response_a == preferred),
                }
            )
    return (  # pyright: ignore[reportReturnType]
        pd.DataFrame(rows)
        .groupby(["prompt", "response_a", "response_b"], as_index=False)
        .mean()
    )


def community_alignment_anchors(
    pairs_dataframe: pd.DataFrame, held_out_prompts: list[str]
) -> dict[str, str]:
    """Anchor per held-out prompt: the response with the best win fraction."""
    held_out = pairs_dataframe[pairs_dataframe["prompt"].isin(held_out_prompts)]
    # Each pair credits both responses: response_a with its win fraction and
    # response_b with the complement; pooling is a mean over a response's pairs
    credits = list(
        zip(held_out["prompt"], held_out["response_a"], held_out["overall"], strict=True)
    ) + list(
        zip(
            held_out["prompt"],
            held_out["response_b"],
            1.0 - held_out["overall"],
            strict=True,
        )
    )
    win_fractions: pd.DataFrame = (  # pyright: ignore[reportAssignmentType]
        pd.DataFrame(credits, columns=["prompt", "response", "win"])
        .groupby(["prompt", "response"], as_index=False)["win"]
        .mean()
    )
    return {  # pyright: ignore[reportReturnType]
        prompt: group.loc[group["win"].idxmax(), "response"]
        for prompt, group in win_fractions.groupby("prompt")
    }
