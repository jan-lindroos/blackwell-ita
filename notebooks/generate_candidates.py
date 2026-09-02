# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "huggingface_hub",
#     "pandas",
#     "pyarrow",
#     # molab's base image leaks a torchvision built against a different
#     # torch. Install a matching one so transformers doesn't import the
#     # broken system copy
#     "torchvision",
#     "transformers",
#     "vllm",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()

with app.setup:
    import os
    import tempfile
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    ARTIFACTS_REPO = "blackwell-ita/artifacts"
    SPLITS_REPO = "blackwell-ita/helpsteer2-splits"
    DEFAULT_BASE_MODEL = "RLHFlow/LLaMA3-SFT-v2"


@app.function
def pairs_path() -> Path:
    """Download the canonical split pairs, returning the local cache path."""
    return Path(hf_hub_download(SPLITS_REPO, "pairs.parquet", repo_type="dataset"))


@app.function
def pool_prefix(model_name: str) -> str:
    """Artifact path prefix for a backbone's candidate pools.

    The default backbone keeps the original flat helpsteer2/ layout the hub
    artifacts already use; other backbones get their own subfolder so runs
    cannot clobber each other.
    """
    if model_name == DEFAULT_BASE_MODEL:
        return "helpsteer2"
    return f"helpsteer2/{model_name.split('/')[-1].lower()}"


@app.function
def upload_dataframe(filename: str, dataframe: pd.DataFrame, prefix: str) -> None:
    """Upload a dataframe to the artifacts repo as a parquet file."""
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / filename
        dataframe.to_parquet(path)
        HfApi().upload_file(
            path_or_fileobj=path,
            path_in_repo=f"{prefix}/{filename}",
            repo_id=ARTIFACTS_REPO,
            repo_type="dataset",
        )


@app.function
def select_anchors(pairs: pd.DataFrame) -> pd.DataFrame:
    """Anchor per evaluation prompt: the overall-preferred pair response.

    Pairs tied on overall preference (0.5) or missing it carry no strict
    human winner, so they are skipped; the first decisive pair per prompt
    wins. Row order is the canonical evaluation-prompt order.
    """
    anchors: dict[str, str] = {}
    evaluation = pairs[pairs["split"] == "ita_holdout"]
    # pandas-stubs type itertuples rows as plain tuples without the fields
    for row in evaluation.itertuples():
        if row.prompt in anchors or pd.isna(row.overall) or row.overall == 0.5:  # pyright: ignore[reportAttributeAccessIssue]
            continue
        anchors[row.prompt] = row.response_a if row.overall > 0.5 else row.response_b  # pyright: ignore[reportAttributeAccessIssue]
    return pd.DataFrame({"prompt": list(anchors), "anchor": list(anchors.values())})


@app.function
def combine_with_anchors(
    candidates: pd.DataFrame, anchors: pd.DataFrame, samples_per_prompt: int
) -> pd.DataFrame:
    """Index each prompt's samples 0..N-1 and append its anchor as sample N.

    The anchor rides along at index N so downstream preference tensors
    cover it in the same pass; it never enters the candidate pool.
    """
    anchor_texts = anchors.set_index("prompt")["anchor"]
    parts = []
    for prompt, group in candidates.groupby("prompt", sort=False):
        assert len(group) == samples_per_prompt, prompt
        parts.append(
            pd.DataFrame(
                {
                    "prompt": prompt,
                    "sample_index": range(samples_per_prompt + 1),
                    "response": [*group["response"], anchor_texts[prompt]],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


@app.function
def token_counts(texts: list[str], tokenizer) -> list[int]:
    """Token count per text under the base policy tokenizer."""
    return [len(ids) for ids in tokenizer(texts)["input_ids"]]


@app.function
def generate_candidates(
    model_name: str,
    prompts: list[str],
    samples_per_prompt: int,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    seed: int = 1810,
) -> pd.DataFrame:
    """Sample responses per prompt with vLLM; returns a (prompt, response) dataframe."""
    # flashinfer's top-k/top-p sampling kernel is JIT-compiled and needs
    # nvcc, which molab's image lacks; the torch sampler needs no toolkit
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # imported lazily: vllm has no macOS wheels, and tests import this module
    from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]

    llm = LLM(model=model_name)
    params = SamplingParams(
        n=samples_per_prompt,
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(conversations, params)
    return pd.DataFrame(
        {"prompt": prompt, "response": completion.text}
        for prompt, output in zip(prompts, outputs, strict=True)
        for completion in output.outputs
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before running, switch to GPU and sign in with `hf auth login` (terminal
    available via the command menu).
    """)


@app.cell
def _():
    base_model_dropdown = mo.ui.dropdown(
        options=[
            DEFAULT_BASE_MODEL,
            "google/gemma-2b-it",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        value=DEFAULT_BASE_MODEL,
        label="base model",
    )
    samples_per_prompt = 128
    base_model_dropdown
    return base_model_dropdown, samples_per_prompt


@app.cell
def _():
    pairs_dataframe = pd.read_parquet(pairs_path())
    anchors_dataframe = select_anchors(pairs_dataframe)
    len(anchors_dataframe)
    return (anchors_dataframe,)


@app.cell
def _():
    generate_button = mo.ui.run_button(label="Generate candidates")
    generate_button
    return (generate_button,)


@app.cell
def _(anchors_dataframe, base_model_dropdown, generate_button, samples_per_prompt):
    mo.stop(not generate_button.value)
    base_model_name = base_model_dropdown.value
    raw_candidates = generate_candidates(
        base_model_name,
        anchors_dataframe["prompt"].tolist(),
        samples_per_prompt,
    )
    candidates_dataframe = combine_with_anchors(
        raw_candidates, anchors_dataframe, samples_per_prompt
    )
    candidates_dataframe = candidates_dataframe.assign(
        tokens=token_counts(
            candidates_dataframe["response"].tolist(),
            AutoTokenizer.from_pretrained(base_model_name),
        )
    )
    upload_dataframe("anchors.parquet", anchors_dataframe, pool_prefix(base_model_name))
    upload_dataframe(
        "candidates.parquet", candidates_dataframe, pool_prefix(base_model_name)
    )
    candidates_dataframe

if __name__ == "__main__":
    app.run()
