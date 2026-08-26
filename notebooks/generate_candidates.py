# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "huggingface_hub",
#     "pandas",
#     "pyarrow",
#     "torch",
#     # molab's base image leaks a torchvision built against a different
#     # torch. Install a matching one so transformers doesn't import the
#     # broken system copy
#     "torchvision",
#     "transformers",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()

with app.setup:
    import tempfile
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ARTIFACTS_REPO = "jan-lindroos/blackwell-ita-artifacts"


@app.function
def artifact_path(filename: str) -> Path:
    """Download a helpsteer2 artifact, returning its local cache path."""
    return Path(
        hf_hub_download(ARTIFACTS_REPO, f"helpsteer2/{filename}", repo_type="dataset")
    )


@app.function
def upload_dataframe(filename: str, dataframe: pd.DataFrame) -> None:
    """Upload a dataframe to the artifacts repo as a helpsteer2 parquet file."""
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / filename
        dataframe.to_parquet(path)
        HfApi().upload_file(
            path_or_fileobj=path,
            path_in_repo=f"helpsteer2/{filename}",
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
    evaluation = pairs[pairs["split"] == "evaluation"]
    # pandas-stubs type itertuples rows as plain tuples without the fields
    for row in evaluation.itertuples():
        if row.prompt in anchors or pd.isna(row.overall) or row.overall == 0.5:  # pyright: ignore[reportAttributeAccessIssue]
            continue
        anchors[row.prompt] = row.response_a if row.overall > 0.5 else row.response_b  # pyright: ignore[reportAttributeAccessIssue]
    return pd.DataFrame({"prompt": list(anchors), "anchor": list(anchors.values())})


@app.function
def batch_sizes(total: int, batch_size: int) -> list[int]:
    """Split a sample count into generate() batch sizes."""
    return [min(batch_size, total - start) for start in range(0, total, batch_size)]


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
def default_device() -> str:
    """Pick the best available torch device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.function
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
        for count in batch_sizes(samples_per_prompt, batch_size):
            outputs = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=count,
                pad_token_id=tokenizer.eos_token_id,
            )
            rows.extend(
                {
                    "prompt": prompt,
                    "response": tokenizer.decode(
                        output[inputs["input_ids"].shape[1] :], skip_special_tokens=True
                    ),
                }
                for output in outputs
            )
    return pd.DataFrame(rows)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before running, switch to GPU and sign in with `hf auth login` (terminal
    available via the command menu).
    """)


@app.cell
def _():
    base_model_name = "RLHFlow/LLaMA3-SFT-v2"
    samples_per_prompt = 128
    return base_model_name, samples_per_prompt


@app.cell
def _():
    pairs_dataframe = pd.read_parquet(artifact_path("pairs.parquet"))
    anchors_dataframe = select_anchors(pairs_dataframe)
    len(anchors_dataframe)
    return (anchors_dataframe,)


@app.cell
def _():
    generate_button = mo.ui.run_button(label="Generate candidates")
    generate_button
    return (generate_button,)


@app.cell
def _(anchors_dataframe, base_model_name, generate_button, samples_per_prompt):
    mo.stop(not generate_button.value)
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
    upload_dataframe("anchors.parquet", anchors_dataframe)
    upload_dataframe("candidates.parquet", candidates_dataframe)
    candidates_dataframe

if __name__ == "__main__":
    app.run()
