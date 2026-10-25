import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from blackwell_ita.habermas import (
        ensure_candidates,
        ensure_groups,
        ensure_rankings,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Habermas splits and pools

    Builds the question splits, the deliberation groups and every group's candidate pool, uploading each unless the hub has it. Pool generation calls the Claude API, so this notebook runs locally with `uv run --env-file .env marimo edit notebooks/habermas_local.py`.
    """)
    return


@app.cell
def _():
    rankings = ensure_rankings()
    rankings.drop_duplicates("question_id")["split"].value_counts()
    return (rankings,)


@app.cell
def _(rankings):
    groups = ensure_groups(rankings)
    groups["split"].value_counts()
    return (groups,)


@app.cell
def _(groups, rankings):
    candidates = ensure_candidates(rankings, groups)
    candidates.groupby("candidate_kind")["response"].apply(
        lambda responses: responses.str.split().str.len().describe()
    ).unstack()
    return


if __name__ == "__main__":
    app.run()
