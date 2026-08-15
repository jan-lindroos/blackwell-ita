"""Hub locations and transfer helpers for every pipeline artifact."""

import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, file_exists, hf_hub_download

RMS_REPO = "jan-lindroos/blackwell-ita-rms"
ARTIFACTS_REPO = "jan-lindroos/blackwell-ita-artifacts"

HEADLINE_METHODS = [
    "base",
    "bt_best_of_n",
    "best_of_nash",
    "mean_criterion_best_of_n",
    "worst_criterion_best_of_n",
    "blackwell",
    "bt_mean_criterion_best_of_n",
    "bt_worst_criterion_best_of_n",
]


def model_path(dataset: str, filename: str) -> Path:
    """Download a reward-model artifact, returning its local cache path."""
    return Path(hf_hub_download(RMS_REPO, f"{dataset}/{filename}"))


def artifact_path(dataset: str, filename: str) -> Path:
    """Download a results artifact, returning its local cache path."""
    return Path(
        hf_hub_download(ARTIFACTS_REPO, f"{dataset}/{filename}", repo_type="dataset")
    )


def upload_model(dataset: str, local_path: Path) -> None:
    """Upload a reward-model artifact to the hub."""
    api = HfApi()
    api.create_repo(RMS_REPO, exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{dataset}/{local_path.name}",
        repo_id=RMS_REPO,
    )


def upload_artifact(dataset: str, local_path: Path) -> None:
    """Upload a results artifact to the hub."""
    api = HfApi()
    api.create_repo(ARTIFACTS_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{dataset}/{local_path.name}",
        repo_id=ARTIFACTS_REPO,
        repo_type="dataset",
    )


def upload_dataframe(dataset: str, filename: str, dataframe: pd.DataFrame) -> None:
    """Upload a dataframe to the artifacts repo as a parquet file."""
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / filename
        dataframe.to_parquet(path)
        upload_artifact(dataset, path)


def upsert_dataframe(
    dataset: str, filename: str, dataframe: pd.DataFrame, keys: list[str]
) -> None:
    """Upload a dataframe, replacing existing hub rows that match its keys."""
    if file_exists(ARTIFACTS_REPO, f"{dataset}/{filename}", repo_type="dataset"):
        existing = pd.read_parquet(artifact_path(dataset, filename))
        replaced = pd.MultiIndex.from_frame(existing[keys]).isin(  # pyright: ignore[reportArgumentType]
            pd.MultiIndex.from_frame(dataframe[keys])  # pyright: ignore[reportArgumentType]
        )
        dataframe = pd.concat([existing[~replaced], dataframe], ignore_index=True)  # pyright: ignore[reportAssignmentType]
    upload_dataframe(dataset, filename, dataframe)
