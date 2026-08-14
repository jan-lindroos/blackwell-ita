"""Hub locations and transfer helpers for every pipeline artifact."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RMS_REPO = "janlindroos/blackwell-ita-rms"
ARTIFACTS_REPO = "janlindroos/blackwell-ita-artifacts"

HEADLINE_METHODS = [
    "base",
    "bt_best_of_n",
    "best_of_nash",
    "mean_criterion_best_of_n",
    "worst_criterion_best_of_n",
    "blackwell",
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
