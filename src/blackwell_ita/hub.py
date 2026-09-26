from pathlib import Path

from huggingface_hub import HfApi, file_exists, hf_hub_download

SPLITS_REPOSITORY = "blackwell-ita/helpsteer2-splits"
REWARD_MODELS_REPOSITORY = "blackwell-ita/reward-models"
HUB_PREFIX = "helpsteer2-v2"
PAIRS_FILENAME = "pairs.parquet"


def repository_type(repository_id: str) -> str:
    """The splits live in a dataset repository, checkpoints in a model one."""
    return "dataset" if repository_id == SPLITS_REPOSITORY else "model"


def hub_file_exists(repository_id: str, filename: str) -> bool:
    """Whether ``filename`` exists under the hub prefix."""
    return file_exists(
        repository_id,
        f"{HUB_PREFIX}/{filename}",
        repo_type=repository_type(repository_id),
    )


def download_hub_file(repository_id: str, filename: str) -> Path:
    """Download ``filename`` from under the hub prefix, returning its cache path."""
    return Path(
        hf_hub_download(
            repository_id,
            f"{HUB_PREFIX}/{filename}",
            repo_type=repository_type(repository_id),
        )
    )


def upload_hub_file(repository_id: str, local_path: Path) -> None:
    """Upload ``local_path`` under the hub prefix, keeping its filename."""
    hub_api = HfApi()
    hub_api.create_repo(
        repository_id, repo_type=repository_type(repository_id), exist_ok=True
    )
    hub_api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{HUB_PREFIX}/{local_path.name}",
        repo_id=repository_id,
        repo_type=repository_type(repository_id),
    )
