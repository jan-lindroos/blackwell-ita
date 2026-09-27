import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, file_exists, hf_hub_download

SPLITS_REPOSITORY = "blackwell-ita/helpsteer2-splits"
ARTIFACTS_REPOSITORY = "blackwell-ita/artifacts"
REWARD_MODELS_REPOSITORY = "blackwell-ita/reward-models"
DATASET_REPOSITORIES = {SPLITS_REPOSITORY, ARTIFACTS_REPOSITORY}
HUB_PREFIX = "helpsteer2-v2"
PAIRS_FILENAME = "pairs.parquet"


def repository_type(repository_id: str) -> str:
    """Splits and artifacts live in dataset repositories, checkpoints in a model one."""
    return "dataset" if repository_id in DATASET_REPOSITORIES else "model"


def hub_file_exists(
    repository_id: str, filename: str, prefix: str = HUB_PREFIX
) -> bool:
    """Whether ``filename`` exists under ``prefix``."""
    return file_exists(
        repository_id,
        f"{prefix}/{filename}",
        repo_type=repository_type(repository_id),
    )


def download_hub_file(
    repository_id: str, filename: str, prefix: str = HUB_PREFIX
) -> Path:
    """Download ``filename`` from under ``prefix``, returning its cache path."""
    return Path(
        hf_hub_download(
            repository_id,
            f"{prefix}/{filename}",
            repo_type=repository_type(repository_id),
        )
    )


def upload_hub_file(
    repository_id: str, local_path: Path, prefix: str = HUB_PREFIX
) -> None:
    """Upload ``local_path`` under ``prefix``, keeping its filename."""
    hub_api = HfApi()
    hub_api.create_repo(
        repository_id, repo_type=repository_type(repository_id), exist_ok=True
    )
    hub_api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{prefix}/{local_path.name}",
        repo_id=repository_id,
        repo_type=repository_type(repository_id),
    )


def upload_dataframe(
    repository_id: str, filename: str, dataframe: pd.DataFrame, prefix: str
) -> None:
    """Upload ``dataframe`` as parquet under ``prefix``."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / filename
        dataframe.to_parquet(local_path)
        upload_hub_file(repository_id, local_path, prefix)


def read_hub_dataframe(repository_id: str, filename: str, prefix: str) -> pd.DataFrame:
    """Read a parquet file from under ``prefix``."""
    return pd.read_parquet(download_hub_file(repository_id, filename, prefix))


def ensure_hub_dataframe(
    repository_id: str, filename: str, prefix: str, build: Callable[[], pd.DataFrame]
) -> pd.DataFrame:
    """Build and upload ``filename`` unless the hub has it, then read it back."""
    if not hub_file_exists(repository_id, filename, prefix):
        upload_dataframe(repository_id, filename, build(), prefix)
    return read_hub_dataframe(repository_id, filename, prefix)
