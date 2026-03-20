from typing import NamedTuple, TypeVar

from pydantic_settings import BaseSettings

from csi_vae.jobs.vae import CONV_SPECS

NumberT = TypeVar("NumberT", int, float)


class ParamRange[NumberT](NamedTuple):
    """Defines a hyperparameter with a name and a range."""

    min: NumberT
    max: NumberT


class ParamCategorical[T](NamedTuple):
    """Defines a hyperparameter with a name and a list of values."""

    values: list[T]


class LauncherSettings(BaseSettings):
    """Application settings, loaded from environment variables or .env file."""

    launch_name: str = "default"
    """Name of this launch, used for naming the Optuna study and AWS Batch jobs."""
    storage_dir: str | None = f"out/{launch_name}"
    """Path to the Optuna storage dir for this study. If None, the study will be created without persistent storage."""
    n_trials: int = 100
    """Total number of Optuna trials to run."""
    starter_seed: int = 42
    """Seed used for generating the trials' seeds"""
    n_seeds_per_trial: int = 10
    """Number of different random seeds to run for each trial configuration."""
    max_pruned_seeds: int = 2
    """Maximum number of seed collapses before pruning the trial."""
    min_accuracy_delta: float = 0.02
    """Minimum improvement in best-trial accuracy required to continue to the next latent dim."""
    batch_job_queue: str = "CSIVAEJobQueue"
    """Name of the AWS Batch job queue to submit trials to."""
    batch_job_definition: str = "CSIVAEJobDefinition"
    """Name of the AWS Batch job definition to use."""
    bucket_name: str = "csi-vae-816558913136-us-east-1-an"
    """Name of the S3 bucket to use for storing trial results and logs."""
    region_name: str = "us-east-1"
    """Default AWS region to use for Batch operations."""
    poll_interval: int = 30  # 30 seconds
    """Seconds to wait between polling AWS Batch for job status."""
    poll_timeout: int = 60 * 60  # 1 hour
    """Maximum seconds to wait for a batch of trials to complete before giving up."""
    latent_dim_patience: int = 2
    """Number of consecutive latent dimensions to allow without improvement before stopping the study."""

    batch_size: ParamRange[int] = ParamRange(min=32, max=128)
    lr: ParamRange[float] = ParamRange(min=1e-3, max=3e-2)
    kl_max: ParamRange[float] = ParamRange(min=1.0, max=20.0)
    latent_dim: ParamRange[int] = ParamRange(min=1, max=10)
    conv_channels: ParamRange[int] = ParamRange(min=32, max=64)
    conv_layers_spec: ParamCategorical[int] = ParamCategorical(values=[*range(len(CONV_SPECS))])
    n_fusion_layers: ParamRange[int] = ParamRange(min=1, max=3)
