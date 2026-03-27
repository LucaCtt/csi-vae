from pydantic_settings import BaseSettings


class JobSettings(BaseSettings):
    """Job settings, loaded from environment variables or .env file."""

    trial_number: int = 0
    """Unique identifier for the trial, used for logging and result storage."""
    dataset_path: str = "dataset.h5"
    """Path to the dataset to be used for training and evaluation."""
    queue_url: str | None = None
    """URL of the SQS message queue. If set to None, the job will not send results to a queue."""
    bucket_name: str | None = None
    """Name of the S3 bucket where results will be stored. If set to None, results will not be uploaded to S3."""
    bucket_key: str | None = "/"
    """Key for the S3 object key where results will be stored."""
    region_name: str = "us-east-1"
    """AWS region for configuring the S3 client when used."""
    window_size: int = 450
    """Size of the window to use when segmenting the data."""
    n_subcarriers: int = 256
    """Number of subcarriers in the CSI data."""
    n_activities: int = 12
    """Number of activities (classes) in the dataset."""
    n_antennas: int = 4
    """Number of antennas in the CSI data."""
    stride: int = 50
    """Stride to use when segmenting the data (number of samples to skip between windows)."""
    n_epochs: int = 150
    """Number of epochs to train the autoencoder."""
    early_stop_patience: int = 20
    """Number of epochs to wait before early stopping."""
    early_stop_warmup_epochs: int = 10
    """Number of epochs to wait before starting to check for early stopping."""
    collapse_patience: int = 5
    """Number of epochs to wait before collapsing the latent space."""

    seed: int = 42
    """Random seed for reproducibility."""
    batch_size: int = 128
    """Batch size for training the autoencoder."""
    lr: float = 2e-3
    """Learning rate for training the autoencoder."""
    kl_max: float = 2
    """Maximum weight for the KL divergence term during annealing."""
    n_gaussians: int = 2
    """Number of Gaussians to be produced by the autoencoder."""
    conv_layers_spec: int = 0
    """Index of the convolutional layers specification to use for the autoencoder."""
    n_fusion_layers: int = 2
    """Number of layers in the delayed fusion classifier."""
    fusion_dropout: float = 0.2
    """Dropout rate to use in the delayed fusion classifier."""
