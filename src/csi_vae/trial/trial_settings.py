from pydantic_settings import BaseSettings


class TrialSettings(BaseSettings):
    """Trial settings, loaded from environment variables or .env file."""

    study_name: str = "default"
    """Name of the study to which this trial belongs."""
    trial_number: int = 0
    """Unique identifier for the trial within the study."""
    dataset_path: str = "dataset.h5"
    """Path to the dataset to be used for training and evaluation."""
    queue_url: str | None = None
    """URL of the SQS message queue. If set to None, the trial will not send results to a queue."""
    bucket_name: str | None = None
    """Name of the S3 bucket where results will be stored. If set to None, results will not be uploaded to S3."""
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
    patience: int = 30
    """Number of epochs to wait before raising a collapse error or early stop."""
    warmup_epochs: int = 30
    """Number of epochs to wait before early stopping."""

    seed: int = 42
    """Random seed for reproducibility."""
    batch_size: int = 128
    """Batch size for training the autoencoder."""
    lr: float = 2e-3
    """Learning rate for training the autoencoder."""
    kl_max: float = 2
    """Maximum weight for the KL divergence term during annealing."""
    latent_dim: int = 2
    """Dimensionality of the latent space in the autoencoder."""
    conv_channels: int = 32
    """Number of channels in the convolutional layers of the autoencoder."""
    conv_layers_spec: int = 0
    """Index of the convolutional layers specification to use for the autoencoder."""
    n_fusion_layers: int = 2
    """Number of layers in the delayed fusion classifier."""
