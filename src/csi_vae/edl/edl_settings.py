from pydantic_settings import BaseSettings


class EDLSettings(BaseSettings):
    """Job settings, loaded from environment variables or .env file."""

    launch_studies_dir: str = "out/fusion"
    """Directory where the fusion study results are stored."""
    launch_weights_dir: str = "weights/fusion"
    """Directory where the fusion model weights are stored."""
    edl_studies_dir: str = "out/edl"
    """Directory where the study results will be stored."""
    edl_weights_dir: str = "weights/edl"
    """Directory where the EDL model weights will be stored."""
    dataset_path: str = "dataset/S1a.h5"
    """Path to the dataset to be used for training and evaluation."""
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
    n_trials: int = 100
    """Number of trials to run for hyperparameter optimization."""
    n_epochs: int = 150
    """Number of epochs to train the autoencoder."""
    early_stop_patience: int = 20
    """Number of epochs to wait before early stopping."""
    early_stop_warmup_epochs: int = 10
    """Number of epochs to wait before starting to check for early stopping."""
    objective_alpha: float = 0.5
    """Alpha parameter to balance accuracy and uncertainty in the objective function."""
