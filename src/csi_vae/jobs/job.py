import logging
import os
import random
from enum import StrEnum
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from csi_vae.aws import MessagesQueue, ModelSaver
from csi_vae.jobs import dataset, fusion, vae
from csi_vae.jobs.evaluator import Evaluator
from csi_vae.jobs.handlers import QueueHandler, StreamHandler
from csi_vae.jobs.job_settings import JobSettings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set PyTorch matmul precision to high for better performance on compatible hardware
torch.set_float32_matmul_precision("high")


class MessageType(StrEnum):
    """Enumeration of possible job statuses."""

    STARTING = "STARTING"
    SUCCESS = "SUCCESS"
    COLLAPSE = "COLLAPSE"
    ERROR = "ERROR"


def _init_rng(seed: int) -> None:
    """Initialize random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloader(ds: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Create a DataLoader with common settings.

    Arguments:
        ds: Dataset to load data from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the beginning of each epoch.
        seed: Random seed for reproducibility of shuffling.

    Returns:
        A DataLoader instance for the given dataset and settings.

    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,  # This likely needs some tuning based on the instance type and dataset size
        persistent_workers=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def train_gaussian(
    settings: JobSettings,
    antenna_train_ds: dataset.SingleAntenna,
    antenna_val_ds: dataset.SingleAntenna,
) -> vae.SingleAntenna:
    """Train a VAE for a single antenna and return the trained model.

    Arguments:
        settings: JobSettings object containing hyperparameters and other settings for the job.
        antenna_train_ds: The training dataset containing data from the selected antenna.
        antenna_val_ds: The validation dataset containing data from the selected antenna.

    Returns:
        A SingleAntenna VAE model trained on the selected antenna's data.

    """
    antenna_train_dl = make_dataloader(antenna_train_ds, settings.batch_size, shuffle=True, seed=settings.seed)
    antenna_val_dl = make_dataloader(antenna_val_ds, settings.batch_size, shuffle=False, seed=settings.seed)

    gaussian = vae.SingleAntenna(
        settings.window_size,
        settings.n_subcarriers,
        settings.n_gaussians,
        vae.CONV_SPECS[settings.conv_layers_spec],
    )
    gaussian.compile(fullgraph=True)

    trainer = vae.Trainer(
        gaussian,
        antenna_train_dl,
        antenna_val_dl,
        vae.TrainerParams(
            lr=settings.lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
            collapse_patience=settings.collapse_patience,
            kl_max=settings.kl_max,
        ),
    )
    trainer.train(settings.n_epochs)

    return gaussian


def train_fusion(
    settings: JobSettings,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    gaussians: list[vae.SingleAntenna],
) -> fusion.Delayed:
    """Train the delayed fusion model using the trained Gaussian models and return the trained fusion model.

    Arguments:
        settings: JobSettings object containing hyperparameters and other settings for the job.
        train_ds: The training dataset, can be for all antennas or for a single one.
        val_ds: The validation dataset, can be for all antennas or for a single one.
        gaussians: A list of trained SingleAntenna VAE models, one for each antenna.

    Returns:
        A trained Delayed fusion model that combines the outputs of the Gaussian models for activity classification.

    """
    full_train_dl = make_dataloader(train_ds, settings.batch_size, shuffle=True, seed=settings.seed)
    full_val_dl = make_dataloader(val_ds, settings.batch_size, shuffle=False, seed=settings.seed)

    delayed_fusion = fusion.Delayed(
        gaussians,
        settings.n_gaussians,
        settings.n_activities,
        settings.n_fusion_layers,
        settings.fusion_dropout,
    )
    delayed_fusion.compile(fullgraph=True)

    trainer = fusion.Trainer(
        delayed_fusion,
        full_train_dl,
        full_val_dl,
        fusion.TrainerParams(
            lr=settings.lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
        ),
    )
    trainer.train(settings.n_epochs)

    return delayed_fusion


def _train_and_eval(settings: JobSettings) -> float:
    """Train the autoencoder and classifier, then evaluate the accuracy on the test set."""
    full_train_ds, full_val_ds, full_test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )

    gaussians = []
    antennas_to_train = [settings.antenna_select] if settings.antenna_select is not None else range(settings.n_antennas)
    for antenna_select in antennas_to_train:
        antenna_train_ds = dataset.SingleAntenna(full_train_ds, antenna_select)
        antenna_val_ds = dataset.SingleAntenna(full_val_ds, antenna_select)
        gaussian = train_gaussian(settings, antenna_train_ds, antenna_val_ds)
        gaussians.append(gaussian)

    delayed_fusion = train_fusion(settings, full_train_ds, full_val_ds, gaussians)

    if settings.bucket_name:
        saver = ModelSaver(settings.bucket_name, settings.region_name)
        saver.save_model(delayed_fusion, f"{settings.bucket_key}/delayed_fusion.pt")

    full_test_dl = make_dataloader(full_test_ds, settings.batch_size, shuffle=False, seed=settings.seed)
    evaluator = Evaluator(delayed_fusion, full_test_dl)

    return evaluator.evaluate()


def run_job(settings: JobSettings | None = None) -> None:
    """Run a single job of training and evaluating the autoencoder and classifier."""
    settings = JobSettings() if settings is None else settings
    _init_rng(settings.seed)

    logger.addHandler(StreamHandler(settings.n_gaussians, settings.trial_number, settings.seed))
    if settings.queue_url:
        queue = MessagesQueue.from_url(settings.queue_url, settings.region_name)
        logger.addHandler(QueueHandler(queue, settings.n_gaussians, settings.trial_number, settings.seed))

    logger.info({"type": MessageType.STARTING})

    try:
        accuracy = _train_and_eval(settings)
    except vae.PosteriorCollapseError:
        logger.exception({"type": MessageType.COLLAPSE})
        raise
    except Exception:
        logger.exception({"type": MessageType.ERROR})
        raise

    logger.info({"type": MessageType.SUCCESS, "accuracy": accuracy})


if __name__ == "__main__":
    run_job()
