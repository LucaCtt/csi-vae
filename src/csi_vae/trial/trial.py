import logging
import os
import random
from enum import StrEnum
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from csi_vae.aws import MessagesQueue, ModelSaver
from csi_vae.trial import dataset, fusion, vae
from csi_vae.trial.evaluator import Evaluator
from csi_vae.trial.handlers import QueueHandler, StreamHandler
from csi_vae.trial.trial_settings import TrialSettings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set PyTorch matmul precision to high for better performance on compatible hardware
torch.set_float32_matmul_precision("high")


class MessageType(StrEnum):
    """Enumeration of possible trial statuses."""

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_dataloader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader with common settings.

    Arguments:
        ds: Dataset to load data from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the beginning of each epoch.

    Returns:
        A DataLoader instance for the given dataset and settings.

    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=(torch.cuda.device_count() if torch.cuda.is_available() else 1) * 4,
        persistent_workers=True,
        pin_memory=True,
    )


def _train_and_eval(settings: TrialSettings) -> tuple[float, float]:
    """Train the autoencoder and classifier, then evaluate the accuracy on the test set."""
    full_train_ds, full_val_ds, full_test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )

    gaussians = []
    total_kl_loss = 0.0

    # Train a separate VAE for each antenna
    for antenna_select in range(settings.n_antennas):
        antenna_train_ds = dataset.SingleAntenna(full_train_ds, antenna_select)
        antenna_val_ds = dataset.SingleAntenna(full_val_ds, antenna_select)

        antenna_train_dl = _make_dataloader(antenna_train_ds, settings.batch_size, shuffle=True)
        antenna_val_dl = _make_dataloader(antenna_val_ds, settings.batch_size, shuffle=False)

        gaussian = vae.SingleAntenna(
            settings.window_size,
            settings.n_subcarriers,
            settings.latent_dim,
            settings.conv_channels,
            vae.CONV_SPECS[settings.conv_layers_spec],
        )
        gaussian.compile(fullgraph=True)

        trainer = vae.Trainer(
            gaussian,
            antenna_train_dl,
            antenna_val_dl,
            vae.TrainerParams(
                settings.lr,
                settings.patience,
                settings.warmup_epochs,
                settings.kl_max,
            ),
        )
        _, _, kl_loss = trainer.train(settings.n_epochs)

        gaussians.append(gaussian)
        total_kl_loss += kl_loss

    # Train the fusion model on the latent representations from all antennas
    full_train_dl = _make_dataloader(full_train_ds, settings.batch_size, shuffle=True)
    full_val_dl = _make_dataloader(full_val_ds, settings.batch_size, shuffle=False)

    delayed_fusion = fusion.Delayed(gaussians, settings.latent_dim, settings.n_activities, settings.n_fusion_layers)
    delayed_fusion.compile(fullgraph=True)

    trainer = fusion.Trainer(
        delayed_fusion,
        full_train_dl,
        full_val_dl,
        fusion.TrainerParams(
            settings.lr,
            settings.patience,
            settings.warmup_epochs,
        ),
    )
    trainer.train(settings.n_epochs)

    if settings.bucket_name:
        saver = ModelSaver(settings.bucket_name, settings.region_name)
        saver.save_model(delayed_fusion, f"{settings.study_name}/{settings.trial_number}/{settings.seed}.pt")

    full_test_dl = _make_dataloader(full_test_ds, settings.batch_size, shuffle=False)
    evaluator = Evaluator(delayed_fusion, full_test_dl)
    return evaluator.evaluate(), total_kl_loss / settings.n_antennas


def run_trial(settings: TrialSettings | None = None) -> None:
    """Run a single trial of training and evaluating the autoencoder and classifier."""
    settings = TrialSettings() if settings is None else settings
    _init_rng(settings.seed)

    logger.addHandler(StreamHandler(settings.study_name, settings.latent_dim, settings.trial_number, settings.seed))
    if settings.queue_url:
        queue = MessagesQueue.from_url(settings.queue_url, settings.region_name)
        logger.addHandler(
            QueueHandler(queue, settings.study_name, settings.latent_dim, settings.trial_number, settings.seed),
        )

    logger.info({"type": MessageType.STARTING})

    try:
        accuracy, kl_loss = _train_and_eval(settings)
    except vae.PosteriorCollapseError:
        logger.exception({"type": MessageType.COLLAPSE})
        raise
    except Exception:
        logger.exception({"type": MessageType.ERROR})
        raise

    logger.info({"type": MessageType.SUCCESS, "accuracy": accuracy, "kl_loss": kl_loss})


if __name__ == "__main__":
    run_trial()
