from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from csi_vae.jobs import vae
from csi_vae.jobs.early_stopping import EarlyStopping
from csi_vae.jobs.vae.collapse_detector import CollapseDetector
from csi_vae.jobs.vae.kl_annealer import KLAnnealer


def _is_dead(tensor: torch.Tensor) -> bool:
    """Check if a tensor contains NaN or infinite values."""
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())


class PosteriorCollapseError(Exception):
    """Raised when the VAE posterior collapses during training."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Posterior collapse detected.")


@dataclass(frozen=True)
class TrainerParams:
    """Parameters for configuring the VAE trainer."""

    lr: float
    """Learning rate for the optimizer."""
    patience: int
    """Patience for both early stopping and collapse detection."""
    warmup_epochs: int
    """Number of epochs to warm up before starting early stopping."""
    kl_max: float
    """Maximum KL divergence weight."""


class Trainer:
    """Trainer class for VAE model."""

    def __init__(
        self,
        gaussian: vae.SingleAntenna,
        train_dl: DataLoader,
        val_dl: DataLoader,
        params: TrainerParams,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Trainer with model, data, and training parameters.

        Arguments:
            gaussian: VAE model to be trained.
            train_dl: DataLoader for training data.
            val_dl: DataLoader for validation data.
            params: TrainerParams object containing training hyperparameters.
            device: Target device; defaults to CUDA if available.

        """
        self.__device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__gaussian = gaussian.to(self.__device)
        self.__train_dl = train_dl
        self.__val_dl = val_dl
        self.__params = params
        self.__optimizer = torch.optim.Adam(self.__gaussian.parameters(), lr=params.lr)
        self.__scaler = torch.GradScaler(device=self.__device.type)
        self.__early_stopping = EarlyStopping(self.__gaussian, params.patience, params.warmup_epochs)
        self.__collapse_detector = CollapseDetector(params.patience)

        self.__len_train = len(train_dl)
        self.__len_val = len(val_dl)

    def __run_batch(self, x_true: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__optimizer.zero_grad()

        with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
            x_recon, mu, logvar = self.__gaussian(x_true)
            loss, recon_loss, kl_loss = vae.loss(x_recon, x_true, mu, logvar, kl_weight)

        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()

        return loss.detach(), recon_loss.detach(), kl_loss.detach()

    def __run_epoch(self, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__gaussian.train()

        metrics = torch.zeros(3, device=self.__device)

        for x_true, _ in self.__train_dl:
            loss, recon_loss, kl_loss = self.__run_batch(x_true.to(self.__device, non_blocking=True), kl_weight)
            metrics[0] += loss
            metrics[1] += recon_loss
            metrics[2] += kl_loss

        metrics /= self.__len_train

        return metrics[0], metrics[1], metrics[2]

    @torch.no_grad()
    def __run_val_epoch(self, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__gaussian.eval()

        metrics = torch.zeros(3, device=self.__device)

        for x_true_cpu, _ in self.__val_dl:
            x_true = x_true_cpu.to(self.__device, non_blocking=True)

            with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
                x_recon, mu, logvar = self.__gaussian(x_true)
                loss, recon_loss, kl_loss = vae.loss(x_recon, x_true, mu, logvar, kl_weight)

            metrics[0] += loss
            metrics[1] += recon_loss
            metrics[2] += kl_loss

        metrics /= self.__len_val

        return metrics[0], metrics[1], metrics[2]

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Maximum number of epochs to train.

        Returns:
            Tuple of (average total loss, average recon loss, average KL loss).

        """
        total_metrics = torch.zeros(3, device=self.__device)
        epochs_run = 0
        annealer = KLAnnealer(epochs, kl_max=self.__params.kl_max)

        for _ in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self.__run_epoch(annealer.weight)

            # We (improperly) consider explosions as a form of collapse
            if _is_dead(torch.tensor([epoch_loss, epoch_recon_loss, epoch_kl_loss])):
                raise PosteriorCollapseError

            self.__collapse_detector.step(epoch_kl_loss)
            if self.__collapse_detector.is_collapsed():
                raise PosteriorCollapseError

            total_metrics[0] += epoch_loss
            total_metrics[1] += epoch_recon_loss
            total_metrics[2] += epoch_kl_loss
            epochs_run += 1

            val_loss, _, _ = self.__run_val_epoch(annealer.weight)
            self.__early_stopping.step_loss(val_loss)
            if self.__early_stopping.should_stop:
                break

            annealer.step()

        self.__early_stopping.restore_best_weights()

        total_metrics /= epochs_run
        return total_metrics[0].item(), total_metrics[1].item(), total_metrics[2].item()
