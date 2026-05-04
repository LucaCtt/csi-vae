from typing import TypedDict

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


class TrainerParams(TypedDict):
    """Parameters for configuring the VAE trainer."""

    lr: float
    """Learning rate for the optimizer."""
    early_stop_patience: int
    """Patience for early stopping."""
    early_stop_warmup_epochs: int
    """Number of epochs to warm up before starting early stopping."""
    collapse_patience: int
    """Patience for detecting posterior collapse."""
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
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._gaussian = gaussian.to(self._device)
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._params = params
        self._optimizer = torch.optim.Adam(self._gaussian.parameters(), lr=params["lr"])
        self._scaler = torch.GradScaler(device=self._device.type)
        self._early_stopping = EarlyStopping(
            self._gaussian,
            params["early_stop_patience"],
            params["early_stop_warmup_epochs"],
        )
        self._collapse_detector = CollapseDetector(params["collapse_patience"])

        self._len_train = len(train_dl)
        self._len_val = len(val_dl)

    def _run_batch(self, x_true: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._optimizer.zero_grad()

        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            x_recon, mu, logvar = self._gaussian(x_true)
            loss, recon_loss, kl_loss = vae.loss(x_recon, x_true, mu, logvar, kl_weight)

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        return loss.detach(), recon_loss.detach(), kl_loss.detach()

    def _run_epoch(self, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._gaussian.train()

        metrics = torch.zeros(3, device=self._device)

        for x_true, _ in self._train_dl:
            loss, recon_loss, kl_loss = self._run_batch(x_true.to(self._device, non_blocking=True), kl_weight)
            metrics[0] += loss
            metrics[1] += recon_loss
            metrics[2] += kl_loss

        metrics /= self._len_train

        return metrics[0], metrics[1], metrics[2]

    @torch.no_grad()
    def _run_val_epoch(self, kl_weight: float) -> torch.Tensor:
        self._gaussian.eval()

        total_loss = torch.tensor(0.0, device=self._device)

        for x_true_cpu, _ in self._val_dl:
            x_true = x_true_cpu.to(self._device, non_blocking=True)

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                x_recon, mu, logvar = self._gaussian(x_true)
                loss, _, _ = vae.loss(x_recon, x_true, mu, logvar, kl_weight)

            total_loss += loss.detach()

        return total_loss / self._len_val

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Maximum number of epochs to train.

        Returns:
            Tuple of (average total loss, average recon loss, average KL loss).

        """
        total_metrics = torch.zeros(3, device=self._device)
        epochs_run = 0
        annealer = KLAnnealer(epochs, kl_max=self._params["kl_max"])

        for _ in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self._run_epoch(annealer.weight)

            # We (improperly) consider dead tensors as a form of collapse
            if _is_dead(torch.tensor([epoch_loss, epoch_recon_loss, epoch_kl_loss])):
                raise PosteriorCollapseError

            self._collapse_detector.step(epoch_kl_loss)
            if self._collapse_detector.is_collapsed():
                raise PosteriorCollapseError

            total_metrics[0] += epoch_loss
            total_metrics[1] += epoch_recon_loss
            total_metrics[2] += epoch_kl_loss
            epochs_run += 1

            val_loss = self._run_val_epoch(annealer.weight)
            self._early_stopping.step(val_loss)
            if self._early_stopping.should_stop:
                break

            annealer.step()

        self._early_stopping.restore_best_weights()

        total_metrics /= epochs_run
        return total_metrics[0].item(), total_metrics[1].item(), total_metrics[2].item()
