from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from csi_vae.jobs import fusion
from csi_vae.jobs.early_stopping import EarlyStopping


class TrainerParams(TypedDict):
    """Parameters for the DelayedFusion Trainer."""

    lr: float
    """Learning rate for the optimizer."""
    early_stop_patience: int
    """Early-stopping patience in epochs."""
    early_stop_warmup_epochs: int
    """Number of epochs to warm up the learning rate."""


class Trainer:
    """Trainer for the DelayedFusion model with early stopping and best-weight restoration."""

    def __init__(
        self,
        model: fusion.Delayed,
        train_dl: DataLoader,
        val_dl: DataLoader,
        params: TrainerParams,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Trainer with model, data loaders, optimizer, and early stopping.

        Arguments:
            model: DelayedFusion model to train.
            train_dl: DataLoader for training data.
            val_dl: DataLoader for validation data.
            params: Trainer parameters.
            criterion: Loss function; defaults to CrossEntropyLoss.
            device: Target device; defaults to CUDA if available.

        """
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=params["lr"])
        self._scaler = torch.GradScaler(device=self._device.type)
        self._early_stopping = EarlyStopping(
            self._model,
            params["early_stop_patience"],
            params["early_stop_warmup_epochs"],
        )

        self._len_train = len(train_dl)
        self._len_val = len(val_dl)

    def _run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._optimizer.zero_grad()

        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            logits = self._model(x)
            loss = self._criterion(logits, y)

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        accuracy = (logits.detach().argmax(dim=1) == y).float().mean()

        return loss.detach(), accuracy

    def _run_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._model.train()

        metrics = torch.zeros(2, device=self._device)

        for x, y in self._train_dl:
            loss, accuracy = self._run_batch(
                x.to(self._device, non_blocking=True),
                y.to(self._device, non_blocking=True),
            )

            metrics[0] += loss
            metrics[1] += accuracy

        metrics /= self._len_train

        return metrics[0], metrics[1]

    @torch.no_grad()
    def _run_val_epoch(self) -> torch.Tensor:
        self._model.eval()

        total_loss = torch.tensor(0.0, device=self._device)

        for x_cpu, y_cpu in self._val_dl:
            x, y = x_cpu.to(self._device, non_blocking=True), y_cpu.to(self._device, non_blocking=True)

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                logits = self._model(x)
                loss = self._criterion(logits, y)

            total_loss += loss.detach()

        return total_loss / self._len_val

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the model, restoring best weights on early stopping.

        Arguments:
            epochs: Maximum number of epochs to train.

        Returns:
            Tuple of (average train loss, average train accuracy) over all epochs run.

        """
        total_metrics = torch.zeros(2, device=self._device)
        epochs_run = 0

        for _ in range(epochs):
            epoch_loss, epoch_accuracy = self._run_epoch()

            total_metrics[0] += epoch_loss
            total_metrics[1] += epoch_accuracy
            epochs_run += 1

            val_loss = self._run_val_epoch()
            self._early_stopping.step(val_loss)
            if self._early_stopping.should_stop:
                break

        self._early_stopping.restore_best_weights()

        total_metrics /= epochs_run
        return total_metrics[0].item(), total_metrics[1].item()
