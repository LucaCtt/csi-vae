from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from csi_vae.jobs import fusion
from csi_vae.jobs.early_stopping import EarlyStopping


@dataclass(frozen=True)
class TrainerParams:
    """Parameters for the DelayedFusion Trainer."""

    lr: float
    """Learning rate for the optimizer."""
    patience: int
    """Early-stopping patience in epochs."""
    warmup_epochs: int
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
            device: Target device; defaults to CUDA if available.

        """
        self.__device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = model.to(self.__device)
        self.__train_dl = train_dl
        self.__val_dl = val_dl
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=params.lr)
        self.__scaler = torch.GradScaler(device=self.__device.type)
        self.__early_stopping = EarlyStopping(self.__model, params.patience, params.warmup_epochs)

        self.__len_train = len(train_dl)
        self.__len_val = len(val_dl)

    def __run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.__optimizer.zero_grad()

        with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
            logits = self.__model(x)
            loss = self.__criterion(logits, y)

        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()

        accuracy = (logits.detach().argmax(dim=1) == y).float().mean()

        return loss.detach(), accuracy

    def __run_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.__model.train()

        metrics = torch.zeros(2, device=self.__device)

        for x, y in self.__train_dl:
            loss, accuracy = self.__run_batch(
                x.to(self.__device, non_blocking=True),
                y.to(self.__device, non_blocking=True),
            )

            metrics[0] += loss
            metrics[1] += accuracy

        metrics /= self.__len_train

        return metrics[0], metrics[1]

    @torch.no_grad()
    def __run_val_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.__model.eval()

        metrics = torch.zeros(2, device=self.__device)

        for x_cpu, y_cpu in self.__val_dl:
            x, y = x_cpu.to(self.__device, non_blocking=True), y_cpu.to(self.__device, non_blocking=True)

            with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
                logits = self.__model(x)
                loss = self.__criterion(logits, y)

            accuracy = (logits.argmax(dim=1) == y).float().mean()
            metrics[0] += loss.detach()
            metrics[1] += accuracy.detach()

        metrics /= self.__len_val
        return metrics[0], metrics[1]

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the model, restoring best weights on early stopping.

        Arguments:
            epochs: Maximum number of epochs to train.

        Returns:
            Tuple of (average train loss, average train accuracy) over all epochs run.

        """
        total_metrics = torch.zeros(2, device=self.__device)
        epochs_run = 0

        for _ in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch()

            total_metrics[0] += epoch_loss
            total_metrics[1] += epoch_accuracy
            epochs_run += 1

            val_loss, _ = self.__run_val_epoch()
            self.__early_stopping.step_loss(val_loss)
            if self.__early_stopping.should_stop:
                break

        self.__early_stopping.restore_best_weights()

        total_metrics /= epochs_run
        return total_metrics[0].item(), total_metrics[1].item()
