from typing import Literal

import torch
from edl_losses.edl import EDLLoss, edl_inference

from csi_vae.jobs import fusion


class EDLTrainerParams(fusion.TrainerParams):
    """Parameters for the EDLTrainer, extending fusion.TrainerParams with EDL-specific parameters."""

    loss_type: Literal["sse", "ce", "mse"]
    """Type of EDL loss to use: 'sse' (sum of squared errors), 'ce' (cross-entropy), or 'mse' (mean squared error)."""

    beta: float | Literal["anneal"]
    """Coefficient for the KL divergence term in the EDL loss;
    if 'anneal', this will be annealed from 0 to 1 over edl_anneal_epochs."""

    anneal_epochs: int
    """Number of epochs over which to anneal the KL divergence term in the EDL loss."""


class EDLTrainer(fusion.Trainer):
    """Wrapper around fusion.Trainer to train the classification head using the EDL loss.

    In addition to standard fusion.Trainer training:
        - Tracks the current epoch to use in the annealing of the EDL loss,
        - Applies exp to logits before EDL loss computation to ensure positivity of evidence.
        - Uses edl inference for accuracy computation during training.
    """

    def __init__(
        self,
        model: fusion.Delayed,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        params: EDLTrainerParams,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the EDLTrainer."""
        super().__init__(model, train_dl, val_dl, params, device)
        self._current_epoch = 0

        edl_criterion = EDLLoss(
            loss_type=params["loss_type"],
            beta=params["beta"],
            anneal_epochs=params["anneal_epochs"],
        )
        self._criterion = lambda logits, y: edl_criterion(logits, y, self._current_epoch)

    def _run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._optimizer.zero_grad()

        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            logits = self._model(x)

        loss = self._criterion(logits, y)
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        class_indices, _, _ = edl_inference(logits)
        accuracy = (class_indices == y).float().mean()

        return loss.detach(), accuracy

    def _run_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._current_epoch += 1
        return super()._run_epoch()
