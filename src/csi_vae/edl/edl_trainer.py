import torch
from edl_losses.edl import EDLLoss, edl_inference

from csi_vae.jobs import fusion


class EDLTrainer(fusion.Trainer):
    """Wrapper around fusion.Trainer to train the classification head using the EDL loss.

    In addition to standard fusion.Trainer training:
        - Tracks the current epoch to use in the annealing of the EDL loss,
        - Applies softplus to logits before EDL loss computation to ensure positivity of evidence.
        - Uses edl inference for accuracy computation during training.
    """

    def __init__(
        self,
        model: fusion.Delayed,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        params: fusion.TrainerParams,
        criterion: EDLLoss,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the EDLTrainer."""
        super().__init__(model, train_dl, val_dl, params, device)
        self._current_epoch = 0
        self._criterion = lambda logits, y: criterion(logits, y, self._current_epoch)

    def _run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._optimizer.zero_grad()

        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            logits = self._model(x)

        logits = torch.nn.functional.softplus(logits)
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
