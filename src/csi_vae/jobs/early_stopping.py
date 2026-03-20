import torch
from torch import nn


class EarlyStopping:
    """Tracks a validation metric and saves/restores best model weights.

    Supports accuracy-based (higher is better) and loss-based
    (lower is better, with optional min delta) stepping.

    Note that the best weights are stored as a copy of the model's state_dict on the same device.
    This avoids unnecessary CPU-GPU transfers on restore, but may be a problem for GPUs with very limited memory.

    Raises:
        RuntimeError: If restore_best_weights is called before any improvement
            has been recorded.

    """

    def __init__(self, model: nn.Module, patience: int, warmup_epochs: int) -> None:
        """Initialize the EarlyStopping instance.

        Arguments:
            model: Model whose weights are tracked.
            patience: Epochs without improvement before early stopping triggers.
            warmup_epochs: Initial epochs to ignore before tracking improvements.

        """
        self.__model = model
        self.__patience = patience
        self.__warmup_remaining = warmup_epochs
        self.__best_loss: torch.Tensor = torch.tensor(float("inf"))
        self.__plateau_counter = 0
        self.__best_weights: dict[str, torch.Tensor] | None = None

    @property
    def should_stop(self) -> bool:
        """Whether training should stop due to lack of improvement."""
        if self.__best_weights is None:
            # No improvement recorded yet, so don't stop
            return False

        return self.__plateau_counter >= self.__patience

    def step(self, val_loss: torch.Tensor, delta: float = 0) -> None:
        """Step using loss (lower is better).

        Arguments:
            val_loss: Validation loss from the most recent epoch.
            delta: Minimum improvement required to reset the plateau counter.

        """
        if self.__tick_warmup():
            return

        if val_loss < self.__best_loss - delta:
            self.__plateau_counter = 0

            # Clone tensors and keep them on GPU.
            # This may be a problem for GPUs with very limited memory,
            # but avoids unnecessary CPU-GPU transfers on restore.
            self.__best_weights = {k: v.clone() for k, v in self.__model.state_dict().items()}
            self.__best_loss = val_loss
        else:
            self.__plateau_counter += 1

    def __tick_warmup(self) -> bool:
        """Decrement warmup counter. Returns True if still in warmup."""
        if self.__warmup_remaining > 0:
            self.__warmup_remaining -= 1
            return True

        return False

    def restore_best_weights(self) -> None:
        """Load the best recorded weights back into the model."""
        if self.__best_weights is None:
            msg = "No checkpoint saved; restore called before any step."
            raise RuntimeError(msg)

        self.__model.load_state_dict(self.__best_weights)
