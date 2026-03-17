import torch


class CollapseDetector:
    """Utility class to detect posterior collapse in VAE training based on KL divergence history."""

    def __init__(self, patience: int, collapse_threshold: float = 1e-4) -> None:
        """Initialize the CollapseDetector with specified parameters.

        Arguments:
            patience: Number of epochs to consider for collapse detection.
            collapse_threshold: KL loss threshold below which the model is considered collapsed.

        """
        self.__patience = patience
        self.__collapse_threshold = collapse_threshold
        self.__kl_history: list[torch.Tensor] = []

    def step(self, kl_loss: torch.Tensor) -> None:
        """Add a new KL loss value and check for collapse.

        Arguments:
            kl_loss: The KL divergence loss for the current epoch.

        """
        self.__kl_history.append(kl_loss)
        self.__kl_history = self.__kl_history[-self.__patience :]

    def is_collapsed(self) -> bool:
        """Check if the model is considered collapsed based on recent KL loss history."""
        if len(self.__kl_history) < self.__patience:
            return False  # Not enough history to determine collapse

        # Check if average KL loss is below the collapse threshold
        if torch.mean(torch.stack(self.__kl_history)) < self.__collapse_threshold:
            return True

        # Check if KL loss is not changing significantly, indicating potential collapse
        return bool(torch.std(torch.stack(self.__kl_history)) < self.__collapse_threshold)
