import torch


class CollapseDetector:
    """Utility class to detect posterior collapse in VAE training based on KL divergence history."""

    def __init__(self, patience: int, collapse_threshold: float = 1e-5) -> None:
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
        # Check for NaN values in KL history, indicating potential numerical instability and collapse
        if any(torch.isnan(kl) or torch.isinf(kl) for kl in self.__kl_history):
            return True

        if len(self.__kl_history) < self.__patience:
            return False  # Not enough history to determine collapse

        return all(kl < self.__collapse_threshold for kl in self.__kl_history) or all(
            abs(self.__kl_history[i] - self.__kl_history[i - 1]) < self.__collapse_threshold
            for i in range(1, len(self.__kl_history))
        )
