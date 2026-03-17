import torch
from torch.utils.data import DataLoader

from csi_vae.jobs import fusion


class Evaluator:
    """Evaluator for VAE and classifier performance."""

    def __init__(
        self,
        model: fusion.Delayed,
        dataloader: DataLoader,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Evaluator with the VAE, classifier, and dataloader."""
        self.__device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = model.to(self.__device)
        self.__dataloader = dataloader

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the classifier on encoded representations.

        Arguments:
            vae : Trained VAE for encoding data.
            clf : Trained classifier.
            dataloader : DataLoader for evaluation data.
            device : Device to run evaluation on. If None, uses CUDA if available.

        Returns:
            Accuracy of the classifier on the data.

        """
        self.__model.eval()

        correct = total = 0

        for batch_x, batch_y in self.__dataloader:
            x, y = batch_x.to(self.__device), batch_y.to(self.__device)
            with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
                preds = self.__model(x).argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

        return correct / total
