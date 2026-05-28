from collections.abc import Callable

import torch

from csi_vae.jobs import fusion

_MIN_SAMPLES_FOR_AUROC = 2


class EDLEvaluator:
    """Evaluator for EDL models."""

    def __init__(
        self,
        model: fusion.Delayed,
        inference_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the EDLEvaluator.

        Arguments:
            model (fusion.Delayed): The trained fusion.Delayed model to evaluate.
            inference_fn (Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
                A function that takes the model's logits and returns predictions, uncertainties, and p_hat values.
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on.
            device (torch.device | None): The device to run the evaluation on.
                If None, uses cuda if available, otherwise cpu.

        """
        self._model = model
        self._inference_fn = inference_fn
        self._dataloader = dataloader
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, float]:
        """Evaluate the EDL model.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - accuracy: The overall classification accuracy.
                - mean_unc_correct: The mean uncertainty for correctly classified samples.
                - mean_unc_wrong: The mean uncertainty for incorrectly classified samples.

        """
        self._model.eval()

        n_correct = total = n_wrong = 0
        mean_uncertainty_correct = mean_uncertainty_wrong = 0

        with torch.no_grad():
            for x, y in self._dataloader:
                with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                    logits = self._model(x.to(self._device))

                pred, uncertainty, _ = self._inference_fn(logits.float())

                mask_correct = pred == y.to(self._device)
                mask_wrong = ~mask_correct

                total += y.size(0)
                mean_uncertainty_correct += uncertainty[mask_correct].sum().item()
                mean_uncertainty_wrong += uncertainty[mask_wrong].sum().item()
                n_correct += mask_correct.sum().item()
                n_wrong += mask_wrong.sum().item()

        return (
            n_correct / total,
            mean_uncertainty_correct / max(n_correct, 1),
            mean_uncertainty_wrong / max(n_wrong, 1),
        )
