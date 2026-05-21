from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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
    def evaluate(self) -> tuple[float, float, float, float]:
        """Evaluate the EDL model on the given dataloader.

        Returns:
            tuple: A tuple containing:
                - float: Accuracy of the model on the dataloader.
                - float: Mean uncertainty for correctly classified samples.
                - float: Mean uncertainty for incorrectly classified samples.
                - float: AUROC for misclassification detection.
                    Measures how well uncertainty discriminates wrong from correct predictions.
                    0.5 = random, 1.0 = perfect separation.

        """
        self._model.eval()

        all_uncertainties = []
        all_correct = []

        for x, y in self._dataloader:
            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                logits = self._model(x.to(self._device))

            pred, uncertainty, _ = self._inference_fn(logits.float())
            mask_correct = pred == y.to(self._device)

            all_correct.extend(mask_correct.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())

        all_correct = np.array(all_correct)
        all_uncertainties = np.array(all_uncertainties)

        accuracy = all_correct.mean()
        unc_correct = all_uncertainties[all_correct].mean() if all_correct.any() else 0.0
        unc_wrong = all_uncertainties[~all_correct].mean() if (~all_correct).any() else 0.0

        # AUROC: label=1 for wrong predictions (should have high uncertainty)
        n_wrong = (~all_correct).sum()
        n_correct = all_correct.sum()
        auroc = (
            0.5
            if n_wrong < _MIN_SAMPLES_FOR_AUROC or n_correct < _MIN_SAMPLES_FOR_AUROC
            else roc_auc_score(~all_correct, all_uncertainties)
        )  # degenerate — nearly perfect accuracy or all wrong

        return accuracy, float(unc_correct), float(unc_wrong), float(auroc)
