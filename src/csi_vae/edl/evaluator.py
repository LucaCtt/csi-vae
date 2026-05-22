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
    def evaluate(self) -> tuple[float, float, float, float]:
        """Evaluate the EDL model.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - accuracy: The overall classification accuracy.
                - mean_unc_correct: The mean uncertainty for correctly classified samples.
                - mean_unc_wrong: The mean uncertainty for incorrectly classified samples.
                - cohens_d: Cohen's d effect size between the uncertainty
                    distributions of correct and incorrect classifications.

        """
        self._model.eval()

        n_correct = n_wrong = 0
        sum_unc_correct = sum_unc_correct_sq = 0.0
        sum_unc_wrong = sum_unc_wrong_sq = 0.0
        total = 0

        for x, y in self._dataloader:
            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                logits = self._model(x.to(self._device, non_blocking=True))

            pred, unc, _ = self._inference_fn(logits.float())
            mask_correct = pred == y.to(self._device, non_blocking=True)
            mask_wrong = ~mask_correct

            unc_correct = unc[mask_correct]
            unc_wrong = unc[mask_wrong]

            n_correct += mask_correct.sum().item()
            n_wrong += mask_wrong.sum().item()
            total += y.size(0)

            sum_unc_correct += unc_correct.sum().item()
            sum_unc_correct_sq += unc_correct.square().sum().item()
            sum_unc_wrong += unc_wrong.sum().item()
            sum_unc_wrong_sq += unc_wrong.square().sum().item()

        accuracy = n_correct / total
        mean_unc_correct = sum_unc_correct / max(n_correct, 1)
        mean_unc_wrong = sum_unc_wrong / max(n_wrong, 1)

        # Var(X) = E[X^2] - E[X]^2
        std_unc_correct = ((sum_unc_correct_sq / n_correct) - mean_unc_correct**2) ** 0.5 if n_correct > 1 else 1.0
        std_unc_wrong = ((sum_unc_wrong_sq / n_wrong) - mean_unc_wrong**2) ** 0.5 if n_wrong > 1 else 1.0

        pooled_std = ((std_unc_correct**2 + std_unc_wrong**2) / 2.0) ** 0.5
        cohens_d = (mean_unc_wrong - mean_unc_correct) / (pooled_std + 1e-8)

        return accuracy, mean_unc_correct, mean_unc_wrong, cohens_d
