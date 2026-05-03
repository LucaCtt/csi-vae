from collections.abc import Callable

import torch

from csi_vae.jobs import fusion


def eval_edl_model(
    model: fusion.Delayed,
    inference_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | None = None,
) -> tuple[float, float, float]:
    """Evaluate the EDL model on the given dataloader.

    Arguments:
        model (fusion.Delayed): The trained fusion.Delayed model to evaluate.
        inference_fn (Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
            A function that takes the model's logits and returns predictions, uncertainties, and p_hat values.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on.
        device (torch.device | None): The device to run the evaluation on.
            If None, uses cuda if available, otherwise cpu.

    Returns:
        tuple: A tuple containing:
            - float: Accuracy of the model on the dataloader.
            - float: Mean uncertainty for correctly classified samples.
            - float: Mean uncertainty for incorrectly classified samples.

    """
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_correct = total = n_wrong = 0
    mean_uncertainty_correct = mean_uncertainty_wrong = 0

    with torch.no_grad():
        for x, y in dataloader:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x.to(device))

            pred, uncertainty, _ = inference_fn(logits)

            mask_correct = pred == y.to(device)
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
