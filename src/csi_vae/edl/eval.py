from collections.abc import Callable

import torch

from csi_vae.jobs import fusion


def eval_edl_model(
    model: fusion.Delayed,
    inference_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Evaluate the EDL model on the given dataloader.

    Arguments:
        model (fusion.Delayed): The trained fusion.Delayed model to evaluate.
        inference_fn (Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
            A function that takes the model's logits and returns predictions, uncertainties, and p_hat values.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: A tuple containing:
            - float: Accuracy of the model on the dataloader.
            - float: Mean uncertainty for correctly classified samples.
            - float: Mean uncertainty for incorrectly classified samples.
            - float: Mean p_hat for correctly classified samples.
            - float: Mean p_hat for incorrectly classified samples.

    """
    model.eval()
    correct = total = 0
    mean_uncertainty_correct = mean_uncertainty_wrong = mean_p_hat_correct = mean_p_hat_wrong = 0
    n_correct = n_wrong = 0

    with torch.no_grad():
        for x, y in dataloader:
            labels = y.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x.to(device))

            pred, uncertainty, p_hat = inference_fn(logits)

            mask_correct = pred == labels
            mask_wrong = ~mask_correct

            correct += mask_correct.sum().item()
            total += labels.size(0)
            mean_uncertainty_correct += uncertainty[mask_correct].sum().item()
            mean_uncertainty_wrong += uncertainty[mask_wrong].sum().item()
            mean_p_hat_correct += p_hat[mask_correct, pred[mask_correct]].sum().item()
            mean_p_hat_wrong += p_hat[mask_wrong, pred[mask_wrong]].sum().item()
            n_correct += mask_correct.sum().item()
            n_wrong += mask_wrong.sum().item()

    return (
        correct / total,
        mean_uncertainty_correct / max(n_correct, 1),
        mean_uncertainty_wrong / max(n_wrong, 1),
        mean_p_hat_correct / max(n_correct, 1),
        mean_p_hat_wrong / max(n_wrong, 1),
    )
