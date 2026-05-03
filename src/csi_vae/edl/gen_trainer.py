from dataclasses import dataclass

import torch
from edl_losses.gen import GENLoss, gen_inference

from csi_vae.jobs import fusion


@torch.no_grad()
def generate_ood_samples(
    model: fusion.Delayed,
    x: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Generate OOD samples by perturbing in the latent space of each antenna VAE.

    This is the simplified version of the VAE+GAN generator from the GEN paper —
    perturb z ~ q(z|x) by adding Gaussian noise, then decode.

    Arguments:
        model: The delayed fusion model.
        x: input tensor of shape (B, n_antennas, window_size, n_subcarriers)
        noise_scale: std of the latent perturbation.

    Returns:
        Tensor of shape (B, n_antennas, window_size, n_subcarriers) containing the OOD samples.

    """
    ood_list = []

    for i, vae_antenna in enumerate(model.antennas):
        mu, logvar = vae_antenna.encode(x[:, i])
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)  # sample z
        z_ood = z + noise_scale * torch.randn_like(z)  # perturb
        x_ood = vae_antenna.decode(z_ood)
        ood_list.append(x_ood)

    return torch.stack(ood_list, dim=1)  # (B, n_antennas, window_size, n_subcarriers)


@dataclass(frozen=True)
class GENTrainerParams(fusion.TrainerParams):
    """Parameters for GENTrainer."""

    noise_scale: float = 0.3  # std of the latent perturbation


class GENTrainer(fusion.Trainer):
    """Wrapper around fusion.Trainer to train the classification head using the GEN loss.

    In addition to standard fusion.Trainer training:
        - Generates OOD samples by perturbing the latent space of the Gaussian encoders,
        - Computes the GEN loss using both in-distribution and generated OOD samples.

    """

    def __init__(
        self,
        model: fusion.Delayed,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        params: GENTrainerParams,
        criterion: GENLoss,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the GENTrainer."""
        super().__init__(model, train_dl, val_dl, params, device)
        self._current_epoch = 0
        self._criterion = criterion
        self._noise_scale = params.noise_scale

    def _run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._optimizer.zero_grad()

        x_ood_list = generate_ood_samples(self._model, x, self._noise_scale)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            logits_in = self._model(x)
            logits_out = self._model(x_ood_list)

        loss = self._criterion(logits_in, logits_out, y)
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        class_indices, _, _ = gen_inference(logits_in)
        accuracy = (class_indices == y).float().mean()

        return loss.detach(), accuracy

    def _run_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._current_epoch += 1
        return super()._run_epoch()

    @torch.no_grad()
    def _run_val_epoch(self) -> torch.Tensor:
        self._model.eval()

        total_loss = torch.tensor(0.0, device=self._device)

        for x_cpu, y_cpu in self._val_dl:
            x, y = x_cpu.to(self._device, non_blocking=True), y_cpu.to(self._device, non_blocking=True)

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                x_ood_list = generate_ood_samples(self._model, x, self._noise_scale)
                logits_in = self._model(x)
                logits_out = self._model(x_ood_list)

            loss = self._criterion(logits_in, logits_out, y)
            total_loss += loss.detach()

        return total_loss / self._len_val
