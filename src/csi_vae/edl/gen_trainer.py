from typing import Literal

import torch
from edl_losses.gen import GENLoss, gen_inference
from torch import nn

from csi_vae.jobs import fusion


class LatentGenerator(nn.Module):
    """Generator G from the GEN paper (Eq. 8).

    Takes a latent code z and outputs the std of the perturbation epsilon.
    Trained to produce perturbations that are:
      - similar to real latents in latent space (via D')
      - distinguishable from real samples in input space (via D)
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32) -> None:
        """Initialize the latent generator.

        Arguments:
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Number of hidden units in the generator network.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),  # std > 0
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Output the std of the perturbation epsilon for a given latent code z.

        Arguments:
            z (torch.Tensor): Input latent code of shape (B, latent_dim).

        Returns:
            torch.Tensor: Perturbation standard deviation of shape (B, latent_dim).

        """
        return self.net(z)

    def perturb(self, z: torch.Tensor) -> torch.Tensor:
        """Perturb the latent code z by adding Gaussian noise with std given by the generator.

        Arguments:
            z (torch.Tensor): Input latent code of shape (B, latent_dim).

        Returns:
            torch.Tensor: Perturbed latent code of shape (B, latent_dim).

        """
        std = self.forward(z)
        epsilon = torch.randn_like(z) * std
        return z + epsilon


class InputDiscriminator(nn.Module):
    """Discriminator D from the GEN paper (Eq. 10).

    Distinguishes real CSI windows from decoded OOD samples in input space.
    Input: flattened CSI window for one antenna (window_size * n_subcarriers).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 400) -> None:
        """Initialize the input discriminator.

        Arguments:
            input_dim (int): Dimensionality of the input space (window_size * n_subcarriers).
            hidden_dim (int): Number of hidden units in the discriminator network.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output the probability that x is a real sample (not generated).

        Arguments:
            x (torch.Tensor): Input tensor of shape (B, window_size * n_subcarriers).

        Returns:
            torch.Tensor: Probability of being a real sample, shape (B, 1).

        """
        return self.net(x.flatten(1))


class LatentDiscriminator(nn.Module):
    """Discriminator D' from the GEN paper (Eq. 9).

    Distinguishes real latent codes from perturbed ones.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32) -> None:
        """Initialize the latent discriminator.

        Arguments:
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Number of hidden units in the discriminator network.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Output the probability that z is a real latent code (not perturbed).

        Arguments:
            z (torch.Tensor): Input latent code of shape (B, latent_dim).

        Returns:
            torch.Tensor: Probability of being a real latent code, shape (B, 1).

        """
        return self.net(z)


class AntennaOODGenerator(nn.Module):
    """VAE+GAN OOD generator for a single antenna, as in Sensoy et al. 2020.

    Wraps G, D, D' for one antenna's latent space.
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,  # window_size * n_subcarriers
        hidden_dim: int = 32,
    ) -> None:
        """Initialize the antenna OOD generator.

        Arguments:
            latent_dim (int): Dimensionality of the latent space.
            input_dim (int): Dimensionality of the input space (window_size * n_subcarriers).
            hidden_dim (int): Number of hidden units in the generator and discriminator networks.

        """
        super().__init__()
        self.G = LatentGenerator(latent_dim, hidden_dim)
        self.D = InputDiscriminator(input_dim)
        self.D_prime = LatentDiscriminator(latent_dim, hidden_dim)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate a perturbed latent code from the input latent code z.

        Arguments:
            z (torch.Tensor): Input latent code of shape (B, latent_dim).

        Returns:
            torch.Tensor: Perturbed latent code of shape (B, latent_dim).

        """
        return self.G.perturb(z)

    def generator_loss(
        self,
        z_perturbed: torch.Tensor,
        x_bar: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for the generator G, combining both objectives.

        Implements Eq. 8: max_G [ log D'(z+eps) + log(1 - D(x_bar)) ]

        Arguments:
            z_perturbed (torch.Tensor): Perturbed latent code of shape (B, latent_dim).
            x_bar (torch.Tensor): Decoded OOD sample from z_perturbed, shape (B, window_size * n_subcarriers).

        Returns:
            torch.Tensor: Generator loss (to minimize).

        """
        a = torch.log(self.D_prime(z_perturbed) + 1e-8).mean()
        b = torch.log(1 - self.D(x_bar) + 1e-8).mean()
        return -(a + b)

    def discriminator_latent_loss(
        self,
        z: torch.Tensor,
        z_perturbed: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for the latent discriminator D'.

        Implements Eq. 9: max_{D'} [ log D'(z) + log(1 - D'(z+eps)) ]

        Arguments:
            z (torch.Tensor): Real latent code of shape (B, latent_dim).
            z_perturbed (torch.Tensor): Perturbed latent code of shape (B, latent_dim).

        Returns:
            torch.Tensor: Discriminator D' loss (to minimize).

        """
        real = torch.log(self.D_prime(z.detach()) + 1e-8).mean()
        fake = torch.log(1 - self.D_prime(z_perturbed.detach()) + 1e-8).mean()
        return -(real + fake)

    def discriminator_input_loss(
        self,
        x_real: torch.Tensor,
        x_bar: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for the input discriminator D.

        Implements eq. 10: max_D [ log D(x_i) + log(1 - D(x_bar)) ]

        Arguments:
            x_real (torch.Tensor): Real input sample of shape (B, window_size * n_subcarriers).
            x_bar (torch.Tensor): Decoded OOD sample from perturbed latent code, shape (B, window_size * n_subcarriers).

        Returns:
            torch.Tensor: Discriminator D loss (to minimize).

        """
        real = torch.log(self.D(x_real) + 1e-8).mean()
        fake = torch.log(1 - self.D(x_bar.detach()) + 1e-8).mean()
        return -(real + fake)


class GENTrainerParams(fusion.TrainerParams):
    """Parameters for GENTrainer."""

    beta: float | Literal["auto", "anneal"]
    """Coefficient for the OOD loss term in the GEN loss;"""
    anneal_epochs: int
    """If beta is 'anneal', number of epochs over which to anneal beta from 0 to 1."""
    gan_hidden_dim: int
    """Hidden dimension for G, D, D'."""
    gan_lr: float
    """Learning rate for GAN components."""


class GENTrainer(fusion.Trainer):
    """GEN trainer with VAE+GAN-based OOD sample generation."""

    def __init__(
        self,
        model: fusion.Delayed,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        params: GENTrainerParams,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the GEN trainer.

        Arguments:
            model (fusion.Delayed): The GEN model to train.
            train_dl (DataLoader): DataLoader for the training set.
            val_dl (DataLoader): DataLoader for the validation set.
            params (GENTrainerParams): Training parameters, including GEN-specific ones.
            device (torch.device | None): Device to use for training. If None, uses CUDA if available.

        """
        super().__init__(model, train_dl, val_dl, params, device)
        self._current_epoch = 0
        self._criterion = GENLoss(beta=params["beta"], anneal_epochs=params["anneal_epochs"])

        self._ood_generators = nn.ModuleList()  # One AntennaOODGenerator per antenna
        self._gan_optimizers = []  # Separate optimizers for each antenna's GAN components
        self._gan_scalers = []  # Separate grad scalers for mixed precision training of GAN components

        with torch.no_grad():
            # Use a dummy batch to infer dimensions for the OOD generators and initialize them
            dummy_x = next(iter(train_dl))[0][:1].to(self._device)
            for i, antenna in enumerate(model.antennas):
                with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                    mu, _ = antenna.encode(dummy_x[:, i])
                latent_d = mu.shape[-1]
                input_d = dummy_x[:, i].flatten(1).shape[-1]

                gen = AntennaOODGenerator(
                    latent_dim=latent_d,
                    input_dim=input_d,
                    hidden_dim=params["gan_hidden_dim"],
                ).to(self._device)
                self._ood_generators.append(gen)

                opt = torch.optim.Adam(
                    [
                        {"params": gen.G.parameters()},
                        {"params": gen.D.parameters()},
                        {"params": gen.D_prime.parameters()},
                    ],
                    lr=params["gan_lr"],
                )
                self._gan_optimizers.append(opt)
                self._gan_scalers.append(torch.GradScaler(device=self._device.type))

    @torch.no_grad()
    def _generate_ood_samples(self, x: torch.Tensor) -> torch.Tensor:
        """Generate OOD samples using the VAE+GAN generators."""
        ood_list = []
        for i, (antenna, gen) in enumerate(zip(self._model.antennas, self._ood_generators, strict=True)):
            if not isinstance(gen, AntennaOODGenerator):
                msg = f"Expected AntennaOODGenerator, got {type(gen)}"
                raise TypeError(msg)

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                mu, logvar = antenna.encode(x[:, i])

            z = mu.float() + torch.exp(0.5 * logvar.float()) * torch.randn_like(logvar.float())

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                z_ood = gen.generate(z)
                x_ood = antenna.decode(z_ood)

            ood_list.append(x_ood)

        return torch.stack(ood_list, dim=1)

    def _update_gan(self, x: torch.Tensor) -> None:
        """One GAN update step for all antenna generators (Eqs. 8, 9, 10)."""
        for i, (antenna, gen, opt, scaler) in enumerate(
            zip(self._model.antennas, self._ood_generators, self._gan_optimizers, self._gan_scalers, strict=True),
        ):
            if not isinstance(gen, AntennaOODGenerator):
                msg = f"Expected AntennaOODGenerator, got {type(gen)}"
                raise TypeError(msg)

            opt.zero_grad()

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                with torch.no_grad():
                    mu, logvar = antenna.encode(x[:, i])

                z = mu.float() + torch.exp(0.5 * logvar.float()) * torch.randn_like(logvar.float())
                z_perturbed = gen.G.perturb(z)

                with torch.no_grad():
                    x_bar = antenna.decode(z_perturbed)

                # D' step (Eq. 9)
                loss_d_prime = gen.discriminator_latent_loss(z, z_perturbed)

                # D step (Eq. 10)
                loss_d = gen.discriminator_input_loss(x[:, i], x_bar)

                # G step (Eq. 8) — needs fresh z_perturbed with grad
                z_perturbed_g = gen.G.perturb(z)
                x_bar_g = antenna.decode(z_perturbed_g)
                loss_g = gen.generator_loss(z_perturbed_g, x_bar_g)

            scaler.scale(loss_d_prime).backward()
            scaler.scale(loss_d).backward()
            opt.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.step(opt)
            scaler.update()

    def _run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Update GAN
        self._update_gan(x)

        # 2. Generate OOD and train classifier
        self._optimizer.zero_grad()
        x_ood = self._generate_ood_samples(x)

        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            logits_in = self._model(x)
            logits_out = self._model(x_ood)

        loss = self._criterion(logits_in.float(), logits_out.float(), y, epoch=self._current_epoch)
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        class_indices, _, _ = gen_inference(logits_in.float())
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
            x_ood = self._generate_ood_samples(x)

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                logits_in = self._model(x)
                logits_out = self._model(x_ood)

            loss = self._criterion(logits_in.float(), logits_out.float(), y, epoch=self._current_epoch)
            total_loss += loss.detach()

        return total_loss / self._len_val
