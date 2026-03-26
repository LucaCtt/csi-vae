import torch
from torch import nn

ConvLayerSpec = list[tuple[int, int]]
"""Specification for convolutional layers.

Each entry represents (kernel_size_time, stride_time).
Subcarrier kernel_size and stride are fixed to 1, as we don't want to convolve across subcarriers.
"""

CONV_SPECS: list[ConvLayerSpec] = [
    [(5, 5), (5, 5), (3, 3)],
    [(5, 5), (5, 5), (2, 2)],
    [(3, 3), (5, 5), (5, 5)],
]


class _AntennaEncoder(nn.Module):
    """Encode a single-antenna CSI window into mean and log-variance vectors."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: ConvLayerSpec,
    ) -> None:
        """Initialize the AntennaEncoder with convolutional layers and linear heads.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The dimensionality of the latent space.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()
        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers

        layers: list[nn.Module] = []
        for kh, sh in conv_layers:
            layers.append(nn.Conv2d(n_subcarriers, n_subcarriers, kernel_size=(kh, 1), stride=(sh, 1)))
            layers.append(nn.BatchNorm2d(n_subcarriers))
            layers.append(nn.GELU())

        layers.append(nn.Flatten())
        self.__conv = nn.Sequential(*layers)

        # Infer flattened feature dimension for linear heads
        _, flat_dim = self.get_shapes()

        # Linear heads for Gaussian parameters
        self.__mu = nn.Linear(flat_dim, n_gaussians)
        self.__logvar = nn.Linear(flat_dim, n_gaussians)

    @torch.no_grad()
    def get_shapes(self) -> tuple[tuple, int]:
        """Return the latent feature map shape and its flattened size.

        Returns:
            latent_feat_shape: The shape of the feature map after convolution (Channels, H, W).
            flat_dim: The total number of features when the feature map is flattened.

        """
        x = torch.zeros(1, self.__n_subcarriers, self.__window_size, 1, device=next(self.parameters()).device)
        x = self.__conv[:-1](x)
        return x.shape[1:], int(x.numel() // x.shape[0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log-variance for a single-antenna input.

        Arguments:
            x: Input tensor of shape (batch_size, window_size, n_subcarriers) for one antenna.

        Returns:
            mu: Tensor of shape (batch_size, antenna_n_gaussians) representing the mean of the latent distribution.
            logvar: Tensor of shape (batch_size, antenna_n_gaussians)

        """
        x = x.permute(0, 2, 1).unsqueeze(-1).contiguous()  # (batch_size, n_subcarriers, window_size, 1)
        z = self.__conv(x)
        return self.__mu(z), torch.clamp(self.__logvar(z), min=-10, max=10)


class _AntennaDecoder(nn.Module):
    """Decode a latent vector back into a CSI window for a single antenna."""

    def __init__(
        self,
        latent_feat_shape: tuple,
        flat_dim: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: ConvLayerSpec,
    ) -> None:
        """Initialize the AntennaDecoder with linear and deconvolutional layers.

        Arguments:
            latent_feat_shape: The shape of the feature map before flattening in the encoder.
            flat_dim: The total number of features when the feature map is flattened.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The number of gaussians to decode from.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride)

        """
        super().__init__()
        self.__latent_feat_shape = latent_feat_shape

        self.__fc = nn.Sequential(nn.Linear(n_gaussians, flat_dim), nn.GELU())

        deconv_layers: list[nn.Module] = []
        reversed_specs = list(reversed(conv_layers))

        for i, (kh, sh) in enumerate(reversed_specs):
            deconv_layers.append(
                nn.ConvTranspose2d(n_subcarriers, n_subcarriers, kernel_size=(kh, 1), stride=(sh, 1)),
            )
            if i < len(reversed_specs) - 1:
                deconv_layers.append(nn.BatchNorm2d(n_subcarriers))
                deconv_layers.append(nn.GELU())

        self.__deconv = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector into a CSI window.

        Arguments:
            z: Input tensor of shape (batch_size, n_gaussians*2) representing the latent vector for one antenna.

        Returns:
            recon: Tensor of shape (batch_size, window_size, n_subcarriers)
                   representing the reconstructed CSI window for one antenna.

        """
        z = self.__fc(z)  # (batch_size, flat_dim)
        z = z.view(z.size(0), *self.__latent_feat_shape).contiguous()  # (batch_size, n_subcarriers, window_size', 1)
        z = self.__deconv(z)  # (batch_size, n_subcarriers, window_size, 1)
        return z.squeeze(-1).permute(0, 2, 1)  # (batch_size, window_size, n_subcarriers)


class SingleAntenna(nn.Module):
    """VAE architecture that encodes a single antenna's CSI data."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: ConvLayerSpec,
    ) -> None:
        """Initialize the SingleAntennaVAE with an encoder and decoder for single-antenna CSI data.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The number of gaussians to encode/decode in the latent space.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()

        self.__encoder = _AntennaEncoder(window_size, n_subcarriers, n_gaussians, conv_layers)
        latent_feat_shape, flat_dim = self.__encoder.get_shapes()
        self.__decoder = _AntennaDecoder(latent_feat_shape, flat_dim, n_subcarriers, n_gaussians, conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(2, window_size, n_subcarriers)
            recon, _, _ = self.forward(dummy)
            if recon.shape != dummy.shape:
                msg = f"Decoder output shape {recon.shape} does not match input shape {dummy.shape}"
                raise ValueError(msg)

    def __reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the Gaussian distribution defined by mu and logvar."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input CSI window into mean and log-variance vectors."""
        return self.__encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector to reconstruct the input."""
        return self.__decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input, sample a latent variable, and decode to reconstruct the input.

        Arguments:
            x: Input tensor of shape (batch_size, window_size, n_subcarriers).

        Returns:
            recon: Tensor of shape (batch_size, window_size, n_subcarriers) representing the reconstructed input.
            mu: Tensor of shape (batch_size, n_gaussians) representing the mean of the latent vector.
            logvar: Tensor of shape (batch_size, n_gaussians) representing the log-variance of the latent.

        """
        mu, logvar = self.encode(x)
        z = self.__reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar
