import torch
import torch.nn.functional as func
from torch import nn

ConvLayerSpec = list[tuple[int, int, int, int]]
"""Specification for convolutional layers.

Each entry: (kernel_h, kernel_w, stride_h, stride_w)
"""

CONV_SPECS: list[ConvLayerSpec] = [
    [(5, 8, 5, 8), (5, 4, 5, 4), (3, 2, 3, 2)],
    [(10, 8, 10, 8)],
    [(5, 8, 5, 8), (5, 4, 5, 4)],
    [(5, 8, 5, 8), (5, 4, 5, 4), (3, 2, 3, 2), (3, 1, 1, 1)],
]


class _AntennaEncoder(nn.Module):
    """Encode a single-antenna CSI window into mean and log-variance vectors."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        latent_dim: int,
        channels: int,
        conv_layers: ConvLayerSpec,
    ) -> None:
        """Initialize the AntennaEncoder with convolutional layers and linear heads.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            latent_dim: The dimensionality of the latent space.
            channels: The number of channels in the convolutional layers.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()
        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers

        layers: list[nn.Module] = []
        in_ch = 1
        for kh, kw, sh, sw in conv_layers:
            layers.append(nn.Conv2d(in_ch, channels, kernel_size=(kh, kw), stride=(sh, sw)))
            layers.append(nn.GELU())
            in_ch = channels

        layers.append(nn.Flatten())
        self.__conv = nn.Sequential(*layers)

        # Infer flattened feature dimension for linear heads
        _, flat_dim = self.get_shapes()

        # Linear heads for Gaussian parameters
        self.__mu = nn.Linear(flat_dim, latent_dim)
        self.__logvar = nn.Linear(flat_dim, latent_dim)

    @torch.no_grad()
    def get_shapes(self) -> tuple[tuple, int]:
        """Return the latent feature map shape and its flattened size.

        Returns:
            latent_feat_shape: The shape of the feature map after convolution (Channels, H, W).
            flat_dim: The total number of features when the feature map is flattened.

        """
        x = torch.zeros(1, 1, self.__window_size, self.__n_subcarriers, device=next(self.parameters()).device)
        x = self.__conv[:-1](x)
        latent_feat_shape = x.shape[1:]
        flat_dim = int(x.numel() // x.shape[0])

        return latent_feat_shape, flat_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log-variance for a single-antenna input.

        Arguments:
            x: Input tensor of shape (batch_size, window_size, n_subcarriers) for one antenna.

        Returns:
            mu: Tensor of shape (batch_size, antenna_latent_dim) representing the mean of the latent distribution.
            logvar: Tensor of shape (batch_size, antenna_latent_dim)

        """
        x = x.unsqueeze(1)  # Add channel dimension
        z = self.__conv(x)  # No need to squeeze since the conv output is already flattened
        return self.__mu(z), torch.clamp(self.__logvar(z), min=-10, max=10)


class _AntennaDecoder(nn.Module):
    """Decode a latent vector back into a CSI window for a single antenna."""

    def __init__(
        self,
        latent_feat_shape: tuple,
        flat_dim: int,
        latent_dim: int,
        channels: int,
        conv_layers: ConvLayerSpec,
    ) -> None:
        """Initialize the AntennaDecoder with linear and deconvolutional layers.

        Arguments:
            latent_feat_shape: The shape of the feature map before flattening in the encoder (Channels, H, W).
            flat_dim: The total number of features when the feature map is flattened.
            latent_dim: The dimensionality of the latent space.
            channels: The number of channels in the convolutional layers.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride)

        """
        super().__init__()

        self.__latent_feat_shape = latent_feat_shape
        self.__fc = nn.Linear(latent_dim, flat_dim)

        deconv_layers: list[nn.Module] = []
        reversed_specs = list(reversed(conv_layers))

        for i, (kh, kw, sh, sw) in enumerate(reversed_specs):
            out_ch = 1 if i == len(reversed_specs) - 1 else channels
            deconv_layers.append(nn.ConvTranspose2d(channels, out_ch, kernel_size=(kh, kw), stride=(sh, sw)))
            if i < len(reversed_specs) - 1:
                deconv_layers.append(nn.GELU())

        self.__deconv = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector into a CSI window.

        Arguments:
            z: Input tensor of shape (batch_size, antenna_latent_dim) representing the latent vector for one antenna.

        Returns:
            recon: Tensor of shape (batch_size, window_size, n_subcarriers)
                   representing the reconstructed CSI window for one antenna.

        """
        z = func.gelu(self.__fc(z))
        z = z.view(z.size(0), *self.__latent_feat_shape)
        return self.__deconv(z).squeeze(1)  # Remove channel dimension


class SingleAntenna(nn.Module):
    """VAE architecture that encodes a single antenna's CSI data."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        latent_dim: int,
        channels: int,
        conv_layers: ConvLayerSpec,
    ) -> None:
        """Initialize the SingleAntennaVAE with an encoder and decoder for single-antenna CSI data.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            latent_dim: The dimensionality of the latent space.
            channels: The number of channels in the convolutional layers.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()

        self.__encoder = _AntennaEncoder(window_size, n_subcarriers, latent_dim, channels, conv_layers)
        latent_feat_shape, flat_dim = self.__encoder.get_shapes()
        self.__decoder = _AntennaDecoder(latent_feat_shape, flat_dim, latent_dim, channels, conv_layers)

    def __reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the Gaussian distribution defined by mu and logvar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
            mu: Tensor of shape (batch_size, latent_dim) representing the mean of the latent vector.
            logvar: Tensor of shape (batch_size, latent_dim) representing the log-variance of the latent.

        """
        mu, logvar = self.encode(x)
        z = self.__reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar
