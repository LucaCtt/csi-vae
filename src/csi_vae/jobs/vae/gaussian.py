import torch
import torch.nn.functional as func
from torch import nn

Conv1dLayerSpec = list[tuple[int, int]]
ConvLayerSpec = list[tuple[int, int, int, int]]
"""Specification for convolutional layers.

Each entry: (kernel_time, kernel_subcarrier, stride_time, stride_subcarrier)
"""

CONV_SPECS: list[ConvLayerSpec] = [
    [(5, 4, 5, 4), (5, 4, 5, 4), (3, 2, 3, 2)],
    [(10, 8, 10, 8)],
    [(5, 4, 5, 4), (5, 4, 5, 4)],
    [(5, 4, 5, 4), (5, 4, 5, 4), (3, 2, 3, 2), (3, 1, 1, 1)],
]


class _AntennaEncoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        channels: int,
        layers_spec: tuple[Conv1dLayerSpec, Conv1dLayerSpec],
    ) -> None:
        """Initialize the encoder for single-antenna CSI data.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The number of gaussians, lantent space will be n_gaussians*2 (mu and logvar).
            channels: The number of channels in the convolutional layers.
            layers_spec: A tuple containing two lists of tuples specifying the convolutional layers
                (kernel size and stride) for time and subcarrier convolutions, respectively.

        """
        super().__init__()
        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers

        self.__subcarrier_convs = nn.ModuleList()
        self.__time_convs = nn.ModuleList()

        in_channels = 1
        for (ks_t, st_t), (ks_s, st_s) in zip(layers_spec[0], layers_spec[1], strict=True):
            self.__time_convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, channels, kernel_size=ks_t, stride=st_t),
                    nn.GELU(),
                ),
            )
            self.__subcarrier_convs.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=ks_s, stride=st_s),
                    nn.GELU(),
                ),
            )
            in_channels = channels

        self.__flatten = nn.Flatten()

        _, flat_dim = self.get_shapes()
        self.__mu = nn.Linear(flat_dim, n_gaussians)
        self.__logvar = nn.Linear(flat_dim, n_gaussians)

    def __forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, window_size, n_subcarriers = x.shape

        for time_conv, sub_conv in zip(self.__time_convs, self.__subcarrier_convs, strict=True):
            # Time convolution
            x = x.permute(0, 3, 1, 2)  # (batch_size, n_subcarriers, channels, window_size)
            x = x.reshape(batch_size * n_subcarriers, channels, window_size)
            x = time_conv(x)
            _, channels, window_size = x.shape
            x = x.view(batch_size, n_subcarriers, channels, window_size).permute(0, 2, 3, 1)

            # Subcarrier convolution
            x = x.permute(0, 2, 1, 3)  # (batch_size, window_size, channels, n_subcarriers)
            x = x.reshape(batch_size * window_size, channels, n_subcarriers)
            x = sub_conv(x)
            _, channels, n_subcarriers = x.shape
            x = x.view(batch_size, window_size, channels, n_subcarriers).permute(0, 2, 1, 3)

        return x

    @torch.no_grad()
    def get_shapes(self) -> tuple[tuple, int]:
        """Get the shape of the latent features and the flattened dimension after convolutional layers."""
        x = torch.zeros(1, 1, self.__window_size, self.__n_subcarriers, device=next(self.parameters()).device)
        x = self.__forward_conv(x)
        latent_feat_shape = x.shape[1:]
        flat_dim = int(x.numel() // x.shape[0])
        return latent_feat_shape, flat_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder to produce mean and log-variance vectors.

        Arguments:
            x: Input tensor of shape (batch_size, window_size, n_subcarriers).

        Returns:
            mu: Tensor of shape (batch_size, n_gaussians) representing the mean of the latent vector.
            logvar: Tensor of shape (batch_size, n_gaussians) representing the log-variance of the latent vector.

        """
        x = x.unsqueeze(1)  # (B,1,T,S)
        z = self.__forward_conv(x)
        z = self.__flatten(z)
        return self.__mu(z), torch.clamp(self.__logvar(z), -10, 10)


class _AntennaDecoder(nn.Module):
    def __init__(
        self,
        latent_feat_shape: tuple,
        flat_dim: int,
        n_gaussians: int,
        channels: int,
        layers_spec: tuple[Conv1dLayerSpec, Conv1dLayerSpec],
    ) -> None:
        """Initialize the decoder for single-antenna CSI data.

        Arguments:
            latent_feat_shape: The shape of the features after the
                convolutional layers in the encoder (excluding batch dimension).
            flat_dim: The flattened dimension of the features after convolutional layers in the encoder.
            n_gaussians: The number of gaussians, latent space will be n_gaussians*2 (mu and logvar).
            channels: The number of channels in the convolutional layers.
            layers_spec: A tuple containing two lists of tuples specifying the convolutional layers
                (kernel size and stride) used in the encoder, which will be reversed for the decoder.

        Returns:
            A decoder module that takes a latent vector and reconstructs the input CSI data.

        """
        super().__init__()

        self.__latent_feat_shape = latent_feat_shape
        self.__fc = nn.Linear(n_gaussians, flat_dim)

        time_layers = list(reversed(layers_spec[0]))
        subcarrier_layers = list(reversed(layers_spec[1]))

        self.__time_deconvs = nn.ModuleList()
        self.__sub_deconvs = nn.ModuleList()

        for i, ((ks_s, st_s), (ks_t, st_t)) in enumerate(zip(subcarrier_layers, time_layers, strict=True)):
            out_ch = 1 if i == len(time_layers) - 1 else channels

            self.__sub_deconvs.append(
                nn.Sequential(
                    nn.ConvTranspose1d(channels, channels, ks_s, st_s),
                    nn.GELU(),
                ),
            )

            layers: list[nn.Module] = [nn.ConvTranspose1d(channels, out_ch, ks_t, st_t)]
            if i != len(time_layers) - 1:
                layers.append(nn.GELU())

            self.__time_deconvs.append(nn.Sequential(*layers))

    def __forward_deconv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, window_size, n_subcarriers = x.shape

        for sub_deconv, time_deconv in zip(self.__sub_deconvs, self.__time_deconvs, strict=True):
            # Subcarrier deconvolution
            x = x.permute(0, 2, 1, 3)  # (batch_size, window_size, channels, n_subcarriers)
            x = x.reshape(batch_size * window_size, channels, n_subcarriers)
            x = sub_deconv(x)
            _, channels, n_subcarriers = x.shape
            x = x.view(batch_size, window_size, channels, n_subcarriers).permute(0, 2, 1, 3)

            # Time deconvolution
            x = x.permute(0, 3, 1, 2)  # (batch_size, n_subcarriers, channels, window_size)
            x = x.reshape(batch_size * n_subcarriers, channels, window_size)
            x = time_deconv(x)
            _, channels, window_size = x.shape
            x = x.view(batch_size, n_subcarriers, channels, window_size).permute(0, 2, 3, 1)

        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = func.gelu(self.__fc(z))
        z = z.view(z.size(0), *self.__latent_feat_shape)
        x = self.__forward_deconv(z)
        return x.squeeze(1)


class SingleAntenna(nn.Module):
    """VAE architecture that encodes a single antenna's CSI data."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        channels: int,
        conv_layers: ConvLayerSpec,
    ) -> None:
        """Initialize the SingleAntennaVAE with an encoder and decoder for single-antenna CSI data.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The dimensionality of the latent space.
            channels: The number of channels in the convolutional layers.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()

        time_layers = [(ks_t, st_t) for ks_t, _, st_t, _ in conv_layers]
        subcarrier_layers = [(ks_s, st_s) for _, ks_s, _, st_s in conv_layers]

        self.__encoder = _AntennaEncoder(
            window_size,
            n_subcarriers,
            n_gaussians,
            channels,
            (time_layers, subcarrier_layers),
        )
        latent_feat_shape, flat_dim = self.__encoder.get_shapes()
        self.__decoder = _AntennaDecoder(
            latent_feat_shape,
            flat_dim,
            n_gaussians,
            channels,
            (time_layers, subcarrier_layers),
        )

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
            mu: Tensor of shape (batch_size, n_gaussians) representing the mean of the latent vector.
            logvar: Tensor of shape (batch_size, n_gaussians) representing the log-variance of the latent.

        """
        mu, logvar = self.encode(x)
        z = self.__reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar
