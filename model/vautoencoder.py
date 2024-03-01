import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, channels=[4, 8, 32],):
    """Initialize a time-dependent score-based network.
    Args:
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    # Encoding layers where the resolution decreases
    self.encoder = nn.Sequential(
                    nn.Conv2d(1, channels[0], 3, stride=1, bias=True, padding=1),
                    nn.BatchNorm2d(channels[0]),
                    nn.SiLU(),
                    nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=True, padding=1),
                    nn.BatchNorm2d(channels[1]),
                    nn.SiLU(),
                    nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=True, padding=1),
                    nn.BatchNorm2d(channels[2]),
                   ) #nn.SiLU(),
    self.decoder = nn.Sequential(
                    # nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=True, output_padding=1),
                    nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(channels[1]),
                    nn.SiLU(),
                    # nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=True, output_padding=1),
                    nn.ConvTranspose2d(channels[1], channels[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                    nn.BatchNorm2d(channels[0]),
                    nn.SiLU(),
                    # nn.ConvTranspose2d(channels[0], 1, 3, stride=1, bias=True),
                    nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.Sigmoid(),
                    )

  
  def forward(self, x):
    output = self.decoder(self.encoder(x))
    return output

  # Reparameterization Trick을 위한 함수
  def reparameterization(self, mu, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std
    
  def encoder(self, x):
    z = self.encoder(x)
    mu = self.mu(z)
    logvar = self.logvar(z)
    z = self.reparameterization(mu, logvar)
    return z, mu, logvar