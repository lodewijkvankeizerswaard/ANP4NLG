import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

class LatentDistribution(nn.Module):
    """
    A wrapper object for any latent distribution
    Parameters
    ----------
    z_dim : int
        Dimension of latent variable z.
    r_dim : tuple
        Dimension of the parameter set.
    """
    def __init__(self, z_dim: int, r_dim: int):
        super(LatentDistribution, self).__init__()

        self.z_dim = z_dim
        self.r_dim = r_dim

    def forward(self, s: torch.Tensor) -> td.Distribution:
        """
        Return a torch.distribution object parameterized by h
        Parameters
        ----------
        s : torch.Tensor
            Shape (batch_size, h_dim)
        Returns
        -------
        p(z) : td.Distribution
            The latent distribution p(z)
        """
        raise NotImplementedError("Abstract method.")

class NormalLatentDistribution(LatentDistribution):
    def __init__(self, z_dim: int, r_dim: tuple):
        super().__init__(z_dim, r_dim)

    def forward(self, s: torch.Tensor) -> td.Distribution:
        mu = s[..., 0]
        sigma = 0.1 + 0.9 * torch.sigmoid(s[..., 1])

        return td.Normal(loc=mu, scale=sigma) # we have two output axes: B x Z_dim

class StdNormalLatentDistribution(LatentDistribution):
    def __init__(self, z_dim: int, r_dim: tuple):
        super().__init__(z_dim, r_dim)

    def forward(self, s: torch.Tensor) -> td.Distribution:
        # TODO check if we need to use registered buffers for Latent distributions
        mu = s[..., 0]
        sigma = F.softplus(s[..., 1])

        return td.Independent(td.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(sigma)), 0) # we have two output axes: B x Z_dim

