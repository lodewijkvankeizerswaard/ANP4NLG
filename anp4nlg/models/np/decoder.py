import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from fairseq.models import FairseqDecoder

class Decoder(FairseqDecoder):
    """ ## ADAPTED FROM https://github.com/EmilienDupont/neural-processes/ ##
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    r_dim : int
        Dimension of representation r.
    z_dim : int
        Dimension of latent variable z.
    y_dim : int
        Dimension of y values.
    """
    def __init__(self, dictionary, x_dim: int, r_dim: int, z_dim: int, y_dim: int):
        super(Decoder, self).__init__(dictionary)

        self.x_dim = x_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.y_dim = y_dim

    def forward(self, x_target: torch.Tensor, r_c: torch.Tensor, z: torch.Tensor) -> tuple:
        """
        x_target : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        r_c : torch.Tensor
            Shape (batch_size, r_dim)
        z : torch.Tensor
            Shape (batch_size, z_dim)
        Returns
        -------
        Returns parameters for output distribution. Both have shape
        (batch_size, num_points, y_dim)."""
        # TODO fix output shape in docstring
        # TODO decide on y distribution

        raise NotImplementedError("Abstract method.")

class MLPDecoder(Decoder):
    def __init__(self, dictionary, x_dim: int, r_dim: int, z_dim: int, y_dim: int, h_dim: int):
        super().__init__(dictionary, x_dim, r_dim, z_dim, y_dim)
        layers = [nn.Linear(x_dim + z_dim + r_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xrz_to_h = nn.Sequential(*layers)
        self.h_to_dict = nn.Linear(h_dim, len(dictionary))

    def forward(self, x_target: torch.Tensor, r_c: torch.Tensor, z: torch.Tensor) -> tuple:
        batch_size, num_points, _ = x_target.size()
        # Repeat r_c, so it can be concatenated with every x. This changes shape
        # from (batch_size, r_dim, 1) to (batch_size, num_points, r_dim)
        r_c = r_c.unsqueeze(1).squeeze(-1).repeat(1, num_points, 1)
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)

        # print("decoder.x", x_target.shape)
        # print("decoder.r_c", r_c.shape)
        # print("decoder.z", z.shape)

        # Concatenate x_target, r_c, and z and pass them through the mlp
        inp = torch.cat((x_target, r_c, z), dim=-1)
        out = self.xrz_to_h(inp)

        # TODO make output distr. modular
        logits = self.h_to_dict(out)

        # return mu, sigma
        return logits