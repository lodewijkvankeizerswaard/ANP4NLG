import torch
import torch.nn as nn
import numpy as np

from typing import Union
from .util import ReshapeLast

class Aggregator(nn.Module):
    """
    Aggregates representations r_i or paramters s_i into into a single
    representation or parameter set.
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    r_dim : Union[int, tuple]
        Dimension of representation r or parameter set s.
    """
    def __init__(self, x_dim: int, r_dim: Union[int, tuple]):
        super(Aggregator, self).__init__()

        self.x_dim = x_dim
        self.r_dim = r_dim if isinstance(r_dim, tuple) else (r_dim, 1)

    def forward(self, r_i: torch.Tensor, x_context: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Aggregates representations r_i or paramters s_i into into a single
        representation or parameter set.
        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        xtra : tuple
            Any extra objects used for aggregation
        """
        raise NotImplementedError("Abstract method.")

class MeanAggregator(Aggregator):
    def __init__(self, x_dim: int, rs_dim: Union[int, tuple]):
        super().__init__(x_dim, rs_dim)

    def forward(self, rs_i: torch.Tensor, x_context: torch.Tensor=None, x_target: torch.Tensor=None) -> torch.Tensor:
        # TODO check dimension for mean aggregator
        return torch.mean(rs_i, dim=1)

class AttentionAggregator(Aggregator):
    def __init__(self, x_dim: int, r_dim: Union[int, tuple], h_dim):
        super().__init__(x_dim, r_dim)

        output_shape = r_dim
        output_size = np.prod(output_shape)

        layers = [nn.Linear(x_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, output_size)]

        self.batch_mlp = nn.Sequential(*layers)

        self.attn = nn.MultiheadAttention(output_size, 1, batch_first=True)

    def forward(self, r_i: torch.Tensor, x_context: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        # TODO check dimension for mean aggregator
        
        k = self.batch_mlp(x_context)
        q = self.batch_mlp(x_target)
        v = r_i.squeeze(-1)

        print("Query (x_target) shape :", q.shape)
        print("Key (x_context), Value (r_i) shapes", k.shape, v.shape)
        return self.attn(q, k, v)[0]