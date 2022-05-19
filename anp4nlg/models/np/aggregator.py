from os import W_OK
import torch
import torch.nn as nn

from typing import Union

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

    def forward(self, r_i: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, x_dim: int, r_dim: Union[int, tuple]):
        super().__init__(x_dim, r_dim)

    def forward(self, r_i: torch.Tensor) -> torch.Tensor:
        # TODO check dimension for mean aggregator
        print("Aggregator input shape:", r_i.shape)
        print("Aggregator output shape:", torch.mean(r_i, dim=1).shape)
        return torch.mean(r_i, dim=1)

class AttentionAggregator(Aggregator):
    def __init__(self, x_dim: int, r_dim: Union[int, tuple]):
        super().__init__(x_dim, r_dim)

        self.W_Q = nn.Linear(x_dim, x_dim)
        self.W_K = nn.Linear(x_dim, x_dim)
        self.W_V = [nn.Linear(self.r_dim[0], self.r_dim[0]) for _ in range(self.r_dim[1])]

    def forward(self, r_i: torch.Tensor, x_context: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        # TODO check dimension for mean aggregator
        print(r_i.shape, x_context.shape, x_target.shape)
        K = self.W_K(x_context)
        Q = self.W_Q(x_target)
        V = self.W_V(r_i)

        scores = Q @ K.T
        weights = torch.softmax(scores / K.shape[1] ** 0.5, axis=1)
        return weights @ V