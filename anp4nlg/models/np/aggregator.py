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

    def forward(self, r_i: torch.Tensor, xtra: tuple =None) -> torch.Tensor:
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

    def forward(self, r_i: torch.Tensor, xtra: tuple =None) -> torch.Tensor:
        # TODO check dimension for mean aggregator
        print("Aggregator input shape:", r_i.shape)
        return torch.mean(r_i, dim=1)

class AttentionAggregator(Aggregator):
    def __init__(self, x_dim: int, r_dim: Union[int, tuple]):
        super().__init__(x_dim, r_dim)

        self.W_Q = nn.Linear(x_dim, ...)
        self.W_K = nn.Linear(x_dim, ...)
        self.W_V = nn.Linear(r_dim, ...)

    def forward(self, r_i: torch.Tensor, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        # TODO check dimension for mean aggregator
        K = x @ self.W_K 
        Q = x_target @ self.W_Q 
        V = r_i @ self.W_V

        scores = Q @ K.T
        weights = torch.softmax(scores / K.shape[1] ** 0.5, axis=1)
        return weights @ V