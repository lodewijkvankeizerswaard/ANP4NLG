import torch.nn as nn
import numpy as np


def context_target_split(x, y):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)
    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)
    num_context : int
        Number of context points.
    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    num_context = int(0.7 * num_points)
    num_extra_target = num_points - num_context
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_points,
                                 replace=False)
    x_context = x[:, locations[num_context:], :]
    y_context = y[:, locations[num_context:], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target

class ReshapeLast(nn.Module):
    """
    Helper layer to reshape the rightmost dimension of a tensor.

    This can be used as a component of nn.Sequential.
    """

    def __init__(self, shape: tuple):
        """
        shape: desired rightmost shape
        """
        super().__init__()
        self._shape = shape

    def forward(self, input):
        # reshapes the last dimension into self.shape
        return input.reshape(input.shape[:-1] + self._shape)