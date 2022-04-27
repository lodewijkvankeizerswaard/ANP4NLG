
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.distributions.kl import kl_divergence

from random import randint
from collections import defaultdict, OrderedDict

from util import context_target_split
from neural_process import NeuralProcess
from tqdm.auto import tqdm

def np_loss(p_y_pred, y_target, q_target, q_context):
    """
    Computes Neural Process loss.

    Parameters
    ----------
    p_y_pred : one of torch.distributions.Distribution
        Distribution over y output by Neural Process.

    y_target : torch.Tensor
        Shape (batch_size, num_target, y_dim)

    q_target : one of torch.distributions.Distribution
        Latent distribution for target points.

    q_context : one of torch.distributions.Distribution
        Latent distribution for context points.
    """
    # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
    # over batch and sum over number of targets and dimensions of y
    log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
    # KL has shape (batch_size, r_dim). Take mean over batch and sum over
    # r_dim (since r_dim is dimension of normal distribution)
    kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
    return -log_likelihood + kl

def train(device: torch.device, neural_process: NeuralProcess, optimizer: opt.Optimizer, data_loader: DataLoader, epochs: int,
            num_context_range: tuple, num_extra_target_range: tuple):
    """
    Trains Neural Process.

    Parameters
    ----------
    device : torch.device
        The device to train on.

    neural_process : neural_process.NeuralProcess
        The neural process to train.

    optimizer : torch.optim.Optimizer 
        The optimizer.

    dataloader : torch.utils.DataLoader 
        The dataloader object.

    epochs : int
            Number of epochs to train for.

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.
    """
    total_steps = epochs * len(data_loader)
    step = 0
    log = defaultdict(list)

    with tqdm(range(total_steps)) as bar:
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, (x,y) in enumerate(data_loader):
                neural_process.train()
                optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*num_context_range)
                num_extra_target = randint(*num_extra_target_range)

                # Create context and target points and apply neural process
                x_context, y_context, x_target, y_target = \
                    context_target_split(x, y, num_context, num_extra_target)
                p_y_pred, q_target, q_context = \
                    neural_process(x_context, y_context, x_target, y_target)

                loss = np_loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                optimizer.step()

                log["training.loss"].append((step, loss))

                bar_dict = OrderedDict()
                bar_dict['loss'] = f"{loss.item():.2f}"
                # bar_dict[f"dev.L"] =  "{:.2f}".format(log[f"dev.L"][-1][1])
                bar.set_postfix(bar_dict)
                bar.update()

                step += 1

    return log

