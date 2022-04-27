import torch
import torch.nn as nn

from encoder import Encoder
from aggregator import Aggregator
from latent_distribution import LatentDistribution
from decoder import Decoder



class NeuralProcess(nn.Module):
    """ ## FROM https://github.com/EmilienDupont/neural-processes/ ##
    Implements Neural Process for functions of arbitrary dimensions.
    Parameters
    ----------
    deterministic_encoder : Encoder
        The encoder for the deterministic path; computes representations r_i
    deterministic_aggregator: Aggregator
        The aggregator for the deterministic path; computes r_c or r_star
    latent_encoder : Encoder
        The encoder for the latent path; computes s_i
    latent_aggregator : Aggregator
        The aggregator for the latent path; computes s
    latent_distribution : LatentDistribution
        The latent distribution parameterized by s
    decoder : Decoder
        The decoder; computes y_star
    """
    # TODO decide on y_distribution
    def __init__(self, deterministic_encoder: Encoder, deterministic_aggregator: Aggregator, 
                        latent_encoder: Encoder, latent_aggregator: Aggregator, 
                        latent_distribution: LatentDistribution, decoder: Decoder):
        super(NeuralProcess, self).__init__()
        self.deterministic_encoder = deterministic_encoder
        self.deterministic_aggregator = deterministic_aggregator
        self.latent_encoder = latent_encoder
        self.latent_aggregator = latent_aggregator
        self.latent_distribution = latent_distribution
        self.decoder = decoder

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor, 
                        x_target: torch.Tensor, y_target: torch.Tensor =None) -> tuple:
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        Returns
        -------

        """
        # TODO write return statement for neural process forward
        # Infer quantities from tensor dimensions
        # batch_size, num_context, x_dim = x_context.size()
        # _, num_target, _ = x_target.size()
        # _, _, y_dim = y_context.size()

        if self.training:
            # Encode context via deterministic and latent path
            r_i = self.deterministic_encoder(x_context, y_context)
            s_i_context = self.latent_encoder(x_context, y_context)

            # Construct context vector and latent context distribution
            r_context = self.deterministic_aggregator(r_i)
            s_context = self.latent_aggregator(s_i_context)
            q_context = self.latent_distribution(s_context)

            # Encode targets and construct latent target distribution
            s_i_target = self.latent_encoder(x_target, y_target)
            s_target = self.latent_aggregator(s_i_target)
            q_target = self.latent_distribution(s_target)

            # Sample z
            z_context = q_context.sample()

            # Decode 
            p_y_pred = self.decoder(x_target, r_context, z_context)

            return p_y_pred, q_target, q_context
        else:
            # Encode context via deterministic and latent path
            r_i = self.deterministic_encoder(x_context, y_context)
            s_i_context = self.latent_encoder(x_context, y_context)

            # Construct context vector and latent context distribution
            r_context = self.deterministic_aggregator(r_i)
            s_context = self.latent_aggregator(s_i_context)
            q_context = self.latent_distribution(s_context)

            # Sample z
            z_context = q_context.sample()

            # Decode 
            p_y_pred = self.decoder(x_target, r_context, z_context)

            return p_y_pred