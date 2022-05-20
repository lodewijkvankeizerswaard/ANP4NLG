import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel, register_model

from .aggregator import Aggregator, MeanAggregator, AttentionAggregator
from .decoder import Decoder, MLPDecoder
from .encoder import Encoder, MLPEncoder, AttentionEncoder
from .latent_distribution import LatentDistribution, NormalLatentDistribution
from.util import context_target_split


@register_model('neural_process')
class NeuralProcess(BaseFairseqModel):
    """ ## FROM https://github.com/EmilienDupont/neural-processes/ ##
    Implements Neural Process for functions of arbitrary dimensions.
    Parameters
    ----------
    positional_embedder: nn.Module
        The module that embeds the word positions; returns x values
    word_embedder: nn.Module
        The module that embeds the word meaning; returns y values
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
    def __init__(self, positional_embedder: nn.Module, word_embedder: nn.Module,
                       deterministic_encoder: Encoder, deterministic_aggregator: Aggregator, 
                       latent_encoder: Encoder, latent_aggregator: Aggregator, 
                       latent_distribution: LatentDistribution, decoder: Decoder):
        super(NeuralProcess, self).__init__()
        self.positional_embedder = positional_embedder
        self.word_embedder = word_embedder
        self.deterministic_encoder = deterministic_encoder
        self.deterministic_aggregator = deterministic_aggregator
        self.latent_encoder = latent_encoder
        self.latent_aggregator = latent_aggregator
        self.latent_distribution = latent_distribution
        self.decoder = decoder

    @property
    def supported_targets(self):
        return {"self"}

    @staticmethod
    def add_args(parser):
        """ TODO add text"""
        parser.add_argument('--positional-embedding', type=str, choices=['scalar'], default='scalar', 
                            help="the type of positonal embedding")
        parser.add_argument('--positional-embedding-len', type=int, default=-1, 
                            help="the maximum sentence length to divide the positional embedding over;"
                            " -1 use maximum sentence length per batch (default: -1)")
        parser.add_argument('--word_embedding_dim', type=int, default=300,
                            help="the dimensionality of the word embeddings")

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        BATCH_SIZE = 32
        X_DIM = 1
        Y_DIM = args.word_embedding_dim
        N = 4
        M = 2
        R_DIM = 20
        S_DIM = (20, 2)
        H_DIM = 20
        Z_DIM = 20

        model = NeuralProcess(
            PositionalEmbedding(args.positional_embedding, max_len=args.positional_embedding_len),
            nn.Embedding(
                num_embeddings=len(task.dictionary),
                embedding_dim=Y_DIM,
                padding_idx=task.dictionary.pad(),
            ),
            MLPEncoder(X_DIM, Y_DIM, R_DIM, H_DIM),
            MeanAggregator(X_DIM, R_DIM),
            MLPEncoder(X_DIM, Y_DIM, S_DIM, H_DIM),
            MeanAggregator(X_DIM, S_DIM),
            NormalLatentDistribution(Z_DIM, S_DIM),
            MLPDecoder(task.target_dictionary, X_DIM, R_DIM, Z_DIM, Y_DIM, H_DIM)
        )

        print(model)

        return model

    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor) -> tuple:
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

        x = self.positional_embedder(src_lengths)
        y = self.word_embedder(src_tokens)

        print('X SHAPE', x.shape)
        print('Y SHAPE', y.shape)

        x_context, y_context, x_target, y_target = context_target_split(x, y)
        # x_context.to(self.device)
        # y_context.to(self.device)
        # x_target.to(self.device)
        # y_target.to(self.device)

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

            s_i_target = self.latent_encoder(x_target, y_target)
            s_target = self.latent_aggregator(s_i_target)
            q_target = self.latent_distribution(s_target)

            # Sample z
            z_context = q_context.sample()

            # Decode 
            p_y_pred = self.decoder(x_target, r_context, z_context)

            return p_y_pred, q_target, q_context

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device

class PositionalEmbedding(nn.Module):
    """Given a tensor of sample sequence lengths (e.g. sentence lengths), create a 
    positional embedding for every word, and return these (x-values). Options are:
    - scalar: scalar between 0 (bos) and 1 (eos).
    """
    def __init__(self, embedding: str, max_len: int=-1):
        """
        encoding: desired positional embedding
        """
        super(PositionalEmbedding, self).__init__()
        if embedding == "scalar":
            self.fn = self._scalar_encoding
        else:
            print("[warning] \"{}\" not implemented, defaulting to scalar embedding!".format(embedding))
            self.fn = self._scalar_encoding

        # We need this to be on the same device as the tensors, but we do not want it to be a parameter
        self.register_buffer('max_len', torch.Tensor([max_len]), persistent=True)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.fn(input)

    def _scalar_encoding(self, y: torch.Tensor) -> torch.Tensor:
        batch_size = y.shape[0]
        max_len = y.max()
        x = torch.arange(max_len).repeat(batch_size, 1).unsqueeze(-1).to(self.device)

        if self.max_len != -1:
            return x / self.max_len
        else:
            return x / max_len

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return self.max_len.device

@register_model('attentive_neural_process')
class AttentiveNeuralProcess(NeuralProcess):
    """ ## FROM https://github.com/EmilienDupont/neural-processes/ ##
    Implements Attentative Neural Process for functions of arbitrary dimensions.
    Parameters
    ----------
    positional_embedder: nn.Module
        The module that embeds the word positions; returns x values
    word_embedder: nn.Module
        The module that embeds the word meaning; returns y values
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
    def __init__(self, *args, **kwargs):
        super(AttentiveNeuralProcess, self).__init__(*args, **kwargs)

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        BATCH_SIZE = 32
        X_DIM = 1
        Y_DIM = args.word_embedding_dim
        N = 4
        M = 2
        R_DIM = 20
        S_DIM = (20, 2)
        H_DIM = 20
        Z_DIM = 20

        model = NeuralProcess(
            PositionalEmbedding(args.positional_embedding, max_len=args.positional_embedding_len),
            nn.Embedding(
                num_embeddings=len(task.dictionary),
                embedding_dim=Y_DIM,
                padding_idx=task.dictionary.pad(),
            ),
            AttentionEncoder(X_DIM, Y_DIM, R_DIM, H_DIM),
            AttentionAggregator(X_DIM, R_DIM),
            AttentionEncoder(X_DIM, Y_DIM, S_DIM, H_DIM),
            AttentionAggregator(X_DIM, S_DIM),
            NormalLatentDistribution(Z_DIM, S_DIM),
            MLPDecoder(task.target_dictionary, X_DIM, R_DIM, Z_DIM, Y_DIM, H_DIM)
        )

        print(model)

        return model

    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor) -> tuple:
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

        x = self.positional_embedder(src_lengths)
        y = self.word_embedder(src_tokens)
        x_context, y_context, x_target, y_target = context_target_split(x, y)
        # x_context.to(self.device)
        # y_context.to(self.device)
        # x_target.to(self.device)
        # y_target.to(self.device)

        if self.training:
            # Encode context via deterministic and latent path
            r_i = self.deterministic_encoder(x_context, y_context)
            s_i_context = self.latent_encoder(x_context, y_context)

            # Construct context vector and latent context distribution
            r_context = self.deterministic_aggregator(r_i, x_context, x_target)
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

            s_i_target = self.latent_encoder(x_target, y_target)
            s_target = self.latent_aggregator(s_i_target)
            q_target = self.latent_distribution(s_target)

            # Sample z
            z_context = q_context.sample()

            # Decode 
            p_y_pred = self.decoder(x_target, r_context, z_context)

            return p_y_pred, q_target, q_context