from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (FairseqIncrementalDecoder, FairseqLanguageModel,
                            register_model)
from fairseq.models.fairseq_encoder import EncoderOut


class MLPEncoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=300, rs_dim=(20, 1), h_dim=20):
        super().__init__()
        self.output_shape = rs_dim
        output_size = np.prod(self.output_shape)

        self.input_to_rs = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, output_size),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input = torch.cat((x,y), dim=2)
        input = self.input_to_rs(input)

        return input.reshape(input.shape[:-1] + self.output_shape)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        dictionary,
        x_dim: int = 1,
        r_dim: int = 20,
        z_dim: int = 20,
        h_dim: int = 20
    ):
        super().__init__()

        self.xrz_to_h = nn.Sequential(
            nn.Linear(x_dim + z_dim + r_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
        )
        self.h_to_dict = nn.Linear(h_dim, len(dictionary))

    def forward(self, x_target: torch.Tensor, r_c: torch.Tensor, z: torch.Tensor) -> torch.distributions.Distribution:
        num_points = x_target.size()[1]
        r_c = r_c.unsqueeze(1).squeeze(-1).repeat(1, num_points, 1)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        inp = torch.cat((x_target, r_c, z), dim=-1)
        out = self.xrz_to_h(inp)
        logits = self.h_to_dict(out)

        return torch.distributions.Categorical(logits=logits)


class NeuralProcessDecoder(FairseqIncrementalDecoder):
    def __init__(self, dictionary, embedding_dim=300):
        super().__init__(dictionary)

        self.dummy_fc = nn.Linear(128, 128)

        # Layer for gneerating word embeddings
        self.embedding = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embedding_dim,
            padding_idx=dictionary.pad(),
        )

        # Deterministic and latent encoders
        self.deterministic_encoder = MLPEncoder()
        self.latent_encoder = MLPEncoder(rs_dim=(20, 2))

        # Decoder
        self.decoder = MLPDecoder(dictionary)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        x = self._encode_positions(prev_output_tokens)
        y = self.embedding(prev_output_tokens)

        x_context, y_context, x_target, y_target = self._context_target_split(x, y)

        # Encode context via deterministic and latent path
        r_i = self.deterministic_encoder(x_context, y_context)
        s_i_context = self.latent_encoder(x_context, y_context)

        # Construct context vector and latent context distribution
        r_context = torch.mean(r_i, dim=1)
        s_context = torch.mean(s_i_context, dim=1)
        q_context = self._normal_latent_distribution(s_context)

        # Encode targets and construct latent target distribution
        s_i_target = self.latent_encoder(x_target, y_target)
        s_target = torch.mean(s_i_target, dim=1)
        q_target = self._normal_latent_distribution(s_target)

        # Sample z
        z_context = q_context.sample()

        # Decode 
        p_y_pred = self.decoder(x_target, r_context, z_context)

        return p_y_pred, q_target, q_context

    def _encode_positions(self, tokens):
        batch_size = tokens.shape[0]
        max_len = tokens.shape[1]
        x = torch.arange(max_len).repeat(batch_size, 1).unsqueeze(-1).to(tokens.device)

        return x / max_len

    def _context_target_split(self, x, y):
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
        num_context = int(0.5 * num_points)
        num_extra_target = int(0.5 * num_points)
        # Sample locations of context and target points
        locations = np.random.choice(num_points,
                                     size=num_context + num_extra_target,
                                     replace=False)
        x_context = x[:, locations[:num_context], :]
        y_context = y[:, locations[:num_context], :]
        x_target = x[:, locations, :]
        y_target = y[:, locations, :]
        return x_context, y_context, x_target, y_target

    def _normal_latent_distribution(self, s: torch.Tensor) -> torch.distributions.Distribution:
        mu = s[..., 0]
        sigma = F.softplus(s[..., 1])
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mu, scale=sigma), 2) # we have two output axes: B x Z_dim


@register_model('neural_process')
class NeuralProcess(FairseqLanguageModel):
    @classmethod
    def build_model(cls, args, task):
        decoder = NeuralProcessDecoder(task.target_dictionary)
        return cls(decoder)

    @property
    def supported_targets(self):
        return {"self"}
