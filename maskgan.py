import torch
import torch.nn as nn

import torch.distributions as td

from fairseq.models.lstm import LSTMEncoder, \
                                LSTMDecoder, \
                                LSTMModel,\
                                base_architecture, \
                                Embedding
from fairseq import options, utils

from maskgan_utils import perplexity, greedy_sample

#### MGAN ####
class LossModel(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion


class MGANModel(nn.Module):
    def __init__(self, generator, discriminator, critic=None, pretrain=False):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.critic = critic
        self.pretrain = pretrain

    @classmethod
    def build_model(cls, args, task, pretrain):
        # Build critic
        critic = MGANCritic.build_model(args, task)
        mse_loss = WeightedMSELoss()
        closs = LossModel(critic, mse_loss)

        # Build generator
        if pretrain:
            generator = MLEGenerator.build_model(args,task)
            gcriterion = TCELoss()
        else:
            generator = MGANGenerator.build_model(args, task)
            reinforce = REINFORCE(gamma=0.01, clip_value=5.0)
            gcriterion = reinforce

        gloss = LossModel(generator, gcriterion)

        # Build discriminator
        discriminator = MGANDiscriminator.build_model(args, task)
        tceloss = TBCELoss()
        dloss = LossModel(discriminator, tceloss)

        return cls(gloss, dloss, closs, pretrain=pretrain)

    def forward(self, *args, **kwargs):
        if 'ppl' not in kwargs:
            kwargs['ppl'] = False

        if kwargs['tag'] == 'g-step':
            if self.pretrain:
                return self._gstep_pretrain(*args, ppl_compute=kwargs['ppl'])
            else:
                return self._gstep(*args, ppl_compute=kwargs['ppl'])
        elif kwargs['tag'] == 'c-step':
            return self._cstep(*args)

        return self._dstep(*args, real=kwargs['real'])

    def _cstep(self, masked, lengths, mask, unmasked):
        with torch.no_grad():
            samples, log_probs, attns = self.generator.model(masked, lengths, unmasked, mask)
            logits, attn_scores = self.discriminator.model(masked, lengths, samples)

        baselines, _ = self.critic.model(masked, lengths, samples)
        with torch.no_grad():
            reward, cumulative_rewards = self.generator.criterion(log_probs, logits, mask, baselines)

        critic_loss = self.critic.criterion(baselines.squeeze(2), cumulative_rewards, mask)
        return critic_loss

    def _gstep(self, masked, lengths, mask, unmasked, ppl_compute=False):
        samples, log_probs, attns = self.generator.model(masked, lengths, unmasked, mask)
        
        # discriminattor
        with torch.no_grad():
            logits, attn_scores = self.discriminator.model(masked, lengths, samples)
            baselines, _ = self.critic.model(masked, lengths, samples)

        reward, cumulative_rewards = self.generator.criterion(log_probs, logits, mask, baselines.detach())
        loss = -1*reward

        # Compute perplexity
        if ppl_compute:
            with torch.no_grad():
                logits = self.generator.model.logits(masked, lengths, unmasked, mask).clone()
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                ppl = perplexity(unmasked, samples, log_probs)
        else:
            ppl = None

        return (loss, samples, ppl)
    

    def _gstep_pretrain(self, masked, lengths, mask, unmasked, ppl_compute=False):
        logits, attns = self.generator.model(masked, lengths, unmasked)
        samples = greedy_sample(logits)
        loss = self.generator.criterion(logits, unmasked)
        if ppl_compute:
            with torch.no_grad():
                log_probs = torch.nn.functional.log_softmax(logits, dim=2).clone()
                ppl = perplexity(unmasked, samples, log_probs)
        else:
            ppl = None
        return (loss, samples, ppl)

    def _dstep(self, masked, lengths, mask, unmasked, real=True):
        logits, attn_scores = self.discriminator.model(masked, lengths, unmasked)
        mask = mask.unsqueeze(2)
        truths = torch.ones_like(logits) if real else torch.ones_like(logits) - mask
        loss = self.discriminator.criterion(logits, truths, weight=mask)
        return loss

#### MLE GENERATOR ####
class MLEGenerator(LSTMModel):
    def forward(self, masked, lengths, unmasked):
        self.encoder.lstm.flatten_parameters()
        return super().forward(masked, lengths, unmasked)

    def logits(self, masked, lengths, unmasked, mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(masked, lengths, unmasked)
        return logits


#### GENERATOR ####

class MGANGEncoder(LSTMEncoder): pass
class MGANGDecoder(LSTMDecoder): pass
class MGANGenerator(LSTMModel):
    def forward(self, masked, lengths, unmasked, mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(masked, lengths, unmasked)
        bsz, seqlen, vocab_size = logits.size()

        # Sample from x converting it to probabilities
        samples = []
        log_probs = []
        for t in range(seqlen):
            logit = logits[:, t, :]
            distribution = td.Categorical(logits=logit)
            sampled = distribution.sample()
            fsampled = torch.where(mask[:, t].byte(), sampled, unmasked[:, t])
            log_prob = distribution.log_prob(fsampled)
            # flog_prob = torch.where(mask[:, t].byte(), log_prob, torch.zeros_like(log_prob))
            log_probs.append(log_prob)
            samples.append(fsampled)

        samples = torch.stack(samples, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return (samples, log_probs, attns)

    def logits(self, masked, lengths, unmasked, mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(masked, lengths, unmasked)
        return logits
    
class MLEGenerator(LSTMModel):
    def forward(self, masked, lengths, unmasked):
        self.encoder.lstm.flatten_parameters()
        return super().forward(masked, lengths, unmasked)

    def logits(self, masked, lengths, unmasked, mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(masked, lengths, unmasked)
        return logits

#### CRITIC ####
class MGANCriticEncoder(LSTMEncoder):
    pass

class MGANCriticDecoder(LSTMDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_embed_dim = self.additional_fc.out_features if hasattr(self, "additional_fc") else self.hidden_size
        self.fc_out = nn.Linear(out_embed_dim, 1)

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        x, attn_scores = super().forward(prev_output_tokens, encoder_out_dict, incremental_state)
        return x, attn_scores


class MGANCritic(LSTMModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise RuntimeError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise RuntimeError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        encoder = MGANCriticEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = MGANCriticDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(encoder, decoder)

    def forward(self, *args, **kwargs):
        self.encoder.lstm.flatten_parameters()
        return super().forward(*args, **kwargs)




#### DISCRIMINATOR ####
class MGANDiscriminatorEncoder(LSTMEncoder):
    pass

class MGANDiscriminatorDecoder(LSTMDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_embed_dim = self.additional_fc.out_features if hasattr(self, "additional_fc") else self.hidden_size
        self.fc_out = nn.Linear(out_embed_dim, 1)

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        x, attn_scores = super().forward(prev_output_tokens, encoder_out_dict, incremental_state)
        # Do not apply sigmoid, numerically unstable while training.
        # Get logits and use BCEWithLogitsLoss() instead.
        # x = torch.sigmoid(x)
        return x, attn_scores


class MGANDiscriminator(LSTMModel):
    # Not great DRY, but works.
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise RuntimeError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise RuntimeError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        encoder = MGANDiscriminatorEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = MGANDiscriminatorDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(encoder, decoder)

    def forward(self, *args, **kwargs):
        self.encoder.lstm.flatten_parameters()
        return super().forward(*args, **kwargs)



class MaskGAN(nn.Module):
    def __init__(self, args):
        super().__init__()

    
