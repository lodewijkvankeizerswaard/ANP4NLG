#!/bin/bash

fairseq-train data-bin/wikitext-103 \
    --latent_std_normal \
    --task language_modeling \
    --self-target --attentive\
    --save-dir checkpoints/transformer_wikitext-103 \
    --tensorboard-logdir tb-logs \
    --arch neural_process_lm  --user-dir anp4nlg \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.005 \
    --lr-scheduler inverse_sqrt \
    --disable-validation \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 2048 \
    --sample-break-mode none \
    --max-tokens 2048 \
    --update-freq 16 \
    --max-update 50000 \
    --batch-size 128\
    --r_dim=256 \
    --s_dim=256 \
    --h_dim=256 \
    --z_dim=256 \
    --criterion neural_process

