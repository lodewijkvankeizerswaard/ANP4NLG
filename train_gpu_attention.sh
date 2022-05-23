#!/bin/bash

fairseq-train data-bin/wikitext-103 \
    --task language_modeling \
    --self-target \
    --save-dir checkpoints/transformer_wikitext-103 \
    --arch attentive_neural_process_lm  --user-dir anp4nlg \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --disable-validation \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 32 \
    --sample-break-mode none \
    --max-tokens 2048 \
    --update-freq 16 \
    --max-update 50000 \
    --batch-size 32\
    --criterion neural_process

