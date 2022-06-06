#!/bin/bash

fairseq-train data-bin/wikitext-103 \
    --attentive \
    --task language_modeling \
    --self-target \
    --save-dir checkpoints/transformer_wikitext-103 \
    --arch neural_process_lm --user-dir anp4nlg \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.001 \
    --lr-scheduler inverse_sqrt \
    --disable-validation \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 64 \
    --sample-break-mode none \
    --max-tokens 2048 \
    --update-freq 16 \
    --max-update 50000 \
    --cpu \
    --criterion neural_process \
    --batch-size 128
