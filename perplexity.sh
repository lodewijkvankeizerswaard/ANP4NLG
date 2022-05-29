#!/bin/bash

fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/transformer_wikitext-103/checkpoint_last.pt \
    --batch-size 128 \
    --tokens-per-sample 64 \
    --context-window 63 \
    --user-dir anp4nlg \
    --self-target \
    --quiet \
    --results-path perplexity-results.txt \
    --output-word-stats