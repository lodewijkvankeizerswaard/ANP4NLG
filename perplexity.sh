#!/bin/bash

# Score a trained language model for perplexity.

# Usage: ./perplexity.sh -m /path/to/model.pt

while getopts m: flag
do
    case "${flag}" in
        m) MODEL_PATH=${OPTARG};;
    esac
done


fairseq-eval-lm data-bin/wikitext-103 \
    --path $MODEL_PATH \
    --batch-size 128 \
    --tokens-per-sample 64 \
    --context-window 1 \
    --user-dir anp4nlg \
    --self-target
