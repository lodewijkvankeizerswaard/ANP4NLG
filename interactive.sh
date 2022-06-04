#!/bin/bash

# Sample text from a language model

# Usage: ./interactive.sh -m /path/to/model.pt

while getopts m:c: flag
do
    case "${flag}" in
        m) MODEL_PATH=${OPTARG};;
    esac
done

fairseq-interactive \
  --path $MODEL_PATH \
  data-bin/wikitext-103/ \
  --task language_modeling \
  --bpe subword_nmt \
  --bpe-codes wikitext-103/codes.bpe \
  --user-dir anp4nlg \
  --self-target \
  --sampling \
  --sampling-topk 20 \
  --no-repeat-ngram-size 3 \
  --nbest 5 \
  --temperature 1.5 \
  --min-len 64