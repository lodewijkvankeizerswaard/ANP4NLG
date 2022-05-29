#!/bin/bash

fairseq-interactive \
  --path checkpoints/transformer_wikitext-103/checkpoint_last.pt \
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
  --min-len 64 \
  --sacrebleu