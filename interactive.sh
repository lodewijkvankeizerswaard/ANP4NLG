#!/bin/bash

fairseq-interactive data-bin/wikitext-103 \
  --path checkpoints/transformer_wikitext-103/checkpoint_best.pt \
  --user-dir anp4nlg \
  --task language_modeling \
  --self-target
