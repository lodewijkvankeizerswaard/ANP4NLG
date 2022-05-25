#!/bin/bash

source activate anp4nlg_gpu

TEXT=wikitext-103

# python preprocess.py "$TEXT"
fairseq-preprocess \
    --only-source \
    --thresholdsrc=2 \
    --tokenizer space \
    --task language_modeling \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
