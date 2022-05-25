#!/bin/bash

TEXT=wikitext-103

fairseq-preprocess \
    --only-source \
    --task language_modeling \
    --trainpref $TEXT/wiki.train.tokens.bpe \
    --validpref $TEXT/wiki.valid.tokens.bpe \
    --testpref $TEXT/wiki.test.tokens.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 20
