#!/bin/bash

TEXT=wikitext-103

python learn_bpe.py --input $TEXT/wiki.train.tokens --output $TEXT/codes.bpe
python apply_bpe.py --input $TEXT/wiki.train.tokens --output $TEXT/wiki.train.tokens.bpe --codes $TEXT/codes.bpe
python apply_bpe.py --input $TEXT/wiki.valid.tokens --output $TEXT/wiki.valid.tokens.bpe --codes $TEXT/codes.bpe
python apply_bpe.py --input $TEXT/wiki.test.tokens --output $TEXT/wiki.test.tokens.bpe --codes $TEXT/codes.bpe

fairseq-preprocess \
    --only-source \
    --task language_modeling \
    --trainpref $TEXT/wiki.train.tokens.bpe \
    --validpref $TEXT/wiki.valid.tokens.bpe \
    --testpref $TEXT/wiki.test.tokens.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 20
