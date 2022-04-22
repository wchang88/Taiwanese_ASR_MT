#!/bin/bash

set -euo pipefail

BINARIZED_DATA=data/suisiann_binarized/nan_spm8000/nan_cmn/
MODEL_DIR=checkpoints/icorpus_nan_spm8000/nan_cmn/
cp $BINARIZED_DATA/dict.nan.txt $BINARIZED_DATA/dict.cmn.txt

## SuiSiann
fairseq-generate $BINARIZED_DATA \
    --gen-subset test \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 32 \
    --remove-bpe sentencepiece \
    --beam 5 | grep ^H | cut -c 3- | sort -n | cut -f3- > "$MODEL_DIR"/suisiann_b5.pred


## TAT-Vol2
fairseq-generate $BINARIZED_DATA \
    --gen-subset valid \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 32 \
    --remove-bpe sentencepiece \
    --beam 5 | grep ^H | cut -c 3- | sort -n | cut -f3- > "$MODEL_DIR"/tatvol2_b5.pred