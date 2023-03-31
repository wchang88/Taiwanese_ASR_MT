#!/bin/bash

set -euo pipefail

RAW_DATA=data/suisiann_raw/nan_cmn/
BINARIZED_DATA=data/suisiann_binarized/nan_spm8000/nan_cmn/
MODEL_DIR=checkpoints/icorpus_nan_spm8000/nan_cmn/
cp $BINARIZED_DATA/dict.nan.txt $BINARIZED_DATA/dict.cmn.txt

## SuiSiann
fairseq-generate $BINARIZED_DATA \
    --gen-subset test \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 32 \
    --remove-bpe sentencepiece \
    --beam 5 > "$MODEL_DIR"/suisiann_b5.rawpred

echo "aligning fairseq-generate predictions to gold truths for suisiaan set"
suisiann_gold_size=$(wc -l < "$RAW_DATA"/all.orig.nan)
python postprocess_fairseq-gen.py "$MODEL_DIR"/suisiann_b5.rawpred \
    "$MODEL_DIR"/suisiann_b5.pred \
    "$suisiann_gold_size"


## TAT-Vol2
fairseq-generate $BINARIZED_DATA \
    --gen-subset valid \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 32 \
    --remove-bpe sentencepiece \
    --beam 5 > "$MODEL_DIR"/tatvol2_b5.rawpred

echo "aligning fairseq-generate predictions to gold truths for tatvol2 set"
tatvol2_gold_size=$(wc -l < "$RAW_DATA"/tatvol2.orig.nan)
python postprocess_fairseq-gen.py "$MODEL_DIR"/tatvol2_b5.rawpred \
    "$MODEL_DIR"/tatvol2_b5.pred \
    "$tatvol2_gold_size"
