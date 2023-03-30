#!/bin/bash

set -euo pipefail

RAW_DATA=data/icorpus_raw/nan_cmn/
BINARIZED_DATA=data/icorpus_binarized/nan_spm8000/nan_cmn/
MODEL_DIR=checkpoints/icorpus_nan_spm8000/nan_cmn/
COMET_DIR=comet
mkdir -p $MODEL_DIR

fairseq-train \
	$BINARIZED_DATA \
	--task translation \
	--arch transformer_iwslt_de_en \
	--max-epoch 80 \
    --patience 5 \
    --distributed-world-size 1 \
	--share-all-embeddings \
	--no-epoch-checkpoints \
	--dropout 0.3 \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	--criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	--max-tokens 4500 \
	--update-freq 2 \
	--seed 2 \
  	--save-dir $MODEL_DIR \
	--log-interval 20 2>&1 | tee $MODEL_DIR/train.log 

# translate & eval the valid and test set
echo "predicting the test set"
fairseq-generate $BINARIZED_DATA \
    --gen-subset test \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 32 \
    --remove-bpe sentencepiece \
    --beam 5 > "$MODEL_DIR"/test_b5.rawpred

echo "aligning fairseq-generate predictions to gold truths for test set"
test_gold_size=$(wc -l < "$RAW_DATA"/test.orig.cmn)
python postprocess_fairseq-gen.py "$MODEL_DIR"/test_b5.rawpred \
    "$MODEL_DIR"/test_b5.pred \
    "$test_gold_size"

echo "evaluating test set"
python score.py "$MODEL_DIR"/test_b5.pred "$RAW_DATA"/test.orig.cmn \
    --src "$RAW_DATA"/test.orig.nan \
    --comet-dir $COMET_DIR \
    --trg_lang zh \
    | tee "$MODEL_DIR"/test_b5.score

echo "predicting the valid set"
fairseq-generate $BINARIZED_DATA \
    --gen-subset valid \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 32 \
    --remove-bpe sentencepiece \
    --beam 5 > "$MODEL_DIR"/valid_b5.rawpred

echo "aligning fairseq-generate predictions to gold truths for valid set"
valid_gold_size=$(wc -l < "$RAW_DATA"/dev.orig.cmn)
python postprocess_fairseq-gen.py "$MODEL_DIR"/valid_b5.rawpred \
    "$MODEL_DIR"/valid_b5.pred \
    "$valid_gold_size"

echo "evaluating valid set"
python score.py "$MODEL_DIR"/valid_b5.pred "$RAW_DATA"/dev.orig.cmn \
    --src "$RAW_DATA"/dev.orig.nan \
    --comet-dir $COMET_DIR \
    --trg_lang zh \
    | tee "$MODEL_DIR"/valid_b5.score