#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi
EVAL_NAME=$1
DECODE_FOLDER=eval_pseudolabel/$EVAL_NAME/decode_asr_lm_lm_train_lm_nan_char_valid.loss.ave_asr_model_valid.acc.ave  # DECODE_FOLDER/dev/text, DECODE_FOLDER/test/text contain the texts


COMET_DIR=comet

LANG=nan
TRG_LANG=cmn


if [ ! -f "eval_pseudolabel/$EVAL_NAME/dev.score" ] || [ ! -f "eval_pseudolabel/$EVAL_NAME/test.score" ]
then
    for split in dev test
    do

      awk '{print $1}' $DECODE_FOLDER/"$split"/text > eval_pseudolabel/$EVAL_NAME/"$split".ids
      awk '{print $1=""; print $0}' $DECODE_FOLDER/"$split"/text | awk 'NF' > eval_pseudolabel/$EVAL_NAME/"$split".rawpred

      python postprocess_prediction.py \
          --raw_pred eval_pseudolabel/$EVAL_NAME/"$split".rawpred \
          --sent_ids eval_pseudolabel/$EVAL_NAME/"$split".ids \
          --final_pred eval_pseudolabel/$EVAL_NAME/"$split".pred \
          --gt_ids eval_pseudolabel/ref."$split".ids

      python score.py eval_pseudolabel/$EVAL_NAME/"$split".pred eval_pseudolabel/ref."$split".cmn \
          --src eval/$EVAL_NAME/raw/"$split".orig.nan \
          --comet-dir $COMET_DIR \
          --calculate_cer \
          --trg_lang zh \
          | tee eval_pseudolabel/$EVAL_NAME/"$split".score

    done
else
    echo "skip scoring for $EVAL_NAME because score files already exist."
fi