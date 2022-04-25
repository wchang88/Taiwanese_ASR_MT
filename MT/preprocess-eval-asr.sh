#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi
EVAL_NAME=$1
DECODE_FOLDER=eval/$EVAL_NAME/decode_asr_lm_lm_train_lm_nan_char_valid.loss.ave_asr_model_valid.acc.ave  # DECODE_FOLDER/dev/text, DECODE_FOLDER/test/text contain the texts

VOCAB_SIZE=8000
FAIR_SCRIPTS=$FAIRSEQ_DIR/scripts
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
DICT_FILE="data/icorpus_binarized/nan_spm8000/cmn_nan/dict.cmn.txt"
COMET_DIR=comet

LANG=nan
TRG_LANG=cmn



RAW_DDIR=eval/$EVAL_NAME/raw/
PROC_DDIR=eval/$EVAL_NAME/processed/
BINARIZED_DDIR=eval/$EVAL_NAME/binarized/
MODEL_DIR=checkpoints/icorpus_nan_spm8000/nan_cmn/

if [ ! -f "eval/$EVAL_NAME/dev_b5.rawpred" ] || [ ! -f "eval/$EVAL_NAME/test_b5.rawpred" ]
then
    mkdir -p $RAW_DDIR
    mkdir -p $PROC_DDIR
    mkdir -p $BINARIZED_DDIR

    python process_asr.py $DECODE_FOLDER $RAW_DDIR

    spm_model=data/icorpus_processed/nan_spm8000/nan_cmn/spm8000.model

    echo "encoding valid/test data with learned BPE..."
    for split in dev test;
    do
      python "$SPM_ENCODE" \
        --model=$spm_model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$split".orig."$LANG" \
        --outputs "$PROC_DDIR"/"$split".spm"$VOCAB_SIZE"."$LANG"  
    done

    # -- fairseq binarization ---
    echo "Binarize the data..."
    fairseq-preprocess --only-source \
      --source-lang $LANG --target-lang "$TRG_LANG" \
      --joined-dictionary \
      --srcdict $DICT_FILE \
      --testpref "$PROC_DDIR"/test.spm"$VOCAB_SIZE" \
      --validpref "$PROC_DDIR"/dev.spm"$VOCAB_SIZE" \
      --destdir $BINARIZED_DDIR/

      
    cp $BINARIZED_DDIR/dict.nan.txt $BINARIZED_DDIR/dict.cmn.txt

    fairseq-generate $BINARIZED_DDIR \
        --gen-subset test \
        --path $MODEL_DIR/checkpoint_best.pt \
        --batch-size 32 \
        --remove-bpe sentencepiece \
        --beam 5 | grep ^H | cut -c 3- | sort -n | cut -f3- > eval/$EVAL_NAME/test_b5.rawpred

    fairseq-generate $BINARIZED_DDIR \
        --gen-subset valid \
        --path $MODEL_DIR/checkpoint_best.pt \
        --batch-size 32 \
        --remove-bpe sentencepiece \
        --beam 5 | grep ^H | cut -c 3- | sort -n | cut -f3- > eval/$EVAL_NAME/dev_b5.rawpred
else
    echo "skip fairseq generate for $EVAL_NAME because pred files already exist."
fi


if [ ! -f "eval/$EVAL_NAME/dev_b5.score" ] || [ ! -f "eval/$EVAL_NAME/test_b5.score" ]
then
    for split in dev test
    do
      python postprocess_prediction.py \
          --raw_pred eval/$EVAL_NAME/"$split"_b5.rawpred \
          --sent_ids eval/$EVAL_NAME/"$split".ids \
          --final_pred eval/$EVAL_NAME/"$split"_b5.pred \
          --gt_ids eval/ref."$split".ids

      python score.py eval/$EVAL_NAME/"$split"_b5.pred eval/ref."$split".cmn \
          --src eval/$EVAL_NAME/raw/"$split".orig.nan \
          --comet-dir $COMET_DIR \
          --calculate_cer \
          --trg_lang zh \
          | tee eval/$EVAL_NAME/"$split"_b5.score

    done
else
    echo "skip scoring for $EVAL_NAME because score files already exist."
fi