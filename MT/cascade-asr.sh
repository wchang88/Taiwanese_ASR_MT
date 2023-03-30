#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi

VOCAB_SIZE=8000
FAIR_SCRIPTS=$FAIRSEQ_DIR/scripts
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
DICT_FILE="data/icorpus_binarized/nan_spm8000/cmn_nan/dict.cmn.txt"
COMET_DIR=comet

LANG=nan
TRG_LANG=cmn

CASCADE_DIR=cascade_asr/
mkdir -p $CASCADE_DIR

RUN_ID=$1

RAW_DDIR=$CASCADE_DIR/$RUN_ID/raw/
PROC_DDIR=$CASCADE_DIR/$RUN_ID/processed/
BINARIZED_DDIR=$CASCADE_DIR/$RUN_ID/binarized/
CASCADE_OUT=$CASCADE_DIR/$RUN_ID/out
MODEL_DIR=checkpoints/icorpus_nan_spm8000/nan_cmn/

mkdir -p $RAW_DDIR
mkdir -p $PROC_DDIR
mkdir -p $BINARIZED_DDIR
mkdir -p $CASCADE_OUT

# Preprocess the ASR data
ASR_PATH=$2
python process_cascade_asr.py $ASR_PATH "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/asr.orig.$LANG

spm_model=data/icorpus_processed/nan_spm8000/nan_cmn/spm8000.model

echo "encoding cascaded ASR data with learned BPE..."
python "$SPM_ENCODE" \
    --model=$spm_model \
    --output_format=piece \
    --inputs "$RAW_DDIR"/test.orig."$LANG" \
    --outputs "$PROC_DDIR"/test.spm"$VOCAB_SIZE"."$LANG"

# -- fairseq binarization ---
echo "Binarize the data..."
fairseq-preprocess --only-source \
    --source-lang $LANG --target-lang "$TRG_LANG" \
    --joined-dictionary \
    --srcdict $DICT_FILE \
    --testpref "$PROC_DDIR"/test.spm"$VOCAB_SIZE" \
    --destdir $BINARIZED_DDIR/

cp $BINARIZED_DDIR/dict.nan.txt $BINARIZED_DDIR/dict.cmn.txt


# translate the cascaded ASR data
echo "predicting the cascaded ASR output"
fairseq-generate $BINARIZED_DDIR \
    --gen-subset test \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 32 \
    --remove-bpe sentencepiece \
    --beam 5 > "$CASCADE_OUT"/asr_b5.rawpred

echo "aligning fairseq-generate predictions to original inputs"
test_gold_size=$(wc -l < "$RAW_DDIR"/asr.orig.nan.id)
python postprocess_fairseq-gen.py "$CASCADE_OUT"/asr_b5.rawpred \
    "$CASCADE_OUT"/asr_b5.pred \
    "$test_gold_size"



