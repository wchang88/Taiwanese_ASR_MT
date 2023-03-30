#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi

VOCAB_SIZE=8000

FAIR_SCRIPTS=$FAIRSEQ_DIR/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
ASR_PATH=$1
DICT_FILE="data/icorpus_binarized/nan_spm8000/cmn_nan/dict.cmn.txt"

LANGS=(nan)
TRG_LANG=cmn

for i in ${!LANGS[*]}; do
  LANG=${LANGS[$i]}

  RAW_DDIR=data/cascade_asr_raw/
  PROC_DDIR=data/cascade_asr_processed/"$LANG"_spm"$VOCAB_SIZE"/
  BINARIZED_DDIR=data/cascade_asr_binarized/"$LANG"_spm"$VOCAB_SIZE"/

  mkdir -p "$RAW_DDIR"/"$LANG"_"$TRG_LANG"
  python process_cascade_asr.py $ASR_PATH "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/asr.orig.$LANG

  mkdir -p "$PROC_DDIR"/"$LANG"_"$TRG_LANG"
  
  spm_model=data/icorpus_processed/nan_spm8000/nan_cmn/spm8000.model

  echo "encoding ASR data with learned BPE..."
  python "$SPM_ENCODE" \
	--model=$spm_model \
	--output_format=piece \
	--inputs "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/asr.orig."$LANG" \
	--outputs "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/asr.spm"$VOCAB_SIZE"."$LANG"  

  # -- fairseq binarization ---
  echo "Binarize the data..."
  fairseq-preprocess --only-source \
    --source-lang $LANG --target-lang "$TRG_LANG" \
	  --joined-dictionary \
    --srcdict $DICT_FILE \
	  --testpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/asr.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/"$LANG"_"$TRG_LANG"/

done