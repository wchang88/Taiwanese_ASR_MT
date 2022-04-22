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
SUISIANN_PATH="data/SuiSiann/SuiSiann.csv"  ## CHANGE ME
TATVOL2_PATH="data/TAT-Vol2/json"  ## CHANGE ME
DICT_FILE="data/icorpus_binarized/nan_spm8000/cmn_nan/dict.cmn.txt"

LANGS=(nan)
TRG_LANG=cmn

for i in ${!LANGS[*]}; do
  LANG=${LANGS[$i]}

  RAW_DDIR=data/suisiann_raw/
  PROC_DDIR=data/suisiann_processed/"$LANG"_spm"$VOCAB_SIZE"/
  BINARIZED_DDIR=data/suisiann_binarized/"$LANG"_spm"$VOCAB_SIZE"/

  mkdir -p "$RAW_DDIR"/"$LANG"_"$TRG_LANG"
  python process_suisiann.py $SUISIANN_PATH "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/all.orig.$LANG
  python process_tatvol2.py $TATVOL2_PATH "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/tatvol2.orig.$LANG

  mkdir -p "$PROC_DDIR"/"$LANG"_"$TRG_LANG"

  
  spm_model=data/icorpus_processed/nan_spm8000/nan_cmn/spm8000.model

  echo "encoding valid/test data with learned BPE..."
  for split in all tatvol2;
  do
    python "$SPM_ENCODE" \
	    --model=$spm_model \
	    --output_format=piece \
	    --inputs "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/"$split".orig."$LANG" \
	    --outputs "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/"$split".spm"$VOCAB_SIZE"."$LANG"  
  done

  # -- fairseq binarization ---
  echo "Binarize the data..."
  fairseq-preprocess --only-source \
    --source-lang $LANG --target-lang "$TRG_LANG" \
	  --joined-dictionary \
    --srcdict $DICT_FILE \
	  --testpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/all.spm"$VOCAB_SIZE" \
    --validpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/tatvol2.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/"$LANG"_"$TRG_LANG"/

done