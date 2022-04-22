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
ICORPUS_PATH="/home/cuichenx/Datasets/TaiwaneseDatasets/icorpus_ka1_han3-ji7"  ## CHANGE ME

LANGS=(nan)
TRG_LANG=cmn

for i in ${!LANGS[*]}; do
  LANG=${LANGS[$i]}

  RAW_DDIR=data/icorpus_raw/
  PROC_DDIR=data/icorpus_processed/"$LANG"_spm"$VOCAB_SIZE"/
  BINARIZED_DDIR=data/icorpus_binarized/"$LANG"_spm"$VOCAB_SIZE"/

  python process_icorpus.py $ICORPUS_PATH

  mkdir -p "$PROC_DDIR"/"$LANG"_"$TRG_LANG"

  # --- learn BPE with sentencepiece ---
  TRAIN_FILES="$RAW_DDIR"/"$LANG"_"$TRG_LANG"/train.orig."$LANG","$RAW_DDIR"/"$LANG"_"$TRG_LANG"/train.orig."$TRG_LANG"
  echo "learning joint BPE over ${TRAIN_FILES}..."
  python "$SPM_TRAIN" \
	    --input=$TRAIN_FILES \
	    --model_prefix="$PROC_DDIR"/"$LANG"_"$TRG_LANG"/spm"$VOCAB_SIZE" \
	    --vocab_size=$VOCAB_SIZE \
	    --character_coverage=1.0 \
	    --model_type=bpe
  spm_model="$PROC_DDIR"/"$LANG"_"$TRG_LANG"/spm"$VOCAB_SIZE".model

  python "$SPM_ENCODE" \
	  --model=$spm_model \
	  --output_format=piece \
	  --inputs "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/train.orig."$LANG" "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/train.orig."$TRG_LANG"  \
	  --outputs "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/train.spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/train.spm"$VOCAB_SIZE"."$TRG_LANG" \
	  --min-len 1 --max-len 200 
 
  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
    python "$SPM_ENCODE" \
	    --model=$spm_model \
	    --output_format=piece \
	    --inputs "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/"$split".orig."$LANG" "$RAW_DDIR"/"$LANG"_"$TRG_LANG"/"$split".orig."$TRG_LANG"  \
	    --outputs "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/"$split".spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/"$split".spm"$VOCAB_SIZE"."$TRG_LANG"  
  done

  # -- fairseq binarization ---
  echo "Binarize the data..."
  fairseq-preprocess --source-lang $LANG --target-lang "$TRG_LANG" \
	  --joined-dictionary \
	  --trainpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/train.spm"$VOCAB_SIZE" \
	  --validpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/dev.spm"$VOCAB_SIZE" \
	  --testpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/test.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/"$LANG"_"$TRG_LANG"/

  echo "Binarize the data..."
  fairseq-preprocess --source-lang "$TRG_LANG" --target-lang $LANG \
	  --joined-dictionary \
	  --trainpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/train.spm"$VOCAB_SIZE" \
	  --validpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/dev.spm"$VOCAB_SIZE" \
	  --testpref "$PROC_DDIR"/"$LANG"_"$TRG_LANG"/test.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/"$TRG_LANG"_"$LANG"/
done