# Taiwanese_ASR_MT
11737 Project: Cascade vs Pseudo-label Speech Translation from Taiwanese to Mandarin
Please see our paper [here](https://github.com/cuichenx/Taiwanese_ASR_MT/blob/main/Cascade%20vs%20Pseudo-Label%20Speech%20to%20Text%20Translation%20from%20Taiwanese%20to%20Mandarin.pdf)

## Data
- [iCorpus](https://github.com/Taiwanese-Corpus/icorpus_ka1_han3-ji7) (MT)
- [SuiSiann](https://suisiann-dataset.ithuan.tw/) (ASR)
- [TAT-Vol2 Sample](https://sites.google.com/speech.ntut.edu.tw/fsw/home/tat-corpus?authuser=0) (ASR)

## To Run
### Setup
- Run `pip install -r requirements.txt` to install necessary dependencies  
- Set the FAIRSEQ_DIR environmental variable via `export FAIRSEQ_DIR="/path/to/fairseq/"`  
### MT
- Change to the MT directory
- Process the input data with `preprocess-icorpus-bilingual.sh /path/to/icorpus/directory`.
- Train the model with `traineval_nan_cmn.sh`

Once model is trained, we can apply it to 1) the SuiSiann dataset, or 2) the output of ASR, 
to get the corresponding sentences in Mandarin (Cascade).


To apply the MT model to SuiSian (and TAT sample), we need to process the SuiSiann csv and TAT sample to a compatible format

1) install a fork of epitran to convert Tailo diacritics to numbers
```bash
git clone https://github.com/kalvinchang/epitran-temp.git
cd epitran-temp
python setup.py install
cd ../MT
```
2) `preprocess-suisiann.sh`

To apply the MT model to the output of ASR for cascade speech translation, place the corresponding ESPNet decode folder in `eval/EVAL_NAME/decode...`. Then run `preprocess-eval-asr.sh EVAL_NAME` to run and evaluate the cascade model.

Alternatively, you can run `cascade_all.sh` to evaluate all the different runs in the `eval` folder.


### ASR
Please see this [ESPNet fork](https://github.com/yunhsuanchen/espnet/tree/master/egs2/nan_suisiann/asr1) for training ASR models and ST models with pseudo label. We use the standard ESPNet training procedures.