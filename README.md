# Taiwanese_ASR_MT
11737 Project: Taiwanese speech to Mandarin text
This README is under construction

## Data
- icorpus
- siusann

## To Run
MT: 
- Process the input data with `preprocess-icorpus-bilingual.sh`.
Make sure to change the ICORPUS_PATH variable in the script.

Once model is trained, we can apply it to 1) the SiuSann dataset, or 2) the output of ASR, 
to get the corresponding sentences in Mandarin.

To process the SiuSann csv to a compatible format
```bash
git clone https://github.com/kalvinchang/epitran-temp.git
cd epitran-temp
python setup.py install
cd ../MT
python 

```