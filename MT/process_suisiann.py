'''
Process the SiuSann dataset into a format compatible with the trained MT model
Known issue: will add tone 1 or 4 to non-Taiwanese (e.g. English) words
'''
import sys, re
import pandas as pd
from util import *

if len(sys.argv) < 3:
    print("usage: python process_siusann.py path/to/SiuSann.csv path/to/fairseq/raw_data/all.orig.nan")
    sys.exit(1)

siusann_path = sys.argv[1]
output_path = sys.argv[2]


def process(line):
    line = diacritics2numbers(line)
    return separate_punctuation(line)


sentences = pd.read_csv(siusann_path)['羅馬字'].to_list()
print(len(sentences))
out_sentences = []
sentence_numbers = []
for i, sent in enumerate(sentences):
    out_sents = process(sent)
    out_sentences.extend(out_sents)
    sentence_numbers.extend([f'{i}\n'] * len(out_sents)) 




with open(output_path, 'w') as f:
    f.writelines(out_sentences)

with open(output_path+'.id', 'w') as f:
    f.writelines(sentence_numbers)