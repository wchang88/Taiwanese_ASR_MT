'''
Process the SiuSann dataset into a format compatible with the trained MT model
Known issue: will add tone 1 or 4 to non-Taiwanese (e.g. English) words
'''
import sys, re
import pandas as pd
import epitran

if len(sys.argv) < 3:
    print("usage: python process_siusann.py path/to/SiuSann.csv path/to/fairseq/raw_data/all.orig.nan")
    sys.exit(1)

siusann_path = sys.argv[1]
output_path = sys.argv[2]
try:
    epi = epitran.Epitran('temp-nan', tones=True)
except KeyError:
    print("You are probably using the official Epitran. Please install our fork from https://github.com/kalvinchang/epitran-temp")
    sys.exit(1)


def diacritics2numbers(line):
    line = re.sub(r'([,.?!:-])', r" \1 ", line)

    punc = {'SP', ',', '.', '?', '!', ':', '-', '"'}
    words = line.split()
    for i in range(len(words)):
        if words[i] not in punc:
            words[i] = epi.transliterate(words[i])

    line = ' '.join(words)

    # then recover by replacing punctuation with the space removed
    # line = re.sub(r'SP', r" SP ", line)
    # only replace dash
    line = re.sub(r' (-) ', r"\1", line)
    # the double dash does not have a dash preceding it
    line = re.sub(r'(--) ', r"\1", line)

    # use fullwidth punctuations because the training data had them
    line = line.replace(",", "，\n")
    line = line.replace(".", "。\n")
    line = line.replace("?", "？\n")
    line = line.replace("!", "！\n")
    line = line.replace('"', "")  # remove quotation marks
    return line


def process(line):
    lines = diacritics2numbers(line).split("\n")
    # sentence has to contain at least one letter
    ret = [l.strip()+'\n' for l in lines if re.match(r'[a-z]', l.strip()) is not None]
    return ret


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