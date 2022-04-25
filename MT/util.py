import sys, re
import pandas as pd
import epitran

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

    return line


def separate_punctuation(line):
    # use fullwidth punctuations because the training data had them
    line = line.replace(",", "，\n")
    line = line.replace(".", "。\n")
    line = line.replace("?", "？\n")
    line = line.replace("!", "！\n")
    line = line.replace('"', "")  # remove quotation marks
    lines = line.split("\n")
    # sentence has to contain at least one letter
    ret = [l.strip()+'\n' for l in lines if re.match(r'[a-z]', l.strip()) is not None]
    return ret
