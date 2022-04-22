'''
Process the asr output into a format compatible with the trained MT model
Known issue: will add tone 1 or 4 to non-Taiwanese (e.g. English) words
'''
import sys, re, os
import epitran

if len(sys.argv) < 3:
    print("usage: python process_asr.py path/to/decode/folder path/to/eval/folder")
    sys.exit(1)

decode_path = sys.argv[1]
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
    return line


for split in ('dev', 'test'):
    with open(os.path.join(decode_path, split, 'text')) as f:
        sentences = f.read().split("\n")
    out_sentences = []
    for sent in sentences:
        if not sent: continue
        actual_sent = sent.split(" ", maxsplit=1)[1]  # remove sentence id
        out_sentences.append(diacritics2numbers(actual_sent)+'\n')

    with open(os.path.join(output_path, split+".orig.nan"), 'w') as f:
        f.writelines(out_sentences)
