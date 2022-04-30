'''
Process the asr output into a format compatible with the trained MT model
Known issue: will add tone 1 or 4 to non-Taiwanese (e.g. English) words
'''
import sys, re, os
from util import *

if len(sys.argv) < 3:
    print("usage: python process_asr.py path/to/decode/folder path/to/eval/folder")
    sys.exit(1)

decode_path = sys.argv[1]
output_path = sys.argv[2]

def process(line):
    line = diacritics2numbers(line)
    return separate_punctuation(line)


for split in ('dev', 'test'):
    with open(os.path.join(decode_path, split, 'text')) as f:
        sentences = f.read().split("\n")
    out_sentences = []
    out_sent_ids = []
    for sent in sentences:
        if not sent: continue
        sent_id, actual_sent = sent.split(" ", maxsplit=1)  # remove sentence id
        processed = process(actual_sent)
        out_sentences.extend(processed)
        out_sent_ids.extend([sent_id+"\n"] * len(processed))

    with open(os.path.join(output_path, split+".orig.nan"), 'w') as f:
        f.writelines(out_sentences)

    with open(os.path.join(output_path, "..", split+".ids"), 'w') as f:
        f.writelines(out_sent_ids)