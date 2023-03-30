'''
Process the cascaded ASR output into a format compatible with the trained MT model
'''
import sys, re
import pandas as pd
from util import *

if len(sys.argv) < 3:
    print("usage: python process_cascase_asr.py path/to/asr/output path/to/asr/raw_data/asr.orig.nan")
    sys.exit(1)

asr_path = sys.argv[1]
output_path = sys.argv[2]

sentences = []
espnet_ids = []
with open(asr_path, encoding="utf-8") as f:
   line = f.readline()
   while line:
      splt = line.split()
      espnet_ids.append(splt[0])
      sentences.append(line[len(splt[0]) + 1:])
      line = f.readline()

with open(output_path, 'w') as f:
    f.writelines(sentences)

with open(output_path+'.id', 'w') as f:
    i = 0
    for espnet_id in espnet_ids:
      f.write(f"{i} {espnet_id}\n")
      i += 1