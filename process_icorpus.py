import os
import numpy as np
np.random.seed(11737)

CORPUS_DIR = "/home/cuichenx/Datasets/TaiwaneseDatasets/icorpus_ka1_han3-ji7"  # change to your path
OUTPUT_DIR = "data/icorpus_raw/nan_cmn"
os.makedirs(OUTPUT_DIR, exist_ok=True)
valtest_ratio = 0.05  # 0.90|0.05|0.05


cmn_path = os.path.join(CORPUS_DIR, "語料", "斷詞華語.txt")
nan_path = os.path.join(CORPUS_DIR, "語料", "自動標人工改音標.txt")  # tailo romanization

with open(cmn_path) as f:
	cmn_sents = np.array(f.read().split('\n'))
with open(nan_path) as f:
	nan_sents = np.array(f.read().split('\n'))
assert len(cmn_sents) == len(nan_sents)

N = len(cmn_sents)
indices = np.arange(N)
np.random.shuffle(indices)
val_indices = indices[0:int(N*valtest_ratio)]
test_indices = indices[int(N*valtest_ratio):int(N*valtest_ratio*2)]
train_indices = indices[int(N*valtest_ratio*2):]

np.savetxt(os.path.join(OUTPUT_DIR, "train.orig.cmn"), cmn_sents[train_indices], delimiter='\n', fmt="%s")
np.savetxt(os.path.join(OUTPUT_DIR, "train.orig.nan"), nan_sents[train_indices], delimiter='\n', fmt="%s")
np.savetxt(os.path.join(OUTPUT_DIR, "dev.orig.cmn"),   cmn_sents[val_indices], delimiter='\n', fmt="%s")
np.savetxt(os.path.join(OUTPUT_DIR, "dev.orig.nan"),   nan_sents[val_indices], delimiter='\n', fmt="%s")
np.savetxt(os.path.join(OUTPUT_DIR, "test.orig.cmn"),  cmn_sents[test_indices], delimiter='\n', fmt="%s")
np.savetxt(os.path.join(OUTPUT_DIR, "test.orig.nan"),  nan_sents[test_indices], delimiter='\n', fmt="%s")
