from helpers import *

# STEP 2 - Build the evidence library and the search index.
train = load_claims("train", N_TRAIN)
dev = load_claims("dev", N_DEV)
print(train.label.value_counts().to_dict())

build_corpus(train, dev)      # write all the Wikipedia sentences we need to data/corpus.jsonl
index = load_index()          # compute meaning vectors (slow the first time, then saved)
print("Step 2 finished. Next: step 3")
