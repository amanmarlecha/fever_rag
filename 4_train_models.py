from helpers import *

# STEP 4 - Train two models:  (A) claim only (no search)   (B) claim + searched evidence
index = load_index()
train = load_claims("train", N_TRAIN)

print("Searching evidence for every training claim ...")
retrieved = search(index, train.claim.tolist(), k=TOP_K + 5)
gold = gold_to_ids(train, index["key_to_id"])
evidence = make_training_evidence(train, retrieved, gold, index)
print("Example claim:", train.claim[0])
print("Example evidence:", evidence[0][:300])

print("\n--- Training model A: claim only ---")
train_model(train.claim.tolist(), None, train.y.values, "models/claim_only", CLAIM_MAX_LEN)

print("\n--- Training model B: claim + evidence ---")
train_model(train.claim.tolist(), evidence, train.y.values, "models/with_evidence", MAX_LEN)
print("Step 4 finished. Next: step 5")
