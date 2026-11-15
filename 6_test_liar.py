from helpers import *

# STEP 6 - Out-of-domain test: models trained on FEVER (Wikipedia) tested on LIAR (political statements)
index = load_index()
liar = load_liar()
claims = liar.claim.tolist()
gold_labels = liar.gold.values
print(len(liar), "LIAR claims. Always guessing the most common label gives",
      round(liar.gold.value_counts(normalize=True).max(), 3))

retrieved = search(index, claims, k=TOP_K)
evidence = [passages_to_text(index, ids) for ids in retrieved]

rows = []
for name, folder, ev, max_len in [("A: claim only", "models/claim_only", None, CLAIM_MAX_LEN),
                                  ("B: claim + evidence", "models/with_evidence", evidence, MAX_LEN)]:
    probs = predict(load_verifier(folder), claims, ev, max_len)
    pred = np.array([LABELS[i] for i in probs.argmax(axis=1)])
    gave_verdict = pred != "NOT ENOUGH INFO"                              # the model did not say "not enough info"
    rows.append({"model": name,
                 "says NOT ENOUGH INFO": 1 - gave_verdict.mean(),
                 "accuracy when it gives a verdict": (pred[gave_verdict] == gold_labels[gave_verdict]).mean(),
                 "accuracy (NEI counts as wrong)": (pred == gold_labels).mean()})
table = pd.DataFrame(rows).set_index("model").round(3)
print(table.to_string())
table.to_csv("outputs/liar_results.csv")
print("\nNote: Wikipedia rarely contains political facts, so model B should often say NOT ENOUGH INFO. That is honest behaviour, not a bug.")
print("Step 6 finished. Next: step 7")
