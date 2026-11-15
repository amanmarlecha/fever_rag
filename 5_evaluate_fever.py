from helpers import *

# STEP 5 - Test on FEVER: accuracy, FEVER score, and error analysis
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

index = load_index()
dev = load_claims("dev", N_DEV)
claims, y = dev.claim.tolist(), dev.y.values
gold = gold_to_ids(dev, index["key_to_id"])

retrieved = search(index, claims, k=TOP_K)                              # the model only gets what the search finds
evidence = [passages_to_text(index, ids) for ids in retrieved]
perfect = [passages_to_text(index, g[0][:TOP_K]) if g else e for g, e in zip(gold, evidence)]   # cheating version (upper limit)

model_a = load_verifier("models/claim_only")
probs_a = predict(model_a, claims, None, CLAIM_MAX_LEN)
del model_a
model_b = load_verifier("models/with_evidence")
probs_b = predict(model_b, claims, evidence, MAX_LEN)
probs_c = predict(model_b, claims, perfect, MAX_LEN)

rows = []
for name, probs, use_fever in [("A: claim only (no search)", probs_a, False),
                               ("B: claim + searched evidence", probs_b, True),
                               ("C: claim + perfect evidence (upper limit)", probs_c, False)]:
    pred = probs.argmax(axis=1)
    rows.append({"model": name,
                 "accuracy": accuracy_score(y, pred),
                 "macro_F1": f1_score(y, pred, average="macro"),
                 "FEVER_score": fever_score(pred, y, gold, retrieved) if use_fever else np.nan})
results = pd.DataFrame(rows).set_index("model").round(3)
print(results.to_string())
results.to_csv("outputs/fever_results.csv")
print("\nModel B per-label report:")
print(classification_report(y, probs_b.argmax(axis=1), target_names=LABELS, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, probs) in zip(axes, [("A: claim only", probs_a), ("B: with evidence", probs_b)]):
    ConfusionMatrixDisplay.from_predictions(y, probs.argmax(axis=1), display_labels=["REF", "SUP", "NEI"],
                                            normalize="true", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.savefig("outputs/confusion_matrices.png", dpi=150)

# ---------- ERROR ANALYSIS: why is model B wrong? ----------
pred_b, pred_c = probs_b.argmax(axis=1), probs_c.argmax(axis=1)
reasons = []
for i in range(len(dev)):
    if pred_b[i] == y[i]:
        reasons.append("correct")
    elif y[i] == 2:
        reasons.append("NEI claim wrongly 'proved'")                     # claim cannot be checked, but model gave a verdict
    elif not complete_evidence_found(gold[i], retrieved[i]):
        reasons.append("SEARCH failed")                                  # the proof was never found
    elif pred_c[i] == y[i]:
        reasons.append("distracted by extra sentences")                  # proof found, perfect evidence works, noisy fails
    else:
        reasons.append("REASONING failed")                               # even perfect evidence gives a wrong answer
dev["error_type"] = reasons
dev["prediction_B"] = [LABELS[i] for i in pred_b]
dev["best_search_result"] = [index["texts"][ids[0]] for ids in retrieved]
dev.drop(columns=["gold_groups"]).to_csv("outputs/dev_predictions.csv", index=False)
dev[dev.error_type != "correct"].drop(columns=["gold_groups"]).to_csv("outputs/error_analysis.csv", index=False)

counts = dev[dev.error_type != "correct"].error_type.value_counts()
print("\nWhy model B makes mistakes:")
print(counts.to_string())
plt.figure(figsize=(7, 3)); plt.barh(counts.index, counts.values, color="#4C72B0"); plt.gca().invert_yaxis()
plt.title("Where do the mistakes come from?"); plt.tight_layout(); plt.savefig("outputs/error_breakdown.png", dpi=150)
print("Step 5 finished. Next: step 6")
