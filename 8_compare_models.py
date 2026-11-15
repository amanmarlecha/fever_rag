from helpers import *

# STEP 8 - COMPARISON: is our model really better? Every model below is tested on the SAME dev claims and SAME library.
index = load_index()
train = load_claims("train", N_TRAIN)
dev = load_claims("dev", N_DEV)
claims, y = dev.claim.tolist(), dev.y.values
gold = gold_to_ids(dev, index["key_to_id"])

# Evidence found by each kind of search (top 5 sentences for every dev claim)
found = {mode: search(index, claims, k=TOP_K, mode=mode) for mode in ["keyword", "meaning", "hybrid"]}
evidence = {mode: [passages_to_text(index, ids) for ids in found[mode]] for mode in found}
perfect = [passages_to_text(index, g[0][:TOP_K]) if g else e for g, e in zip(gold, evidence["hybrid"])]

# Training data for the old-style models (same recipe as our model B, but with keyword search only)
print("Preparing training data for the old-style models ...")
train_found = search(index, train.claim.tolist(), k=TOP_K + 5, mode="keyword")
train_evidence = make_training_evidence(train, train_found, gold_to_ids(train, index["key_to_id"]), index)

rows = []
def add(name, probs_or_pred, retrieved=None):
    pred = probs_or_pred.argmax(axis=1) if probs_or_pred.ndim == 2 else probs_or_pred
    rows.append({"model": name,
                 "accuracy": accuracy_score(y, pred) * 100,
                 "macro_F1": f1_score(y, pred, average="macro") * 100,
                 "FEVER_score": fever_score(pred, y, gold, retrieved) * 100 if retrieved is not None else np.nan})

# 1. The laziest model: always answer with the most common training label
add("1. Always guess one label", np.full(len(y), np.bincount(train.y).argmax()))

# 2. Old-style text-only model (word counts + logistic regression), no search
vectors, lr = fit_classic(train.claim.tolist(), None, train.y.values)
add("2. Old-style, claim only (TF-IDF + LogReg)", lr.predict_proba(classic_features(vectors, claims, None)))

# 3. Old-style PIPELINE in the spirit of the 2018 FEVER baseline: keyword search + word counts + logistic regression
vectors, lr = fit_classic(train.claim.tolist(), train_evidence, train.y.values)
add("3. Old-style pipeline (keyword search + LogReg)",
    lr.predict_proba(classic_features(vectors, claims, evidence["keyword"])), found["keyword"])

# 4. A pretrained language model that has NOT been taught our task (off the shelf)
add("4. Pretrained NLI model, no fine-tuning",
    predict(load_verifier(VERIFIER_BASE), claims, evidence["hybrid"], MAX_LEN), found["hybrid"])

# 5. OUR model A: fine-tuned, claim only
add("5. Ours A: fine-tuned, claim only", predict(load_verifier("models/claim_only"), claims, None, CLAIM_MAX_LEN))

# 6-8. OUR model B with different kinds of search (shows why hybrid search helps)
model_b = load_verifier("models/with_evidence")
add("6. Ours B + keyword search only", predict(model_b, claims, evidence["keyword"], MAX_LEN), found["keyword"])
add("7. Ours B + meaning search only", predict(model_b, claims, evidence["meaning"], MAX_LEN), found["meaning"])
add("8. OURS B + hybrid search (main model)", predict(model_b, claims, evidence["hybrid"], MAX_LEN), found["hybrid"])

# 9. Upper limit: our model B given the correct evidence
add("9. Ours B + perfect evidence (upper limit)", predict(model_b, claims, perfect, MAX_LEN))

table = pd.DataFrame(rows).set_index("model").round(1)
print("\n=== SAME-CONDITIONS COMPARISON (percent) ===")
print(table.to_string())
table.to_csv("outputs/comparison_same_setting.csv")

# What did each idea add?  (difference in accuracy points)
acc = table["accuracy"]
print("\n=== WHAT EACH IDEA ADDS (accuracy points) ===")
print(f"Deep learning instead of word counting (5 vs 2)      : {acc.iloc[4] - acc.iloc[1]:+.1f}")
print(f"Fine-tuning on our data (8 vs 4)                      : {acc.iloc[7] - acc.iloc[3]:+.1f}")
print(f"Adding searched evidence (8 vs 5)                     : {acc.iloc[7] - acc.iloc[4]:+.1f}")
print(f"Hybrid search instead of keyword search (8 vs 6)      : {acc.iloc[7] - acc.iloc[5]:+.1f}")
print(f"Our full model vs old-style pipeline (8 vs 3)         : {acc.iloc[7] - acc.iloc[2]:+.1f}")

# Results published in research papers (numbers copied from the papers - see 'source')
published = pd.DataFrame([
    ("FEVER 2018 baseline (Thorne et al.)", 50.91, 31.87, "FEVER paper abstract, arXiv:1803.05355"),
    ("UKP-Athene (Hanselowski et al.)", 68.49, 64.74, "KGAT paper Table 2 (dev set), arXiv:1910.09796"),
    ("UNC NLP (Nie et al.)", 69.72, 66.49, "KGAT paper Table 2 (dev set), arXiv:1910.09796"),
    ("GEAR (Zhou et al.)", 74.84, 70.69, "KGAT paper Table 2 (dev set), arXiv:1910.09796"),
    ("KGAT, BERT-base (Liu et al.)", 78.02, 75.88, "KGAT paper Table 2 (dev set), arXiv:1910.09796"),
    ("KGAT, RoBERTa-large (Liu et al.)", 78.29, 76.11, "KGAT paper Table 2 (dev set), arXiv:1910.09796"),
], columns=["system", "label_accuracy", "FEVER_score", "source"])
ours = table.loc["8. OURS B + hybrid search (main model)"]
published.loc[len(published)] = ["THIS PROJECT (our small library!)", ours["accuracy"], ours["FEVER_score"], "this run"]
print("\n=== PUBLISHED RESULTS vs OURS (percent) ===")
print(published.drop(columns="source").to_string(index=False))
print("\nCAUTION: the published systems search ALL of Wikipedia (about 5.4 million pages) and use the official test or dev set.")
print("Our library is much smaller, so our numbers are NOT directly comparable and look better than they would at full scale.")
published.to_csv("outputs/comparison_published.csv", index=False)

# Chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names = [n.split(". ", 1)[1] for n in table.index]
colors = ["#C85A1E" if "OURS" in n else ("#4C72B0" if "Ours" in n else "#999999") for n in table.index]
for ax, column, title in [(axes[0], "accuracy", "Label accuracy (%)"), (axes[1], "FEVER_score", "FEVER score (%) - needs search")]:
    values = table[column].fillna(0).values
    ax.barh(names, values, color=colors); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_title(title)
    for i, v in enumerate(table[column].values):
        ax.text(1, i, "n/a" if np.isnan(v) else f"{v:.1f}", va="center", color="white" if (not np.isnan(v) and v > 8) else "black", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/comparison_chart.png", dpi=150)
print("Saved: outputs/comparison_same_setting.csv, comparison_published.csv, comparison_chart.png")
print("Step 8 finished. All done!")
