from helpers import *

# STEP 3 - Test the SEARCH: keyword vs meaning vs hybrid (precision, recall, "found the full proof")
index = load_index()
dev = load_claims("dev", N_DEV)
gold = gold_to_ids(dev, index["key_to_id"])
claims = dev.claim.tolist()

results = {}
for mode in ["keyword", "meaning", "hybrid"]:
    found = search(index, claims, k=10, mode=mode)
    row = {}
    for k in [1, 3, 5]:
        s = retrieval_scores(gold, found, k)
        row[f"precision@{k}"] = s["precision"]
        row[f"recall@{k}"] = s["recall"]
        row[f"found_all@{k}"] = s["found_all"]
    results[mode] = row

table = pd.DataFrame(results).T.round(3)
print(table.to_string())
table.to_csv("outputs/retrieval_results.csv")

plt.figure(figsize=(5, 3.5))
plt.bar(table.index, table["found_all@5"], color=["#999999", "#4C72B0", "#C85A1E"])
plt.ylim(0, 1); plt.ylabel("full proof found in top 5"); plt.title("Search methods compared")
plt.tight_layout(); plt.savefig("outputs/retrieval_comparison.png", dpi=150)
print("Step 3 finished. Next: step 4")
