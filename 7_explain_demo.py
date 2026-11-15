from helpers import *

# STEP 7 - The full pipeline: claim -> search -> verdict -> LLM explanation
MY_CLAIMS = [                      # <-- type your own claims here
    "Barack Obama was born in Hawaii.",
    "The Eiffel Tower is located in Berlin.",
    "Albert Einstein won the Nobel Prize in Physics.",
]

index = load_index()
verifier = load_verifier("models/with_evidence")
llm = load_llm()

def fact_check(claim):
    found = search(index, [claim], k=TOP_K)[0]                            # 1. search for evidence
    passages = [index["texts"][i] for i in found]
    probs = predict(verifier, [claim], [" ".join(passages)], MAX_LEN)[0]  # 2. verify
    verdict = LABELS[int(probs.argmax())]
    why = explain(llm, claim, verdict, passages)                          # 3. explain
    return {"claim": claim, "verdict": verdict, "confidence": round(float(probs.max()), 2),
            "explanation": why, "evidence": passages}

for claim in MY_CLAIMS:
    r = fact_check(claim)
    print(f"\nCLAIM   : {r['claim']}\nVERDICT : {r['verdict']} (confidence {r['confidence']})\nWHY     : {r['explanation']}")
    for i, p in enumerate(r["evidence"][:3]):
        print(f"   [{i + 1}] {p[:150]}")

# Explanations for 30 FEVER dev claims + the citation check
dev = load_claims("dev", N_DEV).head(30)
saved = [fact_check(c) for c in dev.claim]
ok = [citations_ok(r["explanation"], len(r["evidence"])) for r in saved]
print(f"\nExplanations that cite real evidence numbers: {100 * np.mean(ok):.0f}%")
pd.DataFrame([{k: v for k, v in r.items() if k != "evidence"} for r in saved]).to_csv("outputs/explanations.csv", index=False)
print("Step 7 finished. All done!")
