"""helpers.py - all the small tools used by the numbered step files. Settings are at the top."""
import os, re, json, glob, math, random, zlib, unicodedata
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")                      # draw plots into files (works on any computer)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, get_linear_schedule_with_warmup)

# ---------------- 1. SETTINGS (the only numbers you may want to change) ----------------
N_TRAIN = 12000            # how many claims we use to teach the model (4 GB GPU friendly)
N_DEV = 2000               # how many claims we use to test the model
N_DISTRACTOR_PAGES = 20000 # extra random Wikipedia pages added to the library (makes searching harder)
TOP_K = 5                  # how many evidence sentences we give to the model
KEYWORD_WEIGHT = 0.5       # hybrid search = 50% keyword score + 50% meaning score
EPOCHS = 3                 # how many times the model reads the whole training set
BATCH_SIZE = 8             # claims processed at once (lower this if the GPU runs out of memory)
LEARNING_RATE = 2e-5       # size of each learning step
MAX_LEN = 192              # max number of tokens (word pieces) for claim + evidence
CLAIM_MAX_LEN = 48         # max tokens when the model sees the claim only
SEED = 42                  # makes random choices repeatable

# Model names (all downloaded automatically from the internet the first time)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                 # turns a sentence into numbers that capture its MEANING
VERIFIER_BASE = "cross-encoder/nli-deberta-v3-small"   # a model that already knows "does A support B?"
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"                # small chatbot that writes the explanations

LABELS = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]    # label number 0, 1, 2
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
print("Running on:", DEVICE)


# ======================================================================
# 2. READING THE DATA
# ======================================================================
def read_jsonl(path):
    """Read a file where every line is one JSON record."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_text(text):
    """FEVER writes brackets as codes (-LRB-). Change them back to normal symbols."""
    text = text.replace("_", " ")
    for code, symbol in [("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-COLON-", ":")]:
        text = text.replace(code, symbol)
    return text.strip()


def fix_title(title):
    """Some titles have accents that can be typed in two ways. Make them all the same way."""
    return unicodedata.normalize("NFD", title)


def load_claims(split, n):
    """Load n FEVER claims (split = 'train' or 'dev'), the same number from each label.
    Each claim has: the text, the label, and the gold evidence (the sentences that prove it)."""
    file_name = "train.jsonl" if split == "train" else "shared_task_dev.jsonl"
    rows = list(read_jsonl(f"data/raw/fever/{file_name}"))
    random.Random(SEED).shuffle(rows)                      # mix the claims up
    per_label = n // 3                                     # equal number of SUPPORTS / REFUTES / NOT ENOUGH INFO
    count = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
    seen, claims = set(), []
    for row in rows:
        label = row["label"]
        if label not in count or count[label] >= per_label:
            continue
        claim = clean_text(row["claim"])
        if claim in seen:
            continue
        seen.add(claim)
        count[label] += 1
        gold_groups = []                                   # each group = one complete set of proving sentences
        if label != "NOT ENOUGH INFO":
            for group in row["evidence"]:
                sentences = [(fix_title(e[2]), int(e[3])) for e in group if e[2] is not None]
                if sentences and sentences not in gold_groups:
                    gold_groups.append(sentences)
        claims.append({"claim": claim, "label": label, "y": LABELS.index(label), "gold_groups": gold_groups})
    return pd.DataFrame(claims)


def load_liar():
    """Load the LIAR test set (political statements) and turn its 6 ratings into SUPPORTS / REFUTES."""
    columns = ["id", "label", "statement", "subject", "speaker", "job", "state", "party", "c1", "c2", "c3", "c4", "c5", "context"]
    df = pd.read_csv("data/raw/liar/test.tsv", sep="\t", header=None, names=columns, quoting=3)
    rating_to_label = {"true": "SUPPORTS", "mostly-true": "SUPPORTS",
                       "false": "REFUTES", "barely-true": "REFUTES", "pants-fire": "REFUTES"}   # "half-true" is skipped
    df = df[df.label.isin(rating_to_label)].copy()
    df["gold"] = df.label.map(rating_to_label)
    df["claim"] = df.statement.astype(str).str.strip()
    return df.reset_index(drop=True)


# ======================================================================
# 3. THE EVIDENCE LIBRARY (corpus) AND THE SEARCH ENGINE
# ======================================================================
def build_corpus(train, dev):
    """Make data/corpus.jsonl = one line per Wikipedia sentence.
    We keep (a) every page that holds an answer for our claims, (b) some random extra pages as distractors."""
    answer_pages = set()
    for df in (train, dev):
        for groups in df.gold_groups:
            for group in groups:
                for title, sent_number in group:
                    answer_pages.add(title)
    chance = N_DISTRACTOR_PAGES / 5_416_537                # Wikipedia dump has 5.4 million pages
    wiki_files = sorted(glob.glob("data/raw/fever/**/wiki-*.jsonl", recursive=True))
    if not wiki_files:
        raise FileNotFoundError("Wikipedia files not found. Run step 1 first.")
    total = 0
    with open("data/corpus.jsonl", "w", encoding="utf-8") as out:
        for file in wiki_files:
            for page in read_jsonl(file):
                title = page["id"]
                is_answer_page = fix_title(title) in answer_pages
                is_random_page = (zlib.crc32(title.encode("utf-8")) % 1_000_000) / 1_000_000 < chance
                if not (is_answer_page or is_random_page):
                    continue
                for line in page["lines"].split("\n"):     # every line looks like: number <TAB> sentence <TAB> links
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
                        record = {"title": fix_title(title), "sent": int(parts[0]), "text": parts[1].strip()}
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total += 1
    print(f"Library built: {total} sentences. Answer pages needed: {len(answer_pages)}")


_embedder = {}                                              # remembers the loaded model so we load it only once

def embed(texts, is_query=False):
    """Turn each text into a list of 384 numbers (its 'meaning vector'). Similar meaning = similar numbers."""
    if "model" not in _embedder:
        _embedder["tok"] = AutoTokenizer.from_pretrained(EMBED_MODEL)
        model = AutoModel.from_pretrained(EMBED_MODEL)
        if USE_GPU:
            model = model.half()
        _embedder["model"] = model.to(DEVICE).eval()
    tok, model = _embedder["tok"], _embedder["model"]
    if is_query:                                            # this model likes a short instruction before search questions
        texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
    vectors = []
    for start in range(0, len(texts), 64):
        batch = tok(texts[start:start + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            first_token = model(**batch).last_hidden_state[:, 0]       # the first token summarises the sentence
        first_token = torch.nn.functional.normalize(first_token.float(), dim=1)   # make every vector length 1
        vectors.append(first_token.cpu().numpy())
        if (start // 64) % 100 == 0:
            print(f"  embedded {start}/{len(texts)}")
    return np.concatenate(vectors)


def load_index():
    """Load the library and prepare BOTH search tools:
       (1) keyword search = TF-IDF (counts shared words, rare words count more)
       (2) meaning search = embeddings (compares meaning vectors)"""
    passages = list(read_jsonl("data/corpus.jsonl"))
    texts = [clean_text(p["title"]) + ": " + p["text"] for p in passages]      # "Paris: Paris is the capital..."
    key_to_id = {(p["title"], p["sent"]): i for i, p in enumerate(passages)}   # (page, sentence number) -> position
    path = "data/embeddings.npy"
    if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == len(texts):
        embeddings = np.load(path)                                             # already computed earlier
    else:
        print("Computing meaning vectors for the whole library (done once, then saved)...")
        embeddings = embed(texts)
        np.save(path, embeddings)
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float32)
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"Library ready: {len(texts)} sentences")
    return {"texts": texts, "key_to_id": key_to_id, "embeddings": embeddings,
            "vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix}


def standardize(scores):
    """Put scores on the same scale (average 0, spread 1) so keyword and meaning scores can be added fairly."""
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)


def top_ids(scores, k):
    """For every row, return the positions of the k highest scores, best first."""
    best = np.argpartition(-scores, k, axis=1)[:, :k]                       # the k best (not yet sorted)
    order = np.argsort(-np.take_along_axis(scores, best, axis=1), axis=1)   # sort those k, highest first
    return np.take_along_axis(best, order, axis=1)


def search(index, queries, k=5, mode="hybrid"):
    """Find the k best evidence sentences for each claim. mode = 'keyword', 'meaning' or 'hybrid'."""
    found = []
    for start in range(0, len(queries), 100):                               # 100 claims at a time to save memory
        batch = queries[start:start + 100]
        keyword_scores = meaning_scores = None                              # only compute what this mode needs
        if mode in ("keyword", "hybrid"):
            keyword_scores = (index["vectorizer"].transform(batch) @ index["tfidf_matrix"].T).toarray()
        if mode in ("meaning", "hybrid"):
            meaning_scores = embed(batch, is_query=True) @ index["embeddings"].T
        if mode == "keyword":
            scores = keyword_scores
        elif mode == "meaning":
            scores = meaning_scores
        else:
            scores = KEYWORD_WEIGHT * standardize(keyword_scores) + (1 - KEYWORD_WEIGHT) * standardize(meaning_scores)
        found.extend(top_ids(scores, k).tolist())
    return found


def passages_to_text(index, ids):
    """Glue the chosen evidence sentences into one piece of text."""
    return " ".join(index["texts"][i] for i in ids)


# ======================================================================
# 4. MEASURING HOW GOOD THE SEARCH IS
# ======================================================================
def gold_to_ids(df, key_to_id):
    """Change each claim's gold evidence from (page, sentence) pairs into positions in our library."""
    result = []
    for groups in df.gold_groups:
        id_groups = []
        for group in groups:
            ids = [key_to_id.get(key) for key in group]
            if None not in ids:                                             # skip sets that are not fully in our library
                id_groups.append(ids)
        result.append(id_groups)
    return result


def complete_evidence_found(groups, ids):
    """True if at least one COMPLETE proving set is inside the retrieved sentences."""
    return any(set(group) <= set(ids) for group in groups)


def retrieval_scores(gold, retrieved, k):
    """precision = share of retrieved sentences that are correct; recall = share of correct sentences we found."""
    precision, recall, found_all = [], [], []
    for groups, ids in zip(gold, retrieved):
        if not groups:
            continue                                                        # NOT ENOUGH INFO claims have no answer sentences
        top = set(ids[:k])
        answers = {i for group in groups for i in group}
        good = len(top & answers)
        precision.append(good / k)
        recall.append(good / len(answers))
        found_all.append(complete_evidence_found(groups, ids[:k]))
    return {"precision": np.mean(precision), "recall": np.mean(recall), "found_all": np.mean(found_all)}


def fever_score(pred, y, gold, retrieved):
    """FEVER score: the label must be right AND (for SUPPORTS/REFUTES) a complete proof must have been retrieved."""
    points = []
    for p, truth, groups, ids in zip(pred, y, gold, retrieved):
        if p != truth:
            points.append(False)
        elif LABELS[truth] == "NOT ENOUGH INFO":
            points.append(True)
        else:
            points.append(complete_evidence_found(groups, ids))
    return float(np.mean(points))


# ======================================================================
# 5. THE VERIFIER MODEL (reads claim + evidence, picks a label)
# ======================================================================
def make_training_evidence(df, retrieved, gold, index):
    """Evidence the model sees while learning:
       SUPPORTS / REFUTES -> the real proof + some searched sentences as 'noise'
       NOT ENOUGH INFO    -> just the searched sentences"""
    rng = random.Random(SEED)
    texts = []
    for label, found, groups in zip(df.label, retrieved, gold):
        if label == "NOT ENOUGH INFO" or not groups:
            ids = list(found[:TOP_K])
        else:
            ids = list(groups[0])[:TOP_K]                                   # the real proof first
            extra = [i for i in found if i not in ids]                      # searched sentences that are not the proof
            ids += extra[:TOP_K - len(ids)]
        rng.shuffle(ids)                                                    # shuffle so the position gives nothing away
        texts.append(passages_to_text(index, ids))
    return texts


def encode_batch(tok, claims, evidences, max_len):
    """Turn text into numbers (tokens) the model can read."""
    if evidences is None:                                                   # baseline: claim only
        batch = tok(list(claims), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    else:                                                                   # claim + evidence as a pair
        batch = tok(list(claims), list(evidences), truncation="only_second", max_length=max_len, padding=True, return_tensors="pt")
    return batch.to(DEVICE)


def train_model(claims, evidences, labels, save_dir, max_len):
    """Teach the model. This is the classic training loop: predict -> measure mistake -> adjust -> repeat."""
    tok = AutoTokenizer.from_pretrained(VERIFIER_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(VERIFIER_BASE, num_labels=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)      # the thing that adjusts the model
    claims, labels = list(claims), np.array(labels)
    evidences = list(evidences) if evidences is not None else None
    steps_total = EPOCHS * math.ceil(len(claims) / BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps_total), steps_total)   # learning rate: up, then down
    scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)                   # helps fp16 (half precision) training on GPU
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        order = np.random.permutation(len(claims))                           # new random order each epoch
        running_loss = 0.0
        for start in range(0, len(claims), BATCH_SIZE):
            pick = order[start:start + BATCH_SIZE]
            batch = encode_batch(tok, [claims[i] for i in pick],
                                 None if evidences is None else [evidences[i] for i in pick], max_len)
            answers = torch.tensor(labels[pick]).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                loss = model(**batch, labels=answers).loss                   # how wrong was the model? (smaller = better)
            scaler.scale(loss).backward()                                    # work out how to fix the mistakes
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # avoid huge jumps
            scaler.step(optimizer)                                           # adjust the model a little
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch + 1}/{EPOCHS}  step {step}/{steps_total}  loss {running_loss / 100:.3f}")
                running_loss = 0.0
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print("Saved model to", save_dir)
    del model, optimizer
    if USE_GPU:
        torch.cuda.empty_cache()                                             # give GPU memory back


def load_verifier(save_dir):
    """Load a trained model from disk."""
    tok = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(DEVICE).eval()
    return tok, model


def predict(verifier, claims, evidences, max_len):
    """Return, for each claim, 3 probabilities: [REFUTES, SUPPORTS, NOT ENOUGH INFO]."""
    tok, model = verifier
    claims = list(claims)
    evidences = list(evidences) if evidences is not None else None
    result = []
    for start in range(0, len(claims), 32):
        batch = encode_batch(tok, claims[start:start + 32],
                             None if evidences is None else evidences[start:start + 32], max_len)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
            logits = model(**batch).logits
        result.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(result)


# ======================================================================
# 6. THE LLM THAT WRITES THE EXPLANATION
# ======================================================================
def load_llm():
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME)
    if USE_GPU:
        model = model.half()
    return tok, model.to(DEVICE).eval()


def explain(llm, claim, verdict, passages):
    """Ask the LLM to explain the verdict using ONLY the evidence, citing it as [1], [2] ..."""
    tok, model = llm
    evidence = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    question = (f"You are a careful fact-checker. Use ONLY the evidence below.\n\nClaim: {claim}\nVerdict: {verdict}\n\n"
                f"Evidence:\n{evidence}\n\nWrite 1-2 sentences explaining why the evidence leads to this verdict. "
                "Cite evidence numbers like [1]. If the verdict is NOT ENOUGH INFO, say what is missing.")
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def citations_ok(explanation, n_passages):
    """Simple check: the explanation cites at least one [n], and every n is a real evidence number."""
    numbers = [int(n) for n in re.findall(r"\[(\d+)\]", explanation)]
    return len(numbers) > 0 and all(1 <= n <= n_passages for n in numbers)


# ======================================================================
# 7. COMPARISON TOOLS: "old-style" models to compare our model against
# ======================================================================
def word_overlap(claims, evidences):
    """What share of the claim's words also appear in the evidence? (a simple hint that the claim is supported)"""
    shares = []
    for claim, evidence in zip(claims, evidences):
        claim_words = set(re.findall(r"[a-z0-9]+", claim.lower()))
        evidence_words = set(re.findall(r"[a-z0-9]+", evidence.lower()))
        shares.append(len(claim_words & evidence_words) / max(1, len(claim_words)))
    return csr_matrix(np.array(shares).reshape(-1, 1))


def classic_features(vectors, claims, evidences):
    """Turn text into numbers the old way: counting words (TF-IDF). Evidence words + overlap are added if we have evidence."""
    claim_vectorizer, evidence_vectorizer = vectors
    parts = [claim_vectorizer.transform(claims)]
    if evidences is not None:
        parts += [evidence_vectorizer.transform(evidences), word_overlap(claims, evidences)]
    return hstack(parts).tocsr()


def fit_classic(claims, evidences, labels):
    """Train an old-style (not deep learning) classifier: word counts + logistic regression."""
    claim_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    claim_vectorizer.fit(claims)
    evidence_vectorizer = None
    if evidences is not None:
        evidence_vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=50000)
        evidence_vectorizer.fit(evidences)
    vectors = (claim_vectorizer, evidence_vectorizer)
    model = LogisticRegression(max_iter=1000)
    model.fit(classic_features(vectors, list(claims), evidences), labels)
    return vectors, model
