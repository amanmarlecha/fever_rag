# Fact-checking with Search + AI (simple version)

A claim goes in. The program **searches Wikipedia** for evidence, **decides** SUPPORTS / REFUTES / NOT ENOUGH INFO, and **explains** its decision.

```
claim  ->  SEARCH (keyword + meaning)  ->  top 5 sentences  ->  MODEL picks a label  ->  LLM explains
```

## The files (run them in number order)

| File | What it does, in plain words |
|---|---|
| `helpers.py` | All the small tools (functions). The settings you may change are at the **top**. |
| `1_download_data.py` | Downloads FEVER claims, the Wikipedia text, and the LIAR statements. |
| `2_build_library.py` | Cuts Wikipedia into sentences (the "library") and builds the search index. |
| `3_test_search.py` | Checks how good the search is (precision, recall). Compares 3 search types. |
| `4_train_models.py` | Trains model **A** (claim only) and model **B** (claim + evidence). |
| `5_evaluate_fever.py` | Accuracy, FEVER score, confusion matrix, and **error analysis**. |
| `6_test_liar.py` | Tests on political statements it never saw (out-of-domain). |
| `7_explain_demo.py` | The full pipeline with the LLM explanation. Type your own claims. |
| `8_compare_models.py` | **Comparison:** our model vs old-style models vs published research. Makes a table and a chart. |

## Run it in VS Code (Windows)

1. Open this folder in VS Code and click **Trust**. Open a terminal (`Terminal -> New Terminal`).
2. **Easiest: re-use your old environment** (it already has everything installed). Run its activate file, for example:
   ```
   D:\Mtech\MLAD\Project\fake_news_rag_project\fake_news_rag_project\venv\Scripts\Activate.ps1
   ```
   Or make a new one: `python -m venv venv`, `venv\Scripts\Activate.ps1`,
   `pip install torch --index-url https://download.pytorch.org/whl/cu121`, then `pip install -r requirements.txt`.
3. **Save a 2 GB download:** copy the folder `data\raw` from your old project into this project's `data` folder.
4. Run the steps one by one:
   ```
   python 1_download_data.py
   python 2_build_library.py
   python 3_test_search.py
   python 4_train_models.py
   python 5_evaluate_fever.py
   python 6_test_liar.py
   python 7_explain_demo.py
   python 8_compare_models.py
   ```
5. Open the `outputs` folder: the `.csv` tables and `.png` plots are your results.

Approximate time on a 4 GB GPU: step 2 about 10 min, step 4 about 40-60 min, the rest a few minutes each.

## The comparison (step 8) in plain words

All nine models below answer the **same dev claims** using the **same library**, so the comparison is fair.

| # | Model | What it is |
|---|---|---|
| 1 | Always guess one label | The laziest possible model. It shows what "no skill" scores. |
| 2 | Old-style, claim only | Counts words (TF-IDF) and uses logistic regression. No search, no deep learning. |
| 3 | Old-style pipeline | Keyword search + word counts + logistic regression, in the spirit of the 2018 FEVER baseline. |
| 4 | Pretrained model, no fine-tuning | The same AI model we start from, but **not** taught our task. Shows what our training adds. |
| 5 | Ours A | Fine-tuned model that sees the claim only. |
| 6 | Ours B + keyword search | Our model with keyword search only. |
| 7 | Ours B + meaning search | Our model with meaning search only. |
| 8 | **Ours B + hybrid search** | **Our main model.** |
| 9 | Ours B + perfect evidence | The upper limit if the search never failed. |

Step 8 also prints **what each idea adds** (deep learning, fine-tuning, evidence, hybrid search) in accuracy points.

It then lists **published research results** next to ours. The paper numbers were copied from these sources:
FEVER 2018 baseline (arXiv:1803.05355) and the dev-set table in the KGAT paper (arXiv:1910.09796), which lists UKP-Athene, UNC NLP, GEAR and KGAT.

**Important:** the published systems search *all* of Wikipedia (5.4 million pages). Our library is much smaller, so our numbers look better than they would at full scale. Say this clearly in your report. The fair claims are: *"our model beats old-style models under the same conditions"* and *"evidence and hybrid search each help"*. Do **not** claim *"our model beats published research"*.

Results are saved in `outputs/`: `comparison_same_setting.csv`, `comparison_published.csv`, `comparison_chart.png`.

## If the GPU runs out of memory
In `helpers.py` set `BATCH_SIZE = 4` and `MAX_LEN = 160`, then run step 4 again.

## Words used in the code

* **Claim** - a sentence we want to check. **Evidence** - sentences from Wikipedia that prove or disprove it.
* **Keyword search (TF-IDF)** - counts shared words; rare words count more than common ones.
* **Meaning search (embeddings)** - a model turns every sentence into 384 numbers; sentences with similar meaning get similar numbers.
* **Hybrid** - we add the two scores (after putting them on the same scale).
* **Precision** - of the sentences we found, how many were correct. **Recall** - of the correct sentences, how many we found.
* **Fine-tuning** - taking a model that already knows English and teaching it our task with examples.
* **Epoch** - one full pass over the training claims. **Batch** - claims processed together. **Loss** - how wrong the model is (smaller is better).
* **FEVER score** - the label must be right **and** the proof must have been found by the search.

## Honest limits
The library holds only a part of Wikipedia, so the task is easier than the official FEVER challenge. LIAR is political, which Wikipedia rarely covers, so model B often says NOT ENOUGH INFO there.

## Working together with GitHub
The datasets, trained models and results are **not** stored on GitHub (they are too big). Every teammate runs steps 1-8 on their own computer. See the team guide you received for the Git commands.
