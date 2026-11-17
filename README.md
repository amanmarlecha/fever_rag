# Fact-Checking with Search + AI (FEVER RAG)

A claim goes in. The program searches Wikipedia for evidence, predicts whether the claim is SUPPORTS, REFUTES, or NOT ENOUGH INFO, and explains its decision.

```text
Claim
  ↓
Search (Keyword + Meaning)
  ↓
Retrieve Top 5 Evidence Sentences
  ↓
Model Predicts a Label
  ↓
LLM Explains the Decision
```

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based fact-checking system using the FEVER and LIAR datasets.

The system combines information retrieval, machine learning, deep learning, and language models to evaluate factual claims.

## 2. Project Files

Run the Python files in numerical order.

| File | Description |
|---|---|
| `helpers.py` | Common functions and project settings |
| `1_download_data.py` | Downloads and extracts FEVER, Wikipedia and LIAR datasets |
| `2_build_library.py` | Processes Wikipedia into sentences and builds the search index |
| `3_test_search.py` | Evaluates search performance using precision and recall |
| `4_train_models.py` | Trains Model A (claim only) and Model B (claim + evidence) |
| `5_evaluate_fever.py` | Evaluates accuracy, FEVER score, confusion matrix and errors |
| `6_test_liar.py` | Tests the model on LIAR political statements |
| `7_explain_demo.py` | Runs the complete fact-checking pipeline with explanations |
| `8_compare_models.py` | Compares our models with traditional approaches and published research |

## 3. Datasets Used

### FEVER Dataset

FEVER (Fact Extraction and VERification) contains claims that need to be verified using Wikipedia evidence.

GitHub repository includes:
- `train.jsonl`
- `shared_task_dev.jsonl`

The Wikipedia archive is downloaded separately because of its large size.

Official source: https://fever.ai/dataset/fever.html

### LIAR Dataset

LIAR contains statements labelled for factuality classification. It is used to test the system on political statements outside the original FEVER task.

The LIAR dataset is downloaded from its original UCSB source.

Download source: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

**Important:** The LIAR dataset is provided for research purposes. Its original sources retain copyright. Please consult its documentation before redistribution.

### Dataset Download Policy

| Dataset/File | Storage |
|---|---|
| FEVER training file | GitHub repository |
| FEVER development file | GitHub repository |
| FEVER Wikipedia archive | Downloaded locally |
| Extracted Wikipedia files | Local computer |
| LIAR dataset | Downloaded locally |
| Trained models | Local computer |
| Generated embeddings | Local computer |
| Evaluation outputs | Local computer |

The `.gitignore` file prevents large datasets, models and generated outputs from being uploaded accidentally.

## 4. Project Setup (Windows + VS Code)

### Step 1: Clone the GitHub repository

```powershell
git clone https://github.com/amanmarlecha/fever_rag.git
cd fever_rag
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

If dependencies are not available in the requirements file, install the required packages according to the project configuration.

### Step 4: Download datasets

Run:

```powershell
python 1_download_data.py
```

This script:
- Uses the FEVER training and development files included in the repository.
- Downloads the Wikipedia archive if it is missing.
- Downloads the LIAR dataset if it is missing.
- Extracts the required archives.
- Skips files that are already available locally.

**Note:** The Wikipedia archive is approximately 1.6 GB. Ensure sufficient internet bandwidth and disk space for downloading and extracting it.

## 5. Running the Project

After downloading and preparing the datasets, execute the Python scripts in the following order:

```powershell
python 1_download_data.py

python 2_build_library.py

python 3_test_search.py

python 4_train_models.py

python 5_evaluate_fever.py

python 6_test_liar.py

python 7_explain_demo.py

python 8_compare_models.py
```

### What each step produces

1. Dataset preparation.
2. Wikipedia sentence library and search index.
3. Search evaluation results.
4. Trained machine learning models.
5. FEVER evaluation metrics and error analysis.
6. LIAR out-of-domain evaluation.
7. Interactive fact-checking demonstration.
8. Model comparison tables and charts.

## 6. Model Comparison

All nine models answer the same development claims using the same evidence library.

| No. | Model | Description |
|---|---|---|
| 1 | Always guess one label | Baseline with no learning |
| 2 | Old-style, claim only | TF-IDF + Logistic Regression |
| 3 | Old-style pipeline | Keyword search + Logistic Regression |
| 4 | Pretrained model | Model without task-specific fine-tuning |
| 5 | Ours A | Fine-tuned model using the claim only |
| 6 | Ours B + keyword search | Uses keyword-based evidence |
| 7 | Ours B + meaning search | Uses semantic evidence |
| 8 | Ours B + hybrid search | Combines keyword and semantic search |
| 9 | Ours B + perfect evidence | Evaluates performance with correct evidence |

The comparison also examines the contributions of:
- Deep learning
- Fine-tuning
- Evidence retrieval
- Hybrid search

### Published Research Comparison

Published research results are compared with our models using:
- FEVER 2018 baseline
- KGAT research paper
- UKP-Athene
- UNC NLP
- GEAR
- KGAT

**Important limitation:** Published systems search all of Wikipedia (approximately 5.4 million pages), whereas our project uses a smaller Wikipedia library.

Therefore, the results should not be presented as proof that our model outperforms published research under identical conditions.

Our fair comparison is between models evaluated under the same project conditions.

Results are saved in the `outputs/` directory:
- `comparison_same_setting.csv`
- `comparison_published.csv`
- `comparison_chart.png`

## 7. Troubleshooting

### If the GPU runs out of memory

In `helpers.py`, adjust:

```python
BATCH_SIZE = 4
MAX_LEN = 160
```

Then run the training script again:

```powershell
python 4_train_models.py
```

### If datasets are missing

Run:

```powershell
python 1_download_data.py
```

### If Python packages are missing

Activate your virtual environment and run:

```powershell
pip install -r requirements.txt
```

## 8. Important Terminology

- **Claim:** A statement that needs fact-checking.
- **Evidence:** Information retrieved from Wikipedia to support or refute a claim.
- **Keyword Search (TF-IDF):** Finds relevant text based on shared words.
- **Meaning Search (Embeddings):** Finds sentences based on semantic similarity.
- **Hybrid Search:** Combines keyword and meaning-based search.
- **Precision:** How many retrieved sentences are relevant.
- **Recall:** How many relevant sentences were successfully retrieved.
- **Fine-tuning:** Adapting a pretrained model to a specific task.
- **Epoch:** One complete pass through the training dataset.
- **Batch:** A group of examples processed together.
- **Loss:** A measure of model prediction error.
- **FEVER Score:** Measures whether the predicted label is correct and the required evidence is retrieved.

## 9. Honest Project Limitations

- The Wikipedia library contains only a subset of the complete Wikipedia collection.
- The LIAR dataset contains political statements, which may not have sufficient Wikipedia coverage.
- Model B may predict NOT ENOUGH INFO when relevant evidence cannot be retrieved.
- Performance on this project should not be directly compared with systems searching the complete Wikipedia collection without considering the difference in evidence coverage.

## 10. Working Together with GitHub

The GitHub repository contains:
- Python source code.
- Project documentation.
- FEVER training and development datasets.
- Requirements and setup instructions.

Large Wikipedia files, the LIAR dataset, trained models, embeddings and generated results are maintained locally.

Every teammate can clone the repository and execute:

```powershell
python 1_download_data.py
```

This prepares the required datasets from the included FEVER files and their original download sources.

### Git Commands

To get the latest project changes:

```powershell
git pull origin main
```

To upload your own code changes:

```powershell
git add .
git commit -m "Describe your changes"
git push origin main
```

**Repository:** https://github.com/amanmarlecha/fever_rag
