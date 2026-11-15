from helpers import *

# STEP 1 - Download the datasets (FEVER, Wikipedia, LIAR). Files that already exist are skipped.
import requests, zipfile

FILES = [
    ("https://fever.ai/download/fever/train.jsonl", "data/raw/fever/train.jsonl"),
    ("https://fever.ai/download/fever/shared_task_dev.jsonl", "data/raw/fever/shared_task_dev.jsonl"),
    ("https://fever.ai/download/fever/wiki-pages.zip", "data/raw/fever/wiki-pages.zip"),       # about 1.6 GB
    ("https://www.cs.ucsb.edu/~william/data/liar_dataset.zip", "data/raw/liar/liar_dataset.zip"),
]

def download(url, path):
    if os.path.exists(path):
        print("already have", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("downloading", url, "(please wait)")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    os.replace(path + ".part", path)                       # only rename when the download is complete

for url, path in FILES:
    download(url, path)

if not glob.glob("data/raw/fever/**/wiki-*.jsonl", recursive=True):
    print("unzipping Wikipedia ...")
    zipfile.ZipFile("data/raw/fever/wiki-pages.zip").extractall("data/raw/fever")
if not os.path.exists("data/raw/liar/test.tsv"):
    zipfile.ZipFile("data/raw/liar/liar_dataset.zip").extractall("data/raw/liar")
print("Step 1 finished. Next: step 2")
