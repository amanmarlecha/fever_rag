
"""
Step 1: Download FEVER and LIAR datasets.

FEVER train/dev files are already included in this GitHub repository.
Missing files are downloaded from their original sources.

The Wikipedia archive and LIAR dataset are downloaded locally,
not committed to this GitHub repository.
"""

import os
import zipfile
import requests
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
FEVER_DIR = RAW_DIR / "fever"
LIAR_DIR = RAW_DIR / "liar"

# Official dataset download sources
FILES = [
    (
        "https://fever.ai/download/fever/train.jsonl",
        FEVER_DIR / "train.jsonl"
    ),
    (
        "https://fever.ai/download/fever/shared_task_dev.jsonl",
        FEVER_DIR / "shared_task_dev.jsonl"
    ),
    (
        "https://fever.ai/download/fever/wiki-pages.zip",
        FEVER_DIR / "wiki-pages.zip"
    ),
    (
        "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        LIAR_DIR / "liar_dataset.zip"
    ),
]


def download_file(url, destination):

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Skip files that already exist
    if destination.exists() and destination.stat().st_size > 0:
        print(f"\nAlready downloaded: {destination.name}")
        return

    temp_file = Path(str(destination) + ".part")

    print(f"\nDownloading: {destination.name}")
    print(f"Source: {url}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 120),
            headers={"User-Agent": "Mozilla/5.0"}
        ) as response:

            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(temp_file, "wb") as file:

                for chunk in response.iter_content(chunk_size=1024 * 1024):

                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        if total_size:
                            percentage = downloaded / total_size * 100
                            print(
                                f"\rProgress: {percentage:.1f}% "
                                f"({downloaded / (1024**2):.1f} MB)",
                                end=""
                            )

        if total_size and downloaded != total_size:
            raise RuntimeError("Downloaded file size does not match.")

        os.replace(temp_file, destination)

        print(f"\nDownload completed: {destination.name}")

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


def safe_extract(zip_path, destination):

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Invalid ZIP archive: {zip_path}")

    destination.mkdir(parents=True, exist_ok=True)

    # Prevent archive paths from escaping the destination folder
    root = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:

        for member in archive.infolist():
            target = (destination / member.filename).resolve()

            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise RuntimeError("Unsafe path detected in ZIP archive.")

        print(f"\nExtracting: {zip_path.name}")
        archive.extractall(destination)

    print(f"Extraction completed: {zip_path.name}")


def extract_datasets():

    # Extract FEVER Wikipedia only if its files are missing
    wiki_files = list(FEVER_DIR.rglob("wiki-*.jsonl"))

    if not wiki_files:
        wiki_zip = FEVER_DIR / "wiki-pages.zip"

        if wiki_zip.exists():
            safe_extract(wiki_zip, FEVER_DIR)
        else:
            raise FileNotFoundError("FEVER Wikipedia ZIP is missing.")

    else:
        print("\nWikipedia dataset already extracted.")

    # Extract LIAR only if required TSV files are missing
    required_liar_files = ["train.tsv", "test.tsv", "valid.tsv"]

    if not all((LIAR_DIR / name).exists() for name in required_liar_files):

        liar_zip = LIAR_DIR / "liar_dataset.zip"

        if liar_zip.exists():
            safe_extract(liar_zip, LIAR_DIR)
        else:
            raise FileNotFoundError("LIAR dataset ZIP is missing.")

    else:
        print("\nLIAR dataset already extracted.")


def main():

    print("=" * 55)
    print("FEVER + LIAR DATASET DOWNLOAD")
    print("=" * 55)

    # Download missing files
    for url, destination in FILES:
        download_file(url, destination)

    # Extract archives
    extract_datasets()

    print("\n" + "=" * 55)
    print("STEP 1 COMPLETED SUCCESSFULLY!")
    print("FEVER and LIAR datasets are ready.")
    print("Next: Step 2 - Build Library")
    print("=" * 55)


if __name__ == "__main__":
    main()
