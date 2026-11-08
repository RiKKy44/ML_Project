
import csv
import tarfile
import shutil
import requests
from pathlib import Path

languages = ["pl_pl", "tr_tr", "en_us"]
short_names = {"pl_pl": "pl", "tr_tr": "tr", "en_us": "en"}
n_recordings = 60

data_dir = Path("data/not_allowed")
download_dir = Path("data/downloads")

data_dir.mkdir(parents=True, exist_ok=True)
download_dir.mkdir(parents=True, exist_ok=True)

metadata = []

for lang in languages:
    archive = download_dir / f"{lang}_dev.tar.gz"

    if not archive.exists():
        url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{lang}/audio/dev.tar.gz"

        print(f"Downloading {lang}...")
        response = requests.get(url, timeout=180)
        response.raise_for_status()

        with open(archive, "wb") as f:
            f.write(response.content)

    print(f"Extracting {lang}...")

    with tarfile.open(archive, "r:gz") as tar:
        wav_files = [
            file for file in tar.getmembers()
            if file.isfile() and file.name.endswith(".wav")
        ]

        for i, file in enumerate(wav_files[:n_recordings], start=1):
            filename = f"{short_names[lang]}_{i:03d}.wav"
            path = data_dir / filename

            with tar.extractfile(file) as source:
                with open(path, "wb") as output:
                    output.write(source.read())

            metadata.append([
                filename, lang, 0, "FLEURS", file.name
            ])

with open(data_dir / "metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "language", "class", "source", "original_file"])
    writer.writerows(metadata)

