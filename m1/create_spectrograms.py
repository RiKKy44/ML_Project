"""Convert WAV chunks into fixed-size log-Mel spectrograms saved as .npy files.

Input folders are outputs of smart_chunk_wav.py (WAV clips + manifest.csv).
Every chunk is loaded as 16 kHz mono audio, padded with silence to a fixed
duration (3 s by default), turned into a Mel spectrogram and converted to dB.
All spectrograms therefore have the same shape: (n_mels, frames) = (64, 188).

Output folder layout:
    npy/<class>/<input folder name>/<same path as the chunk>.npy
    spectrograms.csv   one row per spectrogram: label + all manifest metadata
    examples/*.pdf     a few example plots per class
    run.json           settings, package versions and summary

Note: source_group (from smart_chunk_wav.py) identifies the original recording
file, not the speaker. Any extra manifest columns (e.g. a speaker column added
by the team) are copied unchanged into spectrograms.csv.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import sys

try:
    import numpy as np
    import soundfile as sf
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")  # Save plots to files; no window needed.
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\nInstall with:\n"
        "python -m pip install numpy soundfile librosa matplotlib"
    ) from exc

CLASS_NAMES = {0: "not_allowed", 1: "allowed"}


def load_chunk_list(folder: Path) -> list[dict]:
    """Return one dict per WAV chunk in a chunk folder.

    Uses manifest.csv written by smart_chunk_wav.py when it exists. Without a
    manifest, every WAV file in the folder is used and the source recording is
    read from the chunk's folder name, which smart_chunk_wav.py writes as
    "<recording name>__<source_group>" (e.g. "speech-01__1a2b3c4d5e6f7a8b").
    """
    manifest = folder / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    print(f"WARNING: no manifest.csv in {folder}; scanning WAV files instead.",
          file=sys.stderr)
    rows = []
    for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".wav"):
        relative = path.relative_to(folder)
        recording = relative.parent
        name, _, group = recording.name.rpartition("__")
        if name and re.fullmatch(r"[0-9a-f]{16}", group):
            source_file = (recording.parent / f"{name}.wav").as_posix()
        else:  # Folder name not in the chunker format: use it as it is.
            source_file, group = recording.as_posix(), recording.name
        rows.append(dict(chunk_file=relative.as_posix(), source_file=source_file,
                         source_group=group, label=""))
    return rows


def load_audio(path: Path, sample_rate: int, num_samples: int) -> tuple[np.ndarray, float]:
    """Load a WAV file as mono audio at `sample_rate`, padded/cut to `num_samples`.

    Returns the fixed-length audio and the original duration in seconds.
    """
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    duration = len(audio) / sample_rate
    audio = librosa.util.fix_length(audio, size=num_samples)  # zero-pads at the end
    return audio, duration


def mel_spectrogram_db(audio: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Compute a Mel spectrogram and convert power to decibels (max = 0 dB)."""
    mel = librosa.feature.melspectrogram(y=audio, sr=args.sample_rate, n_fft=args.n_fft,
                                         hop_length=args.hop_length, n_mels=args.n_mels)
    return librosa.power_to_db(mel, ref=np.max, top_db=args.top_db).astype(np.float32)


def save_example_plot(spec: np.ndarray, title: str, path: Path, args: argparse.Namespace):
    """Save one spectrogram as a PDF figure."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    image = librosa.display.specshow(spec, sr=args.sample_rate, hop_length=args.hop_length,
                                     n_fft=args.n_fft, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(image, ax=ax, format="%+.0f dB")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output", type=Path, help="New or empty output directory")
    parser.add_argument("--allowed", type=Path, nargs="+", default=[],
                        help="Chunk folder(s) with class 1 (allowed) clips")
    parser.add_argument("--not-allowed", type=Path, nargs="+", default=[],
                        help="Chunk folder(s) with class 0 (not allowed) clips")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Fixed input duration in seconds (pad/cut)")
    parser.add_argument("--n-mels", type=int, default=64, help="Number of Mel bands")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length in samples")
    parser.add_argument("--top-db", type=float, default=80.0,
                        help="Lowest dB value kept below the maximum")
    parser.add_argument("--examples", type=int, default=3,
                        help="Example plots per class")
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking examples")
    args = parser.parse_args(argv)
    if not args.allowed and not args.not_allowed:
        parser.error("Give at least one folder with --allowed or --not-allowed.")
    if min(args.sample_rate, args.n_mels, args.n_fft, args.hop_length) <= 0 \
            or not (math.isfinite(args.duration) and args.duration > 0):
        parser.error("Audio and spectrogram settings must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    inputs = [(1, p.expanduser().resolve()) for p in args.allowed] + \
             [(0, p.expanduser().resolve()) for p in args.not_allowed]
    for label, folder in inputs:
        if not folder.is_dir():
            raise SystemExit(f"Input folder does not exist: {folder}")
        if folder == output or output in folder.parents:
            raise SystemExit("Output must not be inside an input folder.")
    names = [(label, folder.name) for label, folder in inputs]
    if len(set(names)) != len(names):
        raise SystemExit("Two input folders of the same class have the same name.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("Output must be a new or empty directory (nothing was overwritten).")
    output.mkdir(parents=True, exist_ok=True)

    num_samples = round(args.duration * args.sample_rate)
    fields = ["npy_file", "label", "class_name", "input_folder", "chunk_file",
              "source_file", "source_group", "duration_seconds", "manifest_duration_seconds",
              "original_sample_rate", "original_channels", "was_cut"]
    chunk_lists = [load_chunk_list(folder) for _, folder in inputs]
    # Keep every other manifest column too (timestamps, speaker info, ...).
    for chunks in chunk_lists:
        for chunk in chunks:
            fields += [key for key in chunk if key not in fields and key is not None]
    summary = {"chunks": 0, "spectrograms": 0, "errors": 0, "cut_to_duration": 0,
               "per_class": {name: 0 for name in CLASS_NAMES.values()}, "shape": None}
    rows_by_class = {0: [], 1: []}
    with (output / "spectrograms.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for (label, folder), chunks in zip(inputs, chunk_lists):
            class_name = CLASS_NAMES[label]
            print(f"{class_name}: {folder} ({len(chunks)} chunks)")
            for chunk in chunks:
                summary["chunks"] += 1
                if chunk.get("label", "") not in ("", str(label)):
                    print(f"ERROR {chunk['chunk_file']}: manifest label {chunk['label']} "
                          f"does not match --{class_name.replace('_', '-')}", file=sys.stderr)
                    summary["errors"] += 1
                    continue
                wav = folder / chunk["chunk_file"]
                try:
                    info = sf.info(str(wav))
                    audio, duration = load_audio(wav, args.sample_rate, num_samples)
                    spec = mel_spectrogram_db(audio, args)
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"ERROR {wav}: {exc}", file=sys.stderr)
                    summary["errors"] += 1
                    continue
                if summary["shape"] is None:
                    summary["shape"] = list(spec.shape)
                elif list(spec.shape) != summary["shape"]:
                    raise SystemExit(f"Unexpected shape {spec.shape} for {wav}")
                npy = Path("npy", class_name, folder.name, chunk["chunk_file"]).with_suffix(".npy")
                (output / npy).parent.mkdir(parents=True, exist_ok=True)
                np.save(output / npy, spec)
                was_cut = duration > args.duration + 1e-6
                row = {key: value for key, value in chunk.items() if key is not None}
                row.update(npy_file=npy.as_posix(), label=label, class_name=class_name,
                           input_folder=str(folder), chunk_file=chunk["chunk_file"],
                           source_file=chunk.get("source_file", ""),
                           source_group=chunk.get("source_group", ""),
                           duration_seconds=round(duration, 4),
                           original_sample_rate=info.samplerate,
                           original_channels=info.channels, was_cut=int(was_cut))
                if "duration_seconds" in chunk:  # keep the chunker's value under its own name
                    row["manifest_duration_seconds"] = chunk["duration_seconds"]
                writer.writerow(row)
                rows_by_class[label].append(row)
                summary["spectrograms"] += 1
                summary["per_class"][class_name] += 1
                summary["cut_to_duration"] += was_cut

    # A few example plots per class, picked reproducibly.
    (output / "examples").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for label, rows in rows_by_class.items():
        picks = rng.choice(len(rows), size=min(args.examples, len(rows)), replace=False)
        for number, index in enumerate(sorted(picks), start=1):
            row = rows[index]
            title = (f"class {label} ({row['class_name']}) - {row['chunk_file']} "
                     f"- {row['duration_seconds']} s")
            save_example_plot(np.load(output / row["npy_file"]), title,
                              output / "examples" / f"{row['class_name']}_{number}.pdf", args)

    record = {
        "script_version": "1.0.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"label": label, "folder": str(folder)} for label, folder in inputs],
        "options": {k: str(v) if isinstance(v, Path) else
                    [str(x) for x in v] if isinstance(v, list) else v
                    for k, v in vars(args).items()},
        "packages": {p: version(p) for p in ["numpy", "soundfile", "librosa", "matplotlib"]},
        "summary": summary,
        "notes": "Values are dB relative to each clip's maximum (0 dB), floored at -top_db. "
                 "Short clips are zero-padded at the end. source_group identifies the "
                 "original recording file, not a speaker.",
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output}")
    if summary["errors"]:
        return 1
    if not summary["spectrograms"]:
        print("No spectrograms created.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
