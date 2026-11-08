from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

try:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import webrtcvad
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\nInstall with:\n"
        "python -m pip install numpy scipy soundfile webrtcvad-wheels"
    ) from exc

DETECT_RATE = 16000
FRAME = 320
FRAME_SECONDS = FRAME / DETECT_RATE


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect(audio: np.ndarray, rate: int, mode: int):
    channel = int(np.argmax(np.mean(audio * audio, axis=0)))
    mono = audio[:, channel]
    divisor = math.gcd(rate, DETECT_RATE)
    if rate != DETECT_RATE:
        mono = resample_poly(mono, DETECT_RATE // divisor, rate // divisor)
    padded = np.pad(mono, (0, (-len(mono)) % FRAME)).reshape(-1, FRAME)
    energy = np.sqrt(np.mean(padded * padded, axis=1))
    pcm = np.rint(np.clip(padded, -1, 32767 / 32768) * 32768).astype("<i2")
    vad = webrtcvad.Vad(mode)  # Reset state for each recording.
    speech = np.array([vad.is_speech(f.tobytes(), DETECT_RATE) for f in pcm])
    return speech, energy, channel


def speech_regions(speech, rate, length, gap_seconds, padding_seconds):
    regions = []
    for first, last in runs(speech):
        start = min(length, round(first * FRAME_SECONDS * rate))
        end = min(length, round(last * FRAME_SECONDS * rate))
        if regions and start - regions[-1][1] <= round(gap_seconds * rate):
            regions[-1][1] = end
        else:
            regions.append([start, end])
    pad = round(padding_seconds * rate)
    padded = []
    for i, (start, end) in enumerate(regions):
        left = 0 if i == 0 else (regions[i - 1][1] + start) // 2
        right = length if i + 1 == len(regions) else (end + regions[i + 1][0]) // 2
        padded.append((max(left, start - pad), min(right, end + pad)))
    return padded


def partition(start, end, rate, speech, energy, minimum, maximum, target):
    pauses = [
        (round(a * FRAME_SECONDS * rate), round(b * FRAME_SECONDS * rate))
        for a, b in runs(~speech) if b - a >= 3
    ]
    while end - start > maximum:
        low = start + minimum
        high = min(start + maximum, end - minimum)
        preferred = min(high, max(low, start + target))
        candidates = []
        for a, b in pauses:
            # Keep cuts inside a pause, not at its speech boundary.
            a, b = max(low, a + 1), min(high, b - 1)
            if a <= b:
                candidates.append(min(b, max(a, preferred)))
        if candidates:
            cut = min(candidates, key=lambda p: (abs(p - preferred), p))
            reason = "pause"
        else:
            # Search +/- 350 ms around the target; score 20 ms frame RMS.
            left = max(low, preferred - round(0.35 * rate))
            right = min(high, preferred + round(0.35 * rate))
            first = max(0, math.ceil(left / rate / FRAME_SECONDS - 0.5))
            last = min(len(energy) - 1, math.floor(right / rate / FRAME_SECONDS - 0.5))
            candidates = [preferred, left, right]
            candidates += [round((i + 0.5) * FRAME_SECONDS * rate)
                           for i in range(first, last + 1)]
            def score(point):
                frame = min(len(energy) - 1, int(point / rate / FRAME_SECONDS))
                return (energy[frame], abs(point - preferred), point)
            cut = min((p for p in candidates if low <= p <= high), key=score)
            reason = "low_energy"
        yield start, cut, reason
        start = cut
    yield start, end, "region_end"


def speech_measure(speech):
    prefix = np.r_[0, np.cumsum(speech)] * FRAME_SECONDS
    def at(seconds):
        frame_position = max(0.0, seconds / FRAME_SECONDS)
        frame = int(frame_position)
        if frame >= len(speech):
            return float(prefix[-1])
        return float(prefix[frame] + (frame_position - frame) * FRAME_SECONDS * speech[frame])
    return lambda start, end: max(0.0, at(end) - at(start))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path, help="WAV file or directory; recursive search")
    parser.add_argument("output", type=Path, help="New or empty output directory")
    parser.add_argument("--label", type=int, choices=[0, 1], help="Class for ALL input files")
    parser.add_argument("--min-seconds", type=float, default=1.0, help="Minimum exported duration")
    parser.add_argument("--max-seconds", type=float, default=3.0, help="Maximum exported duration")
    parser.add_argument("--target-seconds", type=float, default=2.0, help="Preferred split duration")
    parser.add_argument("--gap-ms", type=float, default=400, help="Longest pause to bridge")
    parser.add_argument("--padding-ms", type=float, default=120, help="Original context at region edges")
    parser.add_argument("--vad-mode", type=int, choices=range(4), default=2,
                        help="0 retains more speech; 3 rejects more non-speech")
    parser.add_argument("--min-speech-ratio", type=float, default=0.5,
                        help="Minimum VAD speech fraction in an exported clip")
    args = parser.parse_args(argv)
    values = [args.min_seconds, args.max_seconds, args.target_seconds,
              args.gap_ms, args.padding_ms, args.min_speech_ratio]
    if not all(math.isfinite(x) for x in values):
        parser.error("Numeric options must be finite.")
    if not (0 < args.min_seconds <= args.target_seconds <= args.max_seconds):
        parser.error("Require 0 < min-seconds <= target-seconds <= max-seconds.")
    if args.max_seconds < 2 * args.min_seconds:
        parser.error("max-seconds must be >= 2 * min-seconds to avoid unpartitionable tails.")
    if min(args.gap_ms, args.padding_ms) < 0 or not 0 < args.min_speech_ratio <= 1:
        parser.error("Gap/padding must be nonnegative; speech ratio must be in (0, 1].")
    return args


def main(argv=None):
    args = parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Input does not exist: {source}")
    if source.is_file():
        files = [source] if source.suffix.lower() == ".wav" else []
        root = source.parent
    else:
        root = source
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise SystemExit("No WAV files found.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("Output must be a new or empty directory (nothing was overwritten).")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["chunk_file", "source_file", "source_group", "label", "start_sample",
              "end_sample", "start_seconds", "end_seconds", "duration_seconds",
              "speech_seconds", "speech_ratio", "sample_rate", "channels",
              "vad_channel", "subtype", "end_boundary"]
    skipped_fields = ["source_file", "start_seconds", "end_seconds", "reason", "detail"]
    summary = {"files": len(files), "chunks": 0, "skipped_regions_or_chunks": 0,
               "files_without_chunks": 0, "errors": 0, "exported_seconds": 0.0}
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as mf, \
         (output / "skipped.csv").open("w", newline="", encoding="utf-8") as sk:
        manifest = csv.DictWriter(mf, fields)
        skipped = csv.DictWriter(sk, skipped_fields)
        manifest.writeheader()
        skipped.writeheader()
        for path in files:
            relative = path.relative_to(root).as_posix()
            count_before = summary["chunks"]
            def skip(start, end, reason, detail=""):
                skipped.writerow(dict(source_file=relative, start_seconds=start,
                                      end_seconds=end, reason=reason, detail=detail))
                summary["skipped_regions_or_chunks"] += 1
            try:
                info = sf.info(str(path))
                audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
                if not len(audio):
                    skip(0, 0, "empty_file")
                elif not np.isfinite(audio).all():
                    raise ValueError("Audio contains NaN or infinite samples")
                else:
                    minimum = math.ceil(args.min_seconds * rate)
                    maximum = math.floor(args.max_seconds * rate)
                    target = round(args.target_seconds * rate)
                    if maximum < 2 * minimum:
                        raise ValueError("Sample-rounded maximum must be >= twice minimum")
                    speech, energy, channel = detect(audio, rate, args.vad_mode)
                    measure = speech_measure(speech)
                    regions = speech_regions(speech, rate, len(audio), args.gap_ms / 1000,
                                             args.padding_ms / 1000)
                    if not regions:
                        skip(0, len(audio) / rate, "no_speech_detected")
                    subtype = info.subtype if info.subtype in {
                        "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"
                    } else "PCM_16"
                    group = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
                    folder = Path(relative).parent / (path.stem + "__" + group)
                    number = 0
                    for begin, finish in regions:
                        if finish - begin < minimum:
                            skip(begin / rate, finish / rate, "short_speech_region")
                            continue
                        for start, end, boundary in partition(begin, finish, rate, speech, energy,
                                                              minimum, maximum, target):
                            duration = (end - start) / rate
                            voiced = min(duration, measure(start / rate, end / rate))
                            ratio = voiced / duration
                            if ratio < args.min_speech_ratio:
                                skip(start / rate, end / rate, "low_speech_ratio", f"{ratio:.4f}")
                                continue
                            assert minimum <= end - start <= maximum
                            number += 1
                            destination = folder / f"chunk_{number:05d}.wav"
                            (output / folder).mkdir(parents=True, exist_ok=True)
                            sf.write(str(output / destination), audio[start:end], rate,
                                     format="WAV", subtype=subtype)
                            manifest.writerow(dict(
                                chunk_file=destination.as_posix(), source_file=relative,
                                source_group=group, label="" if args.label is None else args.label,
                                start_sample=start, end_sample=end, start_seconds=start / rate,
                                end_seconds=end / rate, duration_seconds=duration,
                                speech_seconds=voiced, speech_ratio=ratio, sample_rate=rate,
                                channels=audio.shape[1], vad_channel=channel, subtype=subtype,
                                end_boundary=boundary))
                            summary["chunks"] += 1
                            summary["exported_seconds"] += duration
            except (OSError, RuntimeError, ValueError) as exc:
                summary["errors"] += 1
                skip("", "", "file_error", str(exc))
                print(f"ERROR {relative}: {exc}", file=sys.stderr)
            created = summary["chunks"] - count_before
            if not created:
                summary["files_without_chunks"] += 1
            print(f"{relative}: {created} chunks")
    record = {
        "script_version": "1.0.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root), "options": {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
        "packages": {p: version(p) for p in ["numpy", "scipy", "soundfile", "webrtcvad-wheels"]},
        "summary": summary,
        "notes": "source_group identifies a relative path, not a speaker or a recording session."
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output}")
    if summary["errors"]:
        return 1
    if not summary["chunks"]:
        print("No clips exported; review skipped.csv and VAD settings.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
