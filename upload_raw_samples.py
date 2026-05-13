import argparse
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

# raw samples/... folder names -> audio/... top-level folder (parallel to Dregon)
SNR_DIR_BY_RAW_FOLDER = {
    "high SNR": "High SNR",
    "mid SNR": "Middle SNR",
    "low SNR": "Low SNR",
}

# raw samples class subfolder -> audio class subfolder (same codes)
CLASS_DIR_BY_RAW_FOLDER = {
    "HV": "HV",
    "HNV": "HNV",
    "NH": "NH",
}

METHOD_FILE_BY_METHOD_TOKEN = {
    "clean": "clean.wav",
    "noisy": "noisy.wav",
    "uss": "uss.wav",
    "zeroshot": "zeroshot.wav",
    "audiosep": "audiosep.wav",
    "ft": "audiosep-finetuned.wav",
    "audiosep-ft": "audiosep-finetuned.wav",
    "audiosep-finetuned": "audiosep-finetuned.wav",
    "k=6": "droneaudionet.wav",
    "k6": "droneaudionet.wav",
}

_WIN_BAD = set('\\/:*?"<>|')


def sanitize_sample_folder(name: str) -> str:
    name = name.strip().strip(".")
    if not name:
        return "unnamed_clip"
    out = []
    for ch in name:
        if ch in _WIN_BAD or ord(ch) < 32:
            out.append("_")
        else:
            out.append(ch)
    s = "".join(out).strip("._")
    return s or "unnamed_clip"


@dataclass(frozen=True)
class ParsedTarget:
    snr_dir: str
    class_dir: str
    sample_folder: str
    method_file: str
    start_sec: float
    end_sec: float

    def output_path(self, audio_dir: Path) -> Path:
        return audio_dir / self.snr_dir / self.class_dir / self.sample_folder / self.method_file


def _read_state(state_file: Path) -> Dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state_file: Path, state: Dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _parse_time_range(token: str) -> Tuple[float, float]:
    token = token.strip()
    m = re.match(r"^(.+?)-(.+?)$", token)
    if not m:
        raise ValueError(f"Invalid time range token: {token}")
    left, right = m.group(1), m.group(2)
    left = left[:-1] if left.lower().endswith("s") else left
    right = right[:-1] if right.lower().endswith("s") else right
    return float(left), float(right)


def _parse_method_file(method_token: str) -> str:
    t = method_token.strip().lower()
    if t.startswith("k") and "6" in t:
        return METHOD_FILE_BY_METHOD_TOKEN["k=6"]
    if t in METHOD_FILE_BY_METHOD_TOKEN:
        return METHOD_FILE_BY_METHOD_TOKEN[t]
    if t in {"ft", "audiosepft"}:
        return METHOD_FILE_BY_METHOD_TOKEN["ft"]
    if t in {"audiosep-finetuned", "audiosep-finetune", "audiosep-fineturned"}:
        return METHOD_FILE_BY_METHOD_TOKEN["audiosep-finetuned"]
    raise ValueError(f"Unknown method token: {method_token}")


def sample_folder_from_stem(stem: str) -> str:
    """Middle of stem: drop leading method token and trailing start-end range."""
    tokens = stem.split("_")
    if len(tokens) < 3:
        raise ValueError(f"Unexpected filename format (need method_<clip>_start-end): {stem}")
    try:
        _parse_time_range(tokens[-1])
    except ValueError as e:
        raise ValueError(f"{stem}: {e}") from e
    middle = "_".join(tokens[1:-1])
    return sanitize_sample_folder(middle)


def parse_raw_wav_path(raw_wav_path: Path, raw_dir: Path) -> ParsedTarget:
    rel = raw_wav_path.relative_to(raw_dir)
    parts = rel.parts

    snr_dir = None
    class_dir = None
    for p in parts:
        if snr_dir is None and p in SNR_DIR_BY_RAW_FOLDER:
            snr_dir = SNR_DIR_BY_RAW_FOLDER[p]
        if class_dir is None and p in CLASS_DIR_BY_RAW_FOLDER:
            class_dir = CLASS_DIR_BY_RAW_FOLDER[p]
    if snr_dir is None or class_dir is None:
        raise ValueError(f"Could not infer snr/class from path: {raw_wav_path}")

    stem = raw_wav_path.stem
    tokens = stem.split("_")
    if len(tokens) < 3:
        raise ValueError(f"Unexpected filename format: {raw_wav_path.name}")

    method_token = tokens[0]
    start_sec, end_sec = _parse_time_range(tokens[-1])
    if end_sec <= start_sec:
        raise ValueError(f"Invalid time range (end<=start): {raw_wav_path.name}")

    sample_folder = sample_folder_from_stem(stem)
    method_file = _parse_method_file(method_token)
    return ParsedTarget(
        snr_dir=snr_dir,
        class_dir=class_dir,
        sample_folder=sample_folder,
        method_file=method_file,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def iter_raw_wavs(raw_dir: Path) -> Iterable[Path]:
    for p in raw_dir.rglob("*.wav"):
        if p.is_file():
            yield p


def crop_wav(raw_wav_path: Path, output_path: Path, start_sec: float, end_sec: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(raw_wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()

        duration_sec = nframes / float(framerate)
        start_sec = max(0.0, start_sec)
        end_sec = min(duration_sec, end_sec)

        start_frame = int(round(start_sec * framerate))
        end_frame = int(round(end_sec * framerate))
        if end_frame <= start_frame:
            raise ValueError(f"Empty crop window for {raw_wav_path.name}")

        wf.setpos(start_frame)
        frames_to_read = end_frame - start_frame
        audio_bytes = wf.readframes(frames_to_read)

    with wave.open(str(output_path), "wb") as out_wf:
        out_wf.setnchannels(n_channels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)
        out_wf.writeframes(audio_bytes)


def st_key(raw_dir: Path, wav_path: Path) -> str:
    return str(wav_path.relative_to(raw_dir))


def get_file_fingerprint(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop + copy new raw WAVs into audio/ for High / Middle / Low SNR (parallel to Dregon)."
    )
    parser.add_argument("--raw-dir", default="raw samples", help="Folder containing raw samples.")
    parser.add_argument("--audio-dir", default="audio", help="Target folder used by index.html.")
    parser.add_argument("--state-file", default="upload_raw_samples_state.json", help="Local processing state (auto-generated).")
    parser.add_argument("--mode", default="all", choices=["all", "high", "mid", "low"], help="Which SNR buckets to process.")
    parser.add_argument("--dry-run", action="store_true", help="Compute actions but do not write files.")
    parser.add_argument("--overwrite", action="store_true", help="Re-crop even if output exists and matches prior state.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    state_file = Path(args.state_file).resolve()

    wanted_snr_dirs = set(SNR_DIR_BY_RAW_FOLDER.values())
    if args.mode != "all":
        wanted = {
            "high": "High SNR",
            "mid": "Middle SNR",
            "low": "Low SNR",
        }
        wanted_snr_dirs = {wanted[args.mode]}

    state = _read_state(state_file)
    existing_outputs_processed = 0

    to_process: list[Tuple[Path, ParsedTarget, str]] = []
    errors: list[str] = []

    for wav_path in sorted(iter_raw_wavs(raw_dir)):
        key = st_key(raw_dir, wav_path)
        try:
            parsed = parse_raw_wav_path(wav_path, raw_dir)
        except Exception as e:
            errors.append(f"{key}: {e}")
            continue

        if parsed.snr_dir not in wanted_snr_dirs:
            continue

        output_path = parsed.output_path(audio_dir)
        fingerprint = get_file_fingerprint(wav_path)
        prev = state.get(key)

        if output_path.exists() and not args.overwrite:
            if prev:
                if prev.get("fingerprint") == fingerprint:
                    existing_outputs_processed += 1
                    continue
            else:
                existing_outputs_processed += 1
                if not args.dry_run:
                    state[key] = {
                        "fingerprint": fingerprint,
                        "target": str(output_path),
                        "parsed": {
                            "snr_dir": parsed.snr_dir,
                            "class_dir": parsed.class_dir,
                            "sample_folder": parsed.sample_folder,
                            "method_file": parsed.method_file,
                            "start_sec": parsed.start_sec,
                            "end_sec": parsed.end_sec,
                        },
                    }
                continue

        to_process.append((wav_path, parsed, key))

    if args.dry_run:
        print(f"[dry-run] raw_dir={raw_dir}")
        total_wavs = len(list(iter_raw_wavs(raw_dir)))
        print(f"[dry-run] Found {total_wavs} wav files total (including possibly already processed).")

    print(f"Will process {len(to_process)} file(s). Errors: {len(errors)}.")

    output_to_keys: Dict[str, str] = {}
    collisions: list[str] = []
    for wav_path, parsed, key in to_process:
        out = str(parsed.output_path(audio_dir))
        if out in output_to_keys:
            collisions.append(f"{out} mapped from both {output_to_keys[out]} and {key}")
        else:
            output_to_keys[out] = key

    if collisions:
        print("Collisions detected (multiple raw files map to same target). Fix naming or re-run with --overwrite:")
        for c in collisions:
            print(f"  - {c}")

    processed_ok = 0
    processed_failed = 0

    for wav_path, parsed, key in to_process:
        output_path = parsed.output_path(audio_dir)
        print(f"- {key}")
        print(f"  -> {parsed.snr_dir}/{parsed.class_dir}/{parsed.sample_folder}/{parsed.method_file}")
        print(f"  crop: {parsed.start_sec:.3f}s - {parsed.end_sec:.3f}s")
        if args.dry_run:
            continue
        try:
            crop_wav(wav_path, output_path, parsed.start_sec, parsed.end_sec)
            state[key] = {
                "fingerprint": get_file_fingerprint(wav_path),
                "target": str(output_path),
                "parsed": {
                    "snr_dir": parsed.snr_dir,
                    "class_dir": parsed.class_dir,
                    "sample_folder": parsed.sample_folder,
                    "method_file": parsed.method_file,
                    "start_sec": parsed.start_sec,
                    "end_sec": parsed.end_sec,
                },
            }
            processed_ok += 1
        except Exception as e:
            processed_failed += 1
            errors.append(f"{key}: crop failed: {e}")
            continue

    if not args.dry_run:
        _write_state(state_file, state)

    print(f"Done. processed_ok={processed_ok} processed_failed={processed_failed} existing_outputs_processed={existing_outputs_processed}.")
    if errors:
        print("Some files were skipped due to errors:")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more.")


if __name__ == "__main__":
    main()
