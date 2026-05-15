"""
Crop WAV clips from ``raw samples/`` into ``audio/`` paths used by ``index.html``.

Naming (underscore-separated, last field is time range in seconds, e.g. ``20-25`` = 5 s):

  ``<method>_<clipOrRowAlias>_<start>-<end>.wav``

- **method** → output filename: ``clean``, ``noisy``, ``uss``, ``zeroshot``, ``audiosep``,
  ``ft`` / ``audiosep-ft`` → ``audiosep-finetuned.wav``, ``k=6`` / ``k6`` / ``droneaudionet``
  → ``droneaudionet.wav``.
- **clipOrRowAlias** (only when the WAV is directly under ``.../HV/`` with no example subfolder
  in the path): middle part of the filename, e.g. ``crying``, ``male``, or a row alias
  ``sample1`` / ``s1`` / ``row1`` → see ``SAMPLE_ROW_FOLDER`` (must stay in sync with
  ``index.html`` first column / ``audio/.../HV/<example>/``).
- **Path layout** (recommended): ``raw samples/<snr>/HV/<example>/<file>.wav`` — the folder
  ``<example>`` (e.g. ``crying``, ``male``) sets the webpage row / output directory. Shorthand:
  ``raw samples/<snr>/<example>/<file>.wav`` is treated as ``HV/<example>/``.

Each raw WAV maps to **exactly one** output path (one table cell). The SNR bucket comes **only**
from the folder in the raw path (e.g. ``high SNR/...`` → ``audio/High SNR/...``), never copied
across SNRs. Use ``--overwrite`` only when you intend to replace an existing cell.

Run from repo root: ``python scripts/upload_raw_samples.py``  (use ``--dry-run`` first).
State: ``upload_raw_samples_state.json`` at repo root (gitignored).
Then run ``python scripts/sync_example_tables.py`` so ``index.html`` table rows match ``audio/<SNR>/HV/*/`` folders.
"""

import argparse
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Table row order (Sample N / row alias) -> example subfolder under HV/ in index.html.
SAMPLE_ROW_FOLDER: Dict[int, str] = {
    1: "crying",
    2: "male",
    3: "female",
}

_ROW_ALIAS_RE = re.compile(r"^(?:sample|s|row)(\d+)$", re.IGNORECASE)

# raw samples/... folder names -> audio/... top-level folder (parallel to Dregon)
SNR_DIR_BY_RAW_FOLDER = {
    "high SNR": "High SNR",
    "mid SNR": "Middle SNR",
    "low SNR": "Low SNR",
}

# Recognized class folder names under each SNR (case-insensitive match in paths).
_CLASS_FOLDERS_CANON = ("HV", "HNV", "NH")

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
    "droneaudionet": "droneaudionet.wav",
}

_WIN_BAD = set('\\/:*?"<>|')


def _canonical_class_dir(name: str) -> Optional[str]:
    u = name.strip().upper()
    if u == "HV":
        return "HV"
    if u == "HNV":
        return "HNV"
    if u == "NH":
        return "NH"
    return None


def _snr_class_example_from_rel(rel: Path) -> Tuple[str, str, Optional[str]]:
    """From path relative to raw_dir: (snr_dir, class_dir, example_folder or None)."""
    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"Path too short: {rel}")

    snr_idx: Optional[int] = None
    snr_key: Optional[str] = None
    for i, p in enumerate(parts[:-1]):
        if p in SNR_DIR_BY_RAW_FOLDER:
            snr_idx = i
            snr_key = p
            break
    if snr_idx is None or snr_key is None:
        raise ValueError(
            f"No SNR folder (one of {sorted(SNR_DIR_BY_RAW_FOLDER.keys())!r}) in path: {rel}"
        )

    snr_dir = SNR_DIR_BY_RAW_FOLDER[snr_key]
    dirs_between = list(parts[snr_idx + 1 : -1])

    if not dirs_between:
        raise ValueError(
            f"Expected folders under SNR, e.g. .../<snr>/HV/<example>/file.wav or "
            f".../<snr>/<example>/file.wav — got only file under SNR: {rel}"
        )

    first_class = _canonical_class_dir(dirs_between[0])
    if first_class is not None:
        if len(dirs_between) >= 2:
            if len(dirs_between) > 2:
                raise ValueError(
                    f"At most one example subfolder under class (got {dirs_between!r}): {rel}"
                )
            return snr_dir, first_class, sanitize_sample_folder(dirs_between[1])
        return snr_dir, first_class, None

    if len(dirs_between) > 1:
        raise ValueError(
            f"Under SNR without {list(_CLASS_FOLDERS_CANON)}, use exactly one example folder "
            f"(got {dirs_between!r}): {rel}"
        )
    return snr_dir, "HV", sanitize_sample_folder(dirs_between[0])


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


def sample_folder_from_tokens(tokens: list[str]) -> str:
    """Folder under HV/HNV/NH: explicit clip name, or row alias sample1 / s2 / row3."""
    if len(tokens) < 3:
        raise ValueError(f"Unexpected filename format (need method_<clip>_start-end): {tokens!r}")
    try:
        _parse_time_range(tokens[-1])
    except ValueError as e:
        raise ValueError(f"{tokens!r}: {e}") from e
    middle = "_".join(tokens[1:-1]).strip()
    if not middle:
        raise ValueError(f"Missing clip / row token in: {tokens!r}")
    m = _ROW_ALIAS_RE.match(middle)
    if m:
        idx = int(m.group(1))
        if idx not in SAMPLE_ROW_FOLDER:
            raise ValueError(
                f"Row alias {middle!r} -> sample {idx}, but only rows "
                f"{min(SAMPLE_ROW_FOLDER)}..{max(SAMPLE_ROW_FOLDER)} are defined. "
                f"Update SAMPLE_ROW_FOLDER in scripts/upload_raw_samples.py to match index.html."
            )
        return SAMPLE_ROW_FOLDER[idx]
    return sanitize_sample_folder(middle)


def _normalize_filename_tokens(tokens: list[str]) -> list[str]:
    """Collapse ``FT_audiosep_<clip>_...`` / ``ft_audiosep_...`` into ``ft_<clip>_...``."""
    if len(tokens) >= 4 and tokens[1].lower() == "audiosep" and tokens[0].lower() in (
        "ft",
        "audiosep-ft",
    ):
        return ["ft", *tokens[2:]]
    return tokens


def parse_raw_wav_path(raw_wav_path: Path, raw_dir: Path) -> ParsedTarget:
    rel = raw_wav_path.relative_to(raw_dir)
    snr_dir, class_dir, example_from_path = _snr_class_example_from_rel(rel)

    tokens = _normalize_filename_tokens(raw_wav_path.stem.split("_"))
    if len(tokens) < 3:
        raise ValueError(f"Unexpected filename format: {raw_wav_path.name}")

    method_token = tokens[0]
    start_sec, end_sec = _parse_time_range(tokens[-1])
    if end_sec <= start_sec:
        raise ValueError(f"Invalid time range (end<=start): {raw_wav_path.name}")

    if example_from_path is not None:
        sample_folder = example_from_path
    else:
        sample_folder = sample_folder_from_tokens(tokens)
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
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "raw samples",
        help="Folder containing raw samples (default: <repo>/raw samples).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=repo_root / "audio",
        help="Target folder used by index.html (default: <repo>/audio).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=repo_root / "upload_raw_samples_state.json",
        help="Local processing state (default: <repo>/upload_raw_samples_state.json).",
    )
    parser.add_argument("--mode", default="all", choices=["all", "high", "mid", "low"], help="Which SNR buckets to process.")
    parser.add_argument("--dry-run", action="store_true", help="Compute actions but do not write files.")
    parser.add_argument("--overwrite", action="store_true", help="Re-crop even if output exists and matches prior state.")
    parser.add_argument(
        "--expect-5s",
        action="store_true",
        help="Require crop length to be ~5.0 s (|end-start| - 5| <= 0.05); exit with error otherwise.",
    )
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

        if args.expect_5s:
            dur = parsed.end_sec - parsed.start_sec
            if abs(dur - 5.0) > 0.05:
                errors.append(f"{key}: crop length is {dur:.3f}s, expected 5.0s (use --expect-5s only for 5s clips)")
                continue

        output_path = parsed.output_path(audio_dir)
        fingerprint = get_file_fingerprint(wav_path)

        # One raw file -> one table cell (one output path). Never overwrite existing audio
        # unless --overwrite (protects already-published cells even if raw is re-exported).
        if output_path.exists() and not args.overwrite:
            existing_outputs_processed += 1
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
