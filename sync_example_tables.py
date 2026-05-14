"""
Regenerate the three HV example tables in index.html from disk layout.

Each row = one subfolder under ``audio/<High|Middle|Low SNR>/HV/<example>/``.
First column text = folder name. Run after adding audio or example folders:

  python sync_example_tables.py

Optional: ``--dry-run`` prints planned rows without writing index.html.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

SNR_BLOCKS: list[tuple[str, str]] = [
    ("High SNR", "Input SNR: -10 dB to 0 dB"),
    ("Middle SNR", "Input SNR: -20 dB to -10 dB"),
    ("Low SNR", "Input SNR: -30 dB to -20 dB"),
]

METHODS: list[tuple[str, str]] = [
    ("clean.wav", "clean"),
    ("noisy.wav", "noisy"),
    ("uss.wav", "uss"),
    ("zeroshot.wav", "zeroshot"),
    ("audiosep.wav", "audiosep"),
    ("audiosep-finetuned.wav", "audiosep-finetuned"),
    ("droneaudionet.wav", "droneaudionet"),
]


def list_example_dirs(audio_dir: Path, snr_key: str) -> list[str]:
    hv = audio_dir / snr_key / "HV"
    if not hv.is_dir():
        return []
    names = [p.name for p in hv.iterdir() if p.is_dir()]
    return sorted(names, key=str.lower)


def build_rows(snr_key: str, alt_bucket: str, examples: list[str]) -> str:
    lines: list[str] = []
    if not examples:
        lines.append(
            "                <tr>\n"
            f"                  <td colspan=\"8\" class=\"hint\">No example folders under "
            f"<code>audio/{html.escape(snr_key)}/HV/</code> yet. Add subfolders (names become "
            f"the first column), run <code>python upload_raw_samples.py</code>, then re-run "
            f"<code>python sync_example_tables.py</code>.</td>\n"
            "                </tr>"
        )
        return "\n".join(lines)

    for ex in examples:
        ex_esc = html.escape(ex)
        lines.append("                <tr>")
        lines.append(f'                  <th scope="row" class="rowhead">{ex_esc}</th>')
        for wav, label in METHODS:
            alt = html.escape(f"{alt_bucket} {ex} {label} spectrogram (placeholder)", quote=True)
            src = f"./audio/{snr_key}/HV/{ex}/{wav}"
            src_esc = html.escape(src, quote=True)
            lines.append(
                "                  <td><div class=\"cell\">"
                f"<audio controls preload=\"none\"><source src=\"{src_esc}\" type=\"audio/wav\" /></audio>"
                f"<img class=\"cell-img\" src=\"./spectrogram/mic3_8array-up-File1_trad.png\" "
                f"alt=\"{alt}\" loading=\"lazy\" /></div></td>"
            )
        lines.append("                </tr>")
    return "\n".join(lines)


def replace_sn_rows(text: str, snr_key: str, new_rows: str) -> str:
    start = f"<!-- SNR_ROWS:{snr_key} -->"
    end = f"<!-- /SNR_ROWS:{snr_key} -->"
    if start not in text or end not in text:
        raise SystemExit(
            f"Missing markers {start!r} … {end!r} in index.html. "
            "Add them around each SNR table <tbody> row block once."
        )
    i = text.index(start) + len(start)
    j = text.index(end)
    return text[:i] + "\n" + new_rows + "\n                " + text[j:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print rows only; do not write.")
    parser.add_argument("--audio-dir", default="audio", type=Path)
    parser.add_argument("--index", default="index.html", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    audio_dir = (repo / args.audio_dir).resolve()
    index_path = (repo / args.index).resolve()
    text = index_path.read_text(encoding="utf-8")

    for snr_key, alt_bucket in SNR_BLOCKS:
        examples = list_example_dirs(audio_dir, snr_key)
        block = build_rows(snr_key, alt_bucket, examples)
        if args.dry_run:
            print(f"=== {snr_key} ({len(examples)} example dir(s)) ===")
            print(block)
            print()
        else:
            text = replace_sn_rows(text, snr_key, block)

    if not args.dry_run:
        index_path.write_text(text, encoding="utf-8")
        print(f"Updated {index_path} from {audio_dir}.")


if __name__ == "__main__":
    main()
