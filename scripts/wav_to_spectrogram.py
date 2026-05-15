"""
Render PNG spectrograms for WAV files under ``audio/`` (mirrors paths under ``spectrogram/``).

Each ``audio/.../clip.wav`` → ``spectrogram/.../clip.png`` (same relative path, ``.wav`` → ``.png``).

Requires: ``pip install -r scripts/requirements.txt``

Run from repo root::

  python scripts/wav_to_spectrogram.py
  python scripts/wav_to_spectrogram.py --fig-width 12 --fig-height 1.6
  python scripts/wav_to_spectrogram.py --dry-run

After generating PNGs, run ``python scripts/sync_example_tables.py`` so ``index.html`` img ``src``
points at per-clip PNGs when present (otherwise a legacy placeholder path is used).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def render_one(
    wav_path: Path,
    out_png: Path,
    *,
    n_fft: int,
    hop_length: int,
    cmap: str,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {wav_path}")

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis=None,
        y_axis=None,
        cmap=cmap,
        ax=ax,
    )
    # Stretch to the wide canvas (default equal aspect makes spectrograms look tall).
    ax.set_aspect("auto")
    ax.axis("off")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Root folder to scan for .wav (default: <repo>/audio).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Root folder for .png output (default: <repo>/spectrogram).",
    )
    parser.add_argument("--n-fft", type=int, default=2048, help="STFT window length.")
    parser.add_argument("--hop-length", type=int, default=512, help="STFT hop length.")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap name.")
    parser.add_argument(
        "--fig-width",
        type=float,
        default=9.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=1.8,
        help="Figure height in inches.",
    )
    parser.add_argument("--dpi", type=int, default=120, help="Raster resolution for savefig.")
    parser.add_argument("--max-files", type=int, default=0, help="Stop after N files (0 = no limit).")
    parser.add_argument("--dry-run", action="store_true", help="List targets only; do not write PNGs.")
    args = parser.parse_args()

    repo = _repo_root()
    audio_dir = (args.audio_dir or repo / "audio").resolve()
    out_dir = (args.out_dir or repo / "spectrogram").resolve()

    if not audio_dir.is_dir():
        print(f"Audio directory not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    wavs = sorted(audio_dir.rglob("*.wav"))
    if args.max_files > 0:
        wavs = wavs[: args.max_files]

    if args.dry_run:
        for wav in wavs:
            rel = wav.relative_to(audio_dir)
            png = (out_dir / rel).with_suffix(".png")
            print(f"{wav} -> {png}")
        print(f"[dry-run] {len(wavs)} file(s).")
        return

    try:
        import librosa  # noqa: F401
    except ImportError:
        print(
            "Missing dependency. Install with:\n  pip install -r scripts/requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    ok, failed = 0, 0
    for wav in wavs:
        rel = wav.relative_to(audio_dir)
        png = (out_dir / rel).with_suffix(".png")
        try:
            render_one(
                wav,
                png,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                cmap=args.cmap,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                dpi=args.dpi,
            )
            ok += 1
            print(f"OK  {rel.as_posix()}")
        except Exception as e:
            failed += 1
            print(f"ERR {rel.as_posix()}: {e}", file=sys.stderr)

    print(f"Done. wrote={ok} failed={failed} out_dir={out_dir}")
    print("Tip: run python scripts/sync_example_tables.py to wire per-clip PNGs in index.html.")


if __name__ == "__main__":
    main()
