"""Re-measure the front/side GoPro gap by cross-correlating their audio.

Both GoPros record the same room, so their sound (footsteps, kiais, voices)
lines up once the start-time gap is removed. This searches the full length of
both recordings (not a limited window) using an onset envelope at 100 Hz, and
compares the result with gopro_footage/sync_info.json.

Usage: python audit_side_audio.py [--workers 6] [--only P12 P50]
Writes side_audio_audit.csv next to the data root.
"""

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA_ROOT = Path("D:/FlowAnalytics/Iris_Recorded_Taekwondo_Data")
FFMPEG = "C:/Users/BarlabPRIME/anaconda3/envs/video/Library/bin/ffmpeg.exe"
SR = 4000      # decode rate
ENV_RATE = 100  # envelope samples per second


def onset_envelope(path: str) -> np.ndarray:
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    hop = SR // ENV_RATE
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    energy = np.log1p(np.sqrt((frames ** 2).mean(axis=1)))
    onset = np.maximum(np.diff(energy, prepend=energy[0]), 0)  # rises in loudness
    return (onset - onset.mean()) / (onset.std() or 1.0)


def xcorr_lag(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Seconds to add to a front time to get the side time; peak z-score; runner-up z-score."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    cc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    cc = np.concatenate([cc[-(len(a) - 1):], cc[:len(b)]])  # lags -(len(a)-1) .. len(b)-1
    lags = np.arange(-(len(a) - 1), len(b))
    z = (cc - cc.mean()) / cc.std()
    k = int(np.argmax(z))
    masked = z.copy()
    masked[max(0, k - ENV_RATE):k + ENV_RATE] = -np.inf
    return lags[k] / ENV_RATE, float(z[k]), float(masked.max())


def audit(trial: Path) -> dict:
    prepared = json.loads((trial / "MVP" / "generated" / "prepared_assets.json").read_text())
    offsets = json.loads((trial / "MVP" / "sync_offsets.json").read_text())
    info_path = trial / "gopro_footage" / "sync_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    front, side = onset_envelope(prepared["front"]["source"]), onset_envelope(prepared["side"]["source"])
    lag, z, runner_up = xcorr_lag(front, side)
    saved_lag = offsets["side_offset_sec"] - offsets["front_offset_sec"]
    return {
        "trial": trial.name,
        "saved_side_minus_front": round(saved_lag, 3),
        "measured_side_minus_front": round(lag, 3),
        "diff_s": round(lag - saved_lag, 3),
        "peak_z": round(z, 1),
        "runner_up_z": round(runner_up, 1),
        "sync_info_confidence": info.get("confidence_score"),
        "sync_info_status": info.get("status"),
        "front_len_s": round(len(front) / ENV_RATE, 1),
        "side_len_s": round(len(side) / ENV_RATE, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    trials = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("P")
                    and (p / "MVP" / "sync_offsets.json").is_file())
    if args.only:
        trials = [t for t in trials if any(t.name.startswith(o) for o in args.only)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(audit, trials))
    out = DATA_ROOT.parent / "side_audio_audit.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['trial'][:22]:22} saved {r['saved_side_minus_front']:+9.3f}  measured {r['measured_side_minus_front']:+9.3f}"
              f"  diff {r['diff_s']:+8.3f}  z {r['peak_z']:5.1f} (next {r['runner_up_z']:4.1f})  old conf {r['sync_info_confidence']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
