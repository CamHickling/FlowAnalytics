"""Export each trial's lens-corrected front and side GoPro videos as a synced pair.

The two files of a pair start at the same instant and have the same number of
frames, so starting them together plays them in sync. Only the stretch both
cameras recorded is kept. The front/side lag comes from MVP/sync_offsets_v2.json
(falling back to sync_offsets.json): both offsets mark the review start in their
own file, so their difference is how far apart the cameras started recording.
Cuts snap to whole frames (30 fps), leaving at most half a frame (~17 ms) of
residual lag.

After rendering, each pair is checked by cross-correlating the two audio tracks
(the cameras share a room); the measured lag should be ~0.

Usage:
    python export_synced_pairs.py --out D:/Perfomance_Videos_For_Ariel [--only P06 P01] [--skip-existing]
Writes <out>/<trial>/<pid>_front_synced.mp4 and <pid>_side_synced.mp4, and
<out>/sync_pairs.csv with the cuts and the audio check for every trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get(
    "FLOW_ANALYTICS_DATA_ROOT", Path(__file__).resolve().parents[1] / "Iris_Recorded_Taekwondo_Data"
))
FFMPEG = os.environ.get("FLOW_ANALYTICS_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FLOW_ANALYTICS_FFPROBE", "ffprobe")
FPS = 30
AUDIO_SR = 8000


def probe_frames(path: Path) -> int:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    if stream["r_frame_rate"] != f"{FPS}/1":
        raise RuntimeError(f"{path.name} is {stream['r_frame_rate']} fps, expected {FPS}")
    return int(stream["nb_read_packets"])


def load_offsets(trial: Path) -> tuple[dict, str]:
    for name in ("sync_offsets_v2.json", "sync_offsets.json"):
        path = trial / "MVP" / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")), name
    raise RuntimeError("no sync offsets")


def export(src: Path, dst: Path, start_frame: int, frames: int) -> None:
    start = start_frame / FPS
    tmp = dst.with_suffix(".partial.mp4")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", f"{start:.6f}", "-i", str(src),
         "-map", "0:v:0", "-map", "0:a:0?", "-frames:v", str(frames), "-t", f"{frames / FPS:.6f}",
         "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)],
        check=True,
    )
    tmp.replace(dst)


def audio(path: Path) -> np.ndarray:
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.int16).astype(float)


def audio_lag(a: np.ndarray, b: np.ndarray, max_lag: float = 1.0) -> tuple[float, float]:
    """Lag (s) of b relative to a over the loudest 20 s, and the peak's z-score."""
    n = min(len(a), len(b))
    win = 20 * AUDIO_SR
    if n < win + 2 * AUDIO_SR:
        win = n - 2 * AUDIO_SR
    energy = np.convolve(a[:n] ** 2, np.ones(AUDIO_SR), "valid")[::AUDIO_SR]
    centre = int(np.argmax(energy[: (n - win) // AUDIO_SR + 1])) * AUDIO_SR
    lo = max(centre, int(max_lag * AUDIO_SR))
    lo = min(lo, n - win - int(max_lag * AUDIO_SR))
    ref = a[lo:lo + win]
    ref = (ref - ref.mean()) / (ref.std() or 1.0)
    k = int(max_lag * AUDIO_SR)
    seg = b[lo - k:lo + win + k]
    corr = np.correlate(seg - seg.mean(), ref, "valid")
    best = int(np.argmax(np.abs(corr)))
    z = (abs(corr[best]) - np.abs(corr).mean()) / (np.abs(corr).std() or 1.0)
    return (best - k) / AUDIO_SR, float(z)


def process(trial: Path, out_root: Path, skip_existing: bool) -> dict:
    pid = trial.name.split("_")[0]
    gen = trial / "MVP" / "generated"
    src = {cam: gen / f"{pid}_{cam}_undistorted.mp4" for cam in ("front", "side")}
    offsets, offsets_file = load_offsets(trial)
    lag = offsets["front_offset_sec"] - offsets["side_offset_sec"]  # front time at side t=0
    start = {"front": max(0, round(lag * FPS)), "side": max(0, round(-lag * FPS))}
    total = {cam: probe_frames(src[cam]) for cam in src}
    frames = min(total[cam] - start[cam] for cam in src)
    if frames <= 0:
        raise RuntimeError("the two recordings do not overlap")

    out_dir = out_root / trial.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = {cam: out_dir / f"{pid}_{cam}_synced.mp4" for cam in src}
    for cam in src:
        if not (skip_existing and dst[cam].is_file()):
            export(src[cam], dst[cam], start[cam], frames)

    check_lag, check_z = audio_lag(audio(dst["front"]), audio(dst["side"]))
    return {
        "trial": trial.name, "offsets_file": offsets_file,
        "front_minus_side_s": round(lag, 3),
        "front_start_s": round(start["front"] / FPS, 3), "side_start_s": round(start["side"] / FPS, 3),
        "duration_s": round(frames / FPS, 3), "frames": frames,
        "residual_s": round(lag - (start["front"] - start["side"]) / FPS, 4),
        "audio_check_lag_s": round(check_lag, 4), "audio_check_z": round(check_z, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="trial folder prefixes, e.g. P06 P01")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    trials = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name[:1] == "P" and "_" in p.name)
    if args.only:
        trials = [t for t in trials if any(t.name.split("_")[0] == o for o in args.only)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for i, trial in enumerate(trials, 1):
        print(f"[{i}/{len(trials)}] {trial.name}", flush=True)
        try:
            row = process(trial, args.out, args.skip_existing)
            rows.append(row)
            print(f"    {row['duration_s']:.1f} s, front from {row['front_start_s']:.2f} s, side from "
                  f"{row['side_start_s']:.2f} s; audio check {row['audio_check_lag_s']:+.4f} s "
                  f"(z {row['audio_check_z']})", flush=True)
        except (OSError, RuntimeError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
            failed.append(trial.name)
            print(f"    FAILED: {exc}", flush=True)

    report = args.out / "sync_pairs.csv"
    existing = {}
    if report.is_file():
        with report.open(newline="") as f:
            existing = {r["trial"]: r for r in csv.DictReader(f)}
    existing.update({r["trial"]: r for r in rows})
    if existing:
        with report.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(next(iter(existing.values())).keys()))
            writer.writeheader()
            writer.writerows(existing[k] for k in sorted(existing))
    print(f"\nDone: {len(rows)} ok, {len(failed)} failed {failed or ''}. Report: {report}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
