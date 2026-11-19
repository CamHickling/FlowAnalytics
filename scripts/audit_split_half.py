"""Split-half check of the motion-matched GoPro offsets.

audit_offsets.py matches the whole performance. Here the first and second
halves are matched separately (front and side jointly, within a few seconds of
the v2 offset). If the match is real, both halves give the same lag; their
difference is an accuracy estimate that does not rely on front/side agreeing,
and a steady difference across trials would reveal clock drift between cameras.

Usage: python audit_split_half.py [--window 3] [--workers 3]
Writes reports/sync_split_half.csv.
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from audit_offsets import DATA_ROOT, RATE, best_lag, lag_scores, motion


def check(trial: Path, window: float) -> dict:
    events = json.loads(next(trial.glob("*_sync_manifest.json")).read_text())["events"]
    played = next(e["duration_sec"] for e in events if e["event"] == "review_video_player_shown")
    candidates = sorted(trial.glob("performance/*.mp4"))
    lengths = {p: cv2.VideoCapture(str(p)).get(7) / 30.0 for p in candidates}
    overhead = min(candidates, key=lambda p: abs(lengths[p] - played))
    offsets = json.loads((trial / "MVP" / "sync_offsets_v2.json").read_text())
    ref = motion(str(overhead), 0.0, played)
    half = len(ref) // 2
    sigs = {}
    for cam in ("front", "side"):
        source = trial / "MVP" / "generated" / f"{trial.name.split('_')[0]}_{cam}_undistorted.mp4"
        sigs[cam] = motion(str(source), offsets[f"{cam}_offset_sec"] - window, played + 2 * window)
    row = {"trial": trial.name}
    for name, sl in (("first", slice(0, half)), ("second", slice(half, len(ref)))):
        shift = sl.start
        scores = [lag_scores(ref[sl], sigs[c][shift:shift + (sl.stop - sl.start) + int(2 * window * RATE)])
                  for c in ("front", "side")]
        n = min(len(s) for s in scores)
        lag, peak, _ = best_lag((scores[0][:n] + scores[1][:n]) / 2)
        row[f"{name}_lag_s"] = round(lag - window, 3)
        row[f"{name}_peak"] = round(peak, 3)
    row["halves_differ_s"] = round(row["second_lag_s"] - row["first_lag_s"], 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=3.0, help="seconds either side of the v2 offset")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    trials = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and (p / "MVP" / "sync_offsets_v2.json").is_file())
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(check, trials, [args.window] * len(trials)))
    out = DATA_ROOT.parent / "reports" / "sync_split_half.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['trial'][:22]:22} first {r['first_lag_s']:+6.3f}s ({r['first_peak']:.2f})  "
              f"second {r['second_lag_s']:+6.3f}s ({r['second_peak']:.2f})  differ {r['halves_differ_s']:+6.3f}s")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
