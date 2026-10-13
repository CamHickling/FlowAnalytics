"""Independent check of each trial's front/side GoPro offsets.

The athlete's movement produces a motion-energy signal (mean absolute frame
difference) in every camera. Cross-correlating the overhead (performance)
signal with the front and side signals gives the lag at which the cameras
line up, which is compared against MVP/sync_offsets.json.

Usage: python audit_offsets.py [--search 60] [--workers 6] [--only P03 P26]
Writes offset_audit.csv next to the data root.
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("D:/FlowAnalytics/Iris_Recorded_Taekwondo_Data")
RATE = 10.0  # motion samples per second


def motion(path: str, start: float, duration: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(start, 0.0) * 1000)
    step = max(1, int(round(fps / RATE)))
    n = int(duration * RATE)
    out, prev, i = [], None, 0
    while len(out) < n:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            out.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
            prev = g
        i += 1
    sig = np.array(out)
    if start < 0:  # pad the part before the file starts
        sig = np.concatenate([np.full(int(-start * RATE), np.nan), sig])
    return sig


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.log1p(np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0.0))
    return (x - x.mean()) / (x.std() or 1.0)


def best_lag(ref: np.ndarray, sig: np.ndarray) -> tuple[float, float, float]:
    """Lag (s) of ref within sig, peak correlation, and the margin over the next-best peak."""
    ref, sig = zscore(ref), zscore(sig)
    scores = np.array([np.dot(ref, sig[k:k + len(ref)]) / len(ref) for k in range(len(sig) - len(ref) + 1)])
    k = int(np.argmax(scores))
    others = scores.copy()
    others[max(0, k - int(2 * RATE)):k + int(2 * RATE)] = -np.inf  # exclude the peak's neighbourhood
    return k / RATE, float(scores[k]), float(scores[k] - others.max())


def audit(trial: Path, search: float) -> dict:
    events = json.loads(next(trial.glob("*_sync_manifest.json")).read_text())["events"]
    played = next(e["duration_sec"] for e in events if e["event"] == "review_video_player_shown")
    # The played take is the overhead file whose length matches what the player reported
    candidates = sorted(trial.glob("performance/*.mp4"))
    lengths = {p: cv2.VideoCapture(str(p)).get(7) / 30.0 for p in candidates}
    overhead = min(candidates, key=lambda p: abs(lengths[p] - played))
    offsets = json.loads((trial / "MVP" / "sync_offsets.json").read_text())
    prepared = json.loads((trial / "MVP" / "generated" / "prepared_assets.json").read_text())

    ref = motion(str(overhead), 0.0, played)
    row = {"trial": trial.name, "overhead_file": overhead.name, "performance_s": round(played, 1)}
    for cam in ("front", "side"):
        saved = offsets[f"{cam}_offset_sec"]
        sig = motion(prepared[cam]["output"], saved - search, played + 2 * search)
        lag, peak, margin = best_lag(ref, sig)
        measured = saved - search + lag
        row.update({
            f"{cam}_saved": round(saved, 2),
            f"{cam}_measured": round(measured, 2),
            f"{cam}_diff": round(measured - saved, 2),
            f"{cam}_peak": round(peak, 3),
            f"{cam}_margin": round(margin, 3),
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=float, default=60.0, help="seconds either side of the saved offset")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()

    trials = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("P")
                    and (p / "MVP" / "sync_offsets.json").is_file())
    if args.only:
        trials = [t for t in trials if any(t.name.startswith(o) for o in args.only)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(audit, trials, [args.search] * len(trials)))

    out = DATA_ROOT.parent / "offset_audit.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['trial'][:22]:22} front {r['front_diff']:+7.2f}s (peak {r['front_peak']:.2f}, margin {r['front_margin']:.2f})"
              f"   side {r['side_diff']:+7.2f}s (peak {r['side_peak']:.2f}, margin {r['side_margin']:.2f})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
