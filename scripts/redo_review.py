#!/usr/bin/env python3
"""Opens the full-length RAW front GoPro footage (fisheye-distorted but
complete) alongside the overhead footage, one trial at a time, for trials
whose review session was accidentally done against a truncated front video --
so the review/narration can be redone comparing overhead vs. the real
performance without waiting for undistortion to finish (undistortion only
removes lens warp, it doesn't change content or duration).

Usage:
    python redo_review.py [TRIAL ...] [--vlc-path "C:\\...\\vlc.exe"]

With no arguments, goes through P34, P43, and P46 (the trials flagged as
reviewed on a truncated front video) one at a time: opens that trial's
overhead + front videos, waits for you to redo the review and press Enter
(or 'q' to quit) before closing them and moving to the next trial. Pass
explicit trial folder name(s) or prefixes (e.g. "P10") to do others instead.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mvp

DATA_ROOT = Path(r"D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data")
DEFAULT_VLC = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
DEFAULT_TRIALS = ["P34_20260323_101229", "P43_20260327_101321", "P46_20260330_111907"]


def _part_number(path: Path) -> int:
    match = re.search(r"(\d+)of\d+", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else 1


def find_video_parts(directory: Path, keyword: str) -> list[Path]:
    """Overhead recordings are sometimes split into '..._1of2', '..._2of2'
    parts -- return all of them in part order."""
    if not directory.is_dir():
        return []
    matches = [
        p for p in directory.iterdir()
        if p.is_file() and keyword in p.stem.lower() and "sync" not in p.stem.lower()
    ]
    return sorted(matches, key=_part_number)


def resolve_trial(name: str) -> Path:
    exact = DATA_ROOT / name
    if exact.is_dir():
        return exact
    matches = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith(name))
    if not matches:
        raise FileNotFoundError(f"no trial folder matching '{name}' under {DATA_ROOT}")
    if len(matches) > 1:
        raise FileNotFoundError(f"'{name}' matches multiple trial folders: {[m.name for m in matches]}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trials", nargs="*", default=DEFAULT_TRIALS,
        help="trial folder name(s) or prefix(es), e.g. P34. Defaults to P34/P43/P46.",
    )
    parser.add_argument("--vlc-path", default=DEFAULT_VLC)
    args = parser.parse_args()

    if not Path(args.vlc_path).is_file():
        print(f"ERROR: VLC not found at {args.vlc_path}. Pass --vlc-path.", file=sys.stderr)
        return 1

    for name in args.trials:
        try:
            trial = resolve_trial(name)
        except FileNotFoundError as exc:
            print(f"SKIP {name}: {exc}")
            continue

        gopro_dir = trial / "gopro_footage"
        gopro_files = sorted(
            p for p in gopro_dir.iterdir() if p.is_file() and p.suffix.lower() in mvp.VIDEO_EXTENSIONS
        ) if gopro_dir.is_dir() else []
        # find_gopro_video prefers a '_COMBINED' file when present -- that's
        # the full merged recording; the plain '<camera>.MP4' name is just
        # the truncated first ~4GB chapter for trials that split.
        front = mvp.find_gopro_video(gopro_files, "front")
        overhead_parts = find_video_parts(trial / "performance", "overhead")

        if not front:
            print(f"SKIP {trial.name}: no front video found")
            continue
        if not overhead_parts:
            print(f"SKIP {trial.name}: no overhead video found")
            continue

        print(f"=== {trial.name} ===")
        print(f"  front: {front.name}")
        for part in overhead_parts:
            tag = f" (part {_part_number(part)})" if len(overhead_parts) > 1 else ""
            print(f"  overhead{tag}: {part.name}")

        procs = [subprocess.Popen([args.vlc_path, str(front), "--no-one-instance"])]
        for part in overhead_parts:
            procs.append(subprocess.Popen([args.vlc_path, str(part), "--no-one-instance"]))

        choice = input(f"  Redo the review for {trial.name} now, then press Enter to close and move on (q=quit): ").strip()
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
        if choice.lower() == "q":
            break
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
