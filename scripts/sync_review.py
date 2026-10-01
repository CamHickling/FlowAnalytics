#!/usr/bin/env python3
"""Interactive per-trial sync review: pops up the overhead + front videos for
each trial (skipping trials already synced), prompts for the matching
timestamps, and saves the offset immediately via mvp.py's record-match logic.

Run this yourself directly -- it needs live input from you as you watch the
videos, so it can't be driven through the agent.

Usage:
    python sync_review.py [--start P01] [--vlc-path "C:\\...\\vlc.exe"]

At each trial:
  - Front video opens in VLC, then overhead opens in VLC (playing side by side).
  - Look for the checkerboard calibration wave in front (early, unmistakable),
    find the same moment in overhead.
  - Type the two timestamps (MM:SS or seconds). Leave either blank to skip
    this trial (e.g. if the checkerboard isn't visible in overhead this time)
    and move on -- you can rerun the script later; already-synced trials are
    skipped automatically.
  - Type 'q' at either prompt to quit the whole review session.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mvp

DATA_ROOT = Path(r"D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data")
DEFAULT_VLC = r"C:\Program Files\VideoLAN\VLC\vlc.exe"


def _part_number(path: Path) -> int:
    """Extract the N from a '..._NofM...' filename part; 1 if there's no such tag."""
    match = re.search(r"(\d+)of\d+", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else 1


def find_video_parts(directory: Path, keyword: str) -> list[Path]:
    """Like find_video, but returns every matching file (e.g. '..._1of2.mp4',
    '..._2of2.mp4') in part order -- the overhead camera's software splits
    recordings, so assuming a single file silently gives the wrong duration
    when computing where a later part starts in the combined timeline."""
    if not directory.is_dir():
        return []
    matches = [
        path for path in directory.iterdir()
        if path.is_file() and keyword in path.stem.lower() and "sync" not in path.stem.lower()
    ]
    return sorted(matches, key=_part_number)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="trial folder name prefix to start from, e.g. P10")
    parser.add_argument("--vlc-path", default=DEFAULT_VLC)
    args = parser.parse_args()

    if not Path(args.vlc_path).is_file():
        print(f"ERROR: VLC not found at {args.vlc_path}. Pass --vlc-path.", file=sys.stderr)
        return 1

    trials = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("P"))
    if args.start:
        trials = [t for t in trials if t.name >= args.start]

    todo = [t for t in trials if not mvp.sync_offsets_path(t).is_file()]
    print(f"{len(todo)} trial(s) still need sync offsets (of {len(trials)} total).\n")

    for trial in todo:
        overhead_parts = find_video_parts(trial / "performance", "overhead")
        gopro_dir = trial / "gopro_footage"
        gopro_files = sorted(
            p for p in gopro_dir.iterdir() if p.is_file() and p.suffix.lower() in mvp.VIDEO_EXTENSIONS
        ) if gopro_dir.is_dir() else []
        # mvp.find_gopro_video prefers a '_COMBINED' file when present (the
        # GoPro splits recordings >~4GB into chapters; some trials only
        # captured the first chapter under the plain name). Using the same
        # function mvp.py itself uses keeps this tool from ever opening a
        # different file than what the actual pipeline will process.
        front = mvp.find_gopro_video(gopro_files, "front")
        if not overhead_parts or not front:
            print(f"SKIP {trial.name}: missing overhead or front video\n")
            continue

        print(f"=== {trial.name} ===")
        # Cumulative start offset of each part in the combined overhead
        # timeline, e.g. part 2 of 2 starts at part 1's duration.
        cumulative = 0.0
        part_starts = []
        for part in overhead_parts:
            part_starts.append(cumulative)
            duration = (mvp.media_info(str(part)) or {}).get("duration_sec") or 0.0
            tag = f" (part {_part_number(part)})" if len(overhead_parts) > 1 else ""
            print(f"  overhead{tag}: {part.name} [cumulative start {cumulative:.1f}s, duration {duration:.1f}s]")
            cumulative += duration
        print(f"  front: {front.name}")

        procs = []
        for part in overhead_parts:
            procs.append(subprocess.Popen([args.vlc_path, str(part), "--no-one-instance"]))
            time.sleep(0.3)
        procs.append(subprocess.Popen([args.vlc_path, str(front), "--no-one-instance"]))

        try:
            if len(overhead_parts) > 1:
                part_choice = input(
                    f"  Which overhead part did you use? (1-{len(overhead_parts)}, blank=skip, q=quit): "
                ).strip()
                if part_choice.lower() == "q":
                    break
                if not part_choice:
                    overhead_time = ""
                else:
                    part_index = int(part_choice) - 1
                    local_time = input(
                        f"  Timestamp WITHIN part {part_choice} (MM:SS, blank=skip, q=quit): "
                    ).strip()
                    if local_time.lower() == "q":
                        break
                    overhead_time = (
                        "" if not local_time
                        else str(part_starts[part_index] + mvp._parse_timecode(local_time))
                    )
            else:
                overhead_time = input("  Overhead timestamp (MM:SS, blank=skip, q=quit): ").strip()
                if overhead_time.lower() == "q":
                    break

            front_time = ""
            if overhead_time:
                front_time = input("  Front timestamp    (MM:SS, blank=skip, q=quit): ").strip()
                if front_time.lower() == "q":
                    break
        finally:
            for proc in procs:
                try:
                    proc.terminate()
                except Exception:
                    pass

        if not overhead_time or not front_time:
            print(f"  Skipped {trial.name} (no timestamps entered)\n")
            continue

        front_offset = mvp._parse_timecode(front_time) - mvp._parse_timecode(overhead_time)
        sync_info_path = trial / "gopro_footage" / "sync_info.json"
        if sync_info_path.is_file():
            sync_info = mvp.json.loads(sync_info_path.read_text(encoding="utf-8"))
            offset_seconds = sync_info.get("offset_seconds")
            side_offset = front_offset - offset_seconds if offset_seconds is not None else None
        else:
            side_offset = None

        if side_offset is None:
            print(f"  WARNING: no sync_info.json for {trial.name}; side_offset not computed, saving front only as side too")
            side_offset = front_offset

        path = mvp.save_sync_offsets(
            trial, front_offset, side_offset,
            note=f"Manually verified via sync_review.py: overhead {overhead_time} matches front {front_time}.",
        )
        print(f"  Saved front_offset={front_offset:.2f} side_offset={side_offset:.2f} -> {path}\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
