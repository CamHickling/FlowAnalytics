#!/usr/bin/env python3
"""
merge_participant_sessions.py

Strictly scans and merges GoPro split footage:
- Part 1 in gopro_footage/
- Part 2 in gopro_footage/2of2/
Uses imageio-ffmpeg static binary to avoid Windows DLL entry point crashes.
"""

import sys
import os
import re
import subprocess
from pathlib import Path

# Resolve working static FFmpeg binary
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"


def find_gopro_split_pairs(gopro_dir: Path):
    sub_2of2 = gopro_dir / "2of2"
    if not sub_2of2.is_dir():
        return []

    pairs = []
    valid_exts = {'.mp4', '.mov'}
    part2_files = [f for f in sub_2of2.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    part1_files = [f for f in gopro_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

    for p2 in part2_files:
        p2_stem = p2.stem.lower()
        matched_p1 = None

        if "front" in p2_stem:
            matched_p1 = next((f for f in part1_files if "front" in f.stem.lower() and "scoring" not in f.stem.lower() and "_combined" not in f.stem.lower()), None)
        elif "side" in p2_stem:
            matched_p1 = next((f for f in part1_files if "side" in f.stem.lower() and "scoring" not in f.stem.lower() and "_combined" not in f.stem.lower()), None)
        elif "sync_full" in p2_stem:
            matched_p1 = next((f for f in part1_files if "sync_full" in f.stem.lower() and "_combined" not in f.stem.lower()), None)
        else:
            clean_stem = re.sub(r'[_\s\-\(]*2of2\)?', '', p2.stem, flags=re.IGNORECASE)
            matched_p1 = next((f for f in part1_files if f.stem.lower() == clean_stem.lower() and "_combined" not in f.stem.lower()), None)

        if matched_p1:
            out_file = gopro_dir / f"{matched_p1.stem}_COMBINED{matched_p1.suffix}"
            pairs.append((matched_p1, p2, out_file))

    return pairs


def run_ffmpeg_concat(gopro_dir: Path, file1: Path, file2: Path, output_file: Path):
    manifest = gopro_dir / f"tmp_concat_{output_file.stem}.txt"
    
    rel_p1 = file1.relative_to(gopro_dir).as_posix()
    rel_p2 = file2.relative_to(gopro_dir).as_posix()
    rel_out = output_file.relative_to(gopro_dir).as_posix()

    try:
        with open(manifest, "w", encoding="utf-8") as m:
            m.write(f"file '{rel_p1}'\n")
            m.write(f"file '{rel_p2}'\n")

        cmd = [
            FFMPEG_EXE, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", manifest.name,
            "-c", "copy",
            rel_out
        ]

        res = subprocess.run(
            cmd,
            cwd=str(gopro_dir.resolve()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            print(f"\n  [FFmpeg Error]: {err_msg}", file=sys.stderr)
            return False
        return True

    except Exception as e:
        print(f"\n  [Execution Error]: {e}", file=sys.stderr)
        return False
    finally:
        if manifest.exists():
            try:
                manifest.unlink()
            except OSError:
                pass


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    if not root_dir.is_dir():
        print(f"Error: Directory '{root_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Using FFmpeg binary: {FFMPEG_EXE}")
    print(f"Scanning '{root_dir.resolve()}' strictly for GoPro split pairs in gopro_footage/...\n")

    all_jobs = []

    for item in sorted(root_dir.iterdir()):
        if item.is_dir() and item.name.startswith("P"):
            gopro_dir = item / "gopro_footage"
            if gopro_dir.is_dir():
                gp_pairs = find_gopro_split_pairs(gopro_dir)
                for p1, p2, out in gp_pairs:
                    all_jobs.append((item.name, gopro_dir, p1, p2, out))

    if not all_jobs:
        print("No GoPro split video sets found in gopro_footage/2of2/ to process.")
        return

    print(f"Found {len(all_jobs)} GoPro video pair(s) to merge:\n")
    for participant, g_dir, p1, p2, out in all_jobs:
        print(f"  [{participant}]:")
        print(f"    Part 1: {p1.relative_to(root_dir)}")
        print(f"    Part 2: {p2.relative_to(root_dir)}")
        print(f"    Output: {out.relative_to(root_dir)}\n")

    confirm = input("Run lossless batch merge on these GoPro files? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    print("\nProcessing GoPro footage...")
    success_count = 0
    for participant, g_dir, p1, p2, out in all_jobs:
        print(f"Merging -> {out.name} ...", end=" ", flush=True)
        if run_ffmpeg_concat(g_dir, p1, p2, out):
            print("✓ DONE")
            success_count += 1
        else:
            print("✗ FAILED")

    print(f"\nFinished: {success_count}/{len(all_jobs)} GoPro merges completed successfully.")


if __name__ == "__main__":
    main()