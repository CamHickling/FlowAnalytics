#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def extract_chapter_info(filename: str):
    """
    Identifies chaptered files and returns (base_name, chapter_index, extension).
    Supports:
      - Explicit tags: 'Video_1of2.MP4', 'Ride (part 1).mov', 'Clip_seg01.mp4'
      - Standard GoPro naming: 'GX010042.MP4', 'GH021234.MP4'
    """
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()

    # 1. Matches: Name_1of2, Name_2of2, Name-part1, Name_seg01, etc.
    match_tag = re.search(r'^(.*?)[_\s\-\(]*(?:part|seg|chapter)?\s*0*(\d+)\s*(?:of\s*\d+|\))?$', stem, re.IGNORECASE)
    if match_tag and re.search(r'(?:of\s*\d+|part|seg)', stem, re.IGNORECASE):
        base_name = match_tag.group(1).rstrip(' _-')
        chapter_num = int(match_tag.group(2))
        return base_name, chapter_num, ext

    # 2. Standard GoPro HERO6+ format (e.g., GX010042.MP4 -> Chapter 01, ID 0042)
    match_gopro_new = re.match(r'^(G[HX]\d{2})(\d{4})$', stem, re.IGNORECASE)
    if match_gopro_new:
        prefix = match_gopro_new.group(1)[:2]
        chapter_num = int(match_gopro_new.group(1)[2:])
        video_id = match_gopro_new.group(2)
        base_name = f"GoPro_{prefix}_{video_id}"
        return base_name, chapter_num, ext

    # 3. Older GoPro format (e.g., GOPR0042.MP4 (Ch 0) & GP010042.MP4 (Ch 1))
    match_gopro_old_base = re.match(r'^GOPR(\d{4})$', stem, re.IGNORECASE)
    if match_gopro_old_base:
        return f"GoPro_GOPR_{match_gopro_old_base.group(1)}", 0, ext

    match_gopro_old_chap = re.match(r'^GP(\d{2})(\d{4})$', stem, re.IGNORECASE)
    if match_gopro_old_chap:
        chapter_num = int(match_gopro_old_chap.group(1))
        video_id = match_gopro_old_chap.group(2)
        return f"GoPro_GOPR_{video_id}", chapter_num, ext

    return None, None, ext


def scan_and_group(directory: Path):
    video_extensions = {'.mp4', '.mov', '.m4v', '.mkv'}
    groups = defaultdict(list)

    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in video_extensions:
            base_name, chapter_num, _ = extract_chapter_info(entry.name)
            if base_name is not None:
                groups[base_name].append((chapter_num, entry))

    # Sort files inside each group by chapter index
    multi_part_groups = {}
    for base_name, files in groups.items():
        if len(files) > 1:
            files.sort(key=lambda x: x[0])
            multi_part_groups[base_name] = [f[1] for f in files]

    return multi_part_groups


def merge_group(base_name: str, file_list: list[Path], output_dir: Path):
    ext = file_list[0].suffix
    out_file = output_dir / f"{base_name}_merged{ext}"
    list_file = output_dir / f"concat_list_{base_name}.txt"

    # Create FFmpeg demuxer manifest
    with open(list_file, 'w', encoding='utf-8') as f:
        for file_path in file_list:
            # Escape single quotes for ffmpeg concat file
            clean_path = str(file_path.resolve()).replace("'", "'\\''")
            f.write(f"file '{clean_path}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_file)
    ]

    print(f"\nMerging {len(file_list)} files -> {out_file.name}...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✓ Successfully created {out_file.name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error merging {base_name}:", file=sys.stderr)
        print(e.stderr.decode(), file=sys.stderr)
    finally:
        if list_file.exists():
            list_file.unlink()


def main():
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    if not target_dir.is_dir():
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)

    print(f"Scanning directory: {target_dir.resolve()}")
    groups = scan_and_group(target_dir)

    if not groups:
        print("No multi-part chaptered videos found.")
        return

    print(f"\nFound {len(groups)} multi-part video set(s) to combine:\n")
    for base_name, files in groups.items():
        print(f"  [{base_name}] ({len(files)} chapters):")
        for f in files:
            print(f"    - {f.name}")

    confirm = input("\nProceed with lossless merge? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Aborted.")
        return

    for base_name, files in groups.items():
        merge_group(base_name, files, target_dir)

    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()