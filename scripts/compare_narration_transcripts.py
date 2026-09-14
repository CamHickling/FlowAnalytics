#!/usr/bin/env python3
"""Compare cleaned narration text with timed WhisperX narration transcripts."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", text.lower())).strip()


def parse_srt(path: Path) -> list[dict[str, Any]]:
    blocks = re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip())
    entries = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
        entries.append({
            "index": len(entries) + 1,
            "start": parse_timestamp(start_text),
            "end": parse_timestamp(end_text),
            "whisper_text": " ".join(lines[2:]),
        })
    return entries


def parse_timestamp(value: str) -> float:
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
    return [
        {
            "index": index,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "whisper_text": str(segment.get("text", "")).strip(),
        }
        for index, segment in enumerate(segments, 1)
        if "start" in segment and "end" in segment
    ]


def load_timed_segments(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".srt":
        return parse_srt(path)
    if path.suffix.lower() == ".json":
        return parse_json(path)
    raise ValueError(f"Unsupported timed transcript: {path}")


def read_clean_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def align_lines(clean_lines: list[str], timed_segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    timed_text = [normalize(segment["whisper_text"]) for segment in timed_segments]
    clean_text = [normalize(line) for line in clean_lines]
    matcher = difflib.SequenceMatcher(a=clean_text, b=timed_text, autojunk=False)
    synthesis = []
    unmatched_clean = []
    unmatched_timed = []

    for tag, clean_start, clean_end, timed_start, timed_end in matcher.get_opcodes():
        if tag == "equal":
            for clean_index, timed_index in zip(range(clean_start, clean_end), range(timed_start, timed_end)):
                segment = dict(timed_segments[timed_index])
                segment["clean_text"] = clean_lines[clean_index]
                segment["match"] = "exact"
                synthesis.append(segment)
        else:
            unmatched_clean.extend(clean_lines[clean_start:clean_end])
            unmatched_timed.extend(timed_segments[index]["whisper_text"] for index in range(timed_start, timed_end))
            for timed_index in range(timed_start, timed_end):
                segment = dict(timed_segments[timed_index])
                segment["clean_text"] = None
                segment["match"] = "needs_review"
                synthesis.append(segment)

    synthesis.sort(key=lambda segment: segment["start"])
    return synthesis, unmatched_clean, unmatched_timed


def write_srt(path: Path, synthesis: list[dict[str, Any]]) -> None:
    blocks = []
    for index, segment in enumerate(synthesis, 1):
        text = segment["clean_text"] or segment["whisper_text"]
        blocks.append(
            f"{index}\n{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n{text}"
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def compare(clean_path: Path, timed_path: Path, output_dir: Path) -> dict[str, Any]:
    clean_lines = read_clean_lines(clean_path)
    timed_segments = load_timed_segments(timed_path)
    synthesis, unmatched_clean, unmatched_timed = align_lines(clean_lines, timed_segments)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = clean_path.stem
    synthesis_json = output_dir / f"{stem}_synthesis.json"
    synthesis_srt = output_dir / f"{stem}_synthesis.srt"
    report = {
        "clean_source": str(clean_path),
        "timed_source": str(timed_path),
        "clean_line_count": len(clean_lines),
        "timed_segment_count": len(timed_segments),
        "exact_match_count": sum(segment["match"] == "exact" for segment in synthesis),
        "needs_review_count": sum(segment["match"] == "needs_review" for segment in synthesis),
        "unmatched_clean_lines": unmatched_clean,
        "unmatched_timed_segments": unmatched_timed,
        "status": "MATCHED" if not unmatched_clean and not unmatched_timed else "NEEDS_REVIEW",
        "segments": synthesis,
    }
    synthesis_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_srt(synthesis_srt, synthesis)
    report["synthesis_json"] = str(synthesis_json)
    report["synthesis_srt"] = str(synthesis_srt)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare one cleaned narration TXT with timed SRT/JSON")
    parser.add_argument("--clean", type=Path, required=True, help="Human-cleaned narration TXT")
    parser.add_argument("--timed", type=Path, required=True, help="Timed narration SRT or JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for the new synthesis files")
    args = parser.parse_args()
    report = compare(args.clean, args.timed, args.output_dir)
    print(f"Status: {report['status']}")
    print(f"Clean lines: {report['clean_line_count']}")
    print(f"Timed segments: {report['timed_segment_count']}")
    print(f"Exact matches: {report['exact_match_count']}")
    print(f"Needs review: {report['needs_review_count']}")
    print(f"Unmatched clean lines: {len(report['unmatched_clean_lines'])}")
    print(f"Unmatched timed segments: {len(report['unmatched_timed_segments'])}")
    print(f"Synthesis JSON: {report['synthesis_json']}")
    print(f"Synthesis SRT: {report['synthesis_srt']}")
    return 0 if report["status"] == "MATCHED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
