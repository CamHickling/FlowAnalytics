#!/usr/bin/env python3
"""Transcribe a folder of video files with WhisperX.

Example:
    python process_videos_whisperx.py /path/to/videos --output-dir /path/to/output --model large-v3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a folder of video files using WhisperX")
    parser.add_argument("input_dir", nargs="?", default=".", help="Folder containing video files (defaults to the current working directory)")
    parser.add_argument("--output-dir", default=None, help="Where to write transcripts (defaults to the input folder)")
    parser.add_argument("--model", default="large-v3", help="Whisper model name to load (default: large-v3)")
    parser.add_argument("--device", default="cuda" if sys.platform != "darwin" else "cpu", help="Torch device: cpu or cuda")
    parser.add_argument("--compute-type", default="float16", help="Torch compute type for the model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for transcription")
    parser.add_argument("--language", default=None, help="Optional language code, e.g. en")
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="Search subfolders recursively")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not search subfolders")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--no-diarize", dest="diarize", action="store_false", help="Disable speaker diarization")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for diarization (defaults to HF_TOKEN from the environment)")
    parser.add_argument("--no-srt", dest="write_srt", action="store_false", help="Skip writing an SRT subtitle file")
    parser.add_argument("--no-txt", dest="write_txt", action="store_false", help="Skip writing a plain text transcript")
    parser.set_defaults(diarize=True, recursive=True, write_srt=True, write_txt=True)
    return parser.parse_args()


def find_video_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.iterdir()

    files = [path for path in candidates if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        speaker = segment.get("speaker")
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(str(index))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_text(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = [str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = int(total % 60)
    millis = int(round((total - int(total)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_output_paths(video_path: Path, output_dir: Path, overwrite: bool) -> tuple[Path, Path | None, Path | None]:
    stem = video_path.stem
    json_path = output_dir / f"{stem}.json"
    srt_path = output_dir / f"{stem}.srt"
    txt_path = output_dir / f"{stem}.txt"
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {json_path}")
    return json_path, srt_path, txt_path


def transcribe_video(video_path: Path, args: argparse.Namespace, output_dir: Path) -> Path:
    try:
        import whisperx
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("WhisperX is not installed. Install it with: pip install -r requirements-whisperx.txt") from exc

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed. Install it before running WhisperX") from exc

    json_path, srt_path, txt_path = build_output_paths(video_path, output_dir, args.overwrite)

    audio = whisperx.load_audio(str(video_path))
    model = whisperx.load_model(args.model, device=args.device, compute_type=args.compute_type)

    result = model.transcribe(audio, batch_size=args.batch_size, language=args.language, print_progress=False)

    if args.diarize:
        if not args.hf_token:
            raise RuntimeError("Diarization requires --hf-token with a valid Hugging Face token or a HF_TOKEN environment variable")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_token, device=args.device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    if result.get("segments"):
        align_model, align_metadata = whisperx.load_align_model(language_code=result.get("language", "en"), device=args.device)
        result = whisperx.align(result["segments"], align_model, align_metadata, audio, args.device, return_char_alignments=False)

    payload = {
        "video": str(video_path),
        "model": args.model,
        "language": result.get("language"),
        "segments": result.get("segments", []),
        "word_segments": result.get("word_segments", []),
    }
    write_json(json_path, payload)

    if args.write_srt and srt_path is not None:
        write_srt(srt_path, payload["segments"])
    if args.write_txt and txt_path is not None:
        write_text(txt_path, payload["segments"])

    return json_path


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 2

    video_files = find_video_files(input_dir, args.recursive)
    if not video_files:
        print(f"No supported video files found in {input_dir}")
        return 0

    print(f"Found {len(video_files)} video file(s) in {input_dir}")
    for video_path in video_files:
        try:
            output_path = transcribe_video(video_path, args, output_dir)
            print(f"Processed {video_path.name} -> {output_path}")
        except FileExistsError as exc:
            print(str(exc))
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Failed for {video_path.name}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    main()
