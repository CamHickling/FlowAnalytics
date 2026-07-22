#!/usr/bin/env python3
"""Batch transcribe a folder of video files with WhisperX.

This script extracts audio first, writes transcript artifacts, and can resume
cleanly when you run it again overnight.

Example:
    python process_videos_whisperx.py /path/to/videos --output-dir /path/to/output --model large-v3 --device cuda --resume
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transcribe a folder of video files using WhisperX")
    parser.add_argument("input_dir", nargs="?", default=".", help="Folder containing video files (defaults to the current working directory)")
    parser.add_argument("--output-dir", default=None, help="Where to write transcripts (defaults to the folder containing each video)")
    parser.add_argument("--audio-dir", default=None, help="Where to write extracted audio files (defaults to the folder containing each video)")
    parser.add_argument("--model", default="large-v3", help="Whisper model name to load (default: large-v3)")
    parser.add_argument("--device", default="cuda" if sys.platform != "darwin" else "cpu", help="Torch device: cpu or cuda")
    parser.add_argument("--compute-type", default="float16", help="Torch compute type for the model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for transcription")
    parser.add_argument("--language", default=None, help="Optional language code, e.g. en")
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="Search subfolders recursively")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not search subfolders")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--resume", action="store_true", help="Skip videos that already have transcript JSON output")
    parser.add_argument("--ffmpeg-bin", default=os.environ.get("FFMPEG_BIN", "ffmpeg"), help="Path to ffmpeg binary or command")
    parser.add_argument("--diarize", dest="diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--no-diarize", dest="diarize", action="store_false", help="Disable speaker diarization")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for diarization (defaults to HF_TOKEN from the environment)")
    parser.add_argument("--face-only", action="store_true", help="Process only face narration and face scoring videos")
    parser.add_argument("--audio-only", action="store_true", help="Process only audio files in the input directory")
    parser.add_argument("--no-srt", dest="write_srt", action="store_false", help="Skip writing an SRT subtitle file")
    parser.add_argument("--no-txt", dest="write_txt", action="store_false", help="Skip writing a plain text transcript")
    parser.set_defaults(diarize=False, recursive=True, write_srt=True, write_txt=True)
    return parser.parse_args()


def find_video_files(input_dir: Path, recursive: bool, audio_only: bool = False, face_only: bool = False) -> list[Path]:
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.iterdir()

    extensions = AUDIO_EXTENSIONS if audio_only else VIDEO_EXTENSIONS
    files = [path for path in candidates if path.is_file() and path.suffix.lower() in extensions]
    if face_only and not audio_only:
        files = [path for path in files if "face_narration" in path.name.lower() or "face_scoring" in path.name.lower()]
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


def transcribe_with_faster_whisper(audio: Any, sample_rate: int, model_name: str, device: str, language: str | None) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX failed and faster-whisper fallback is not installed. "
            "Install it in your capture environment with: pip install faster-whisper"
        ) from exc

    compute_type = "float16" if device.startswith("cuda") else "int8"
    whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = whisper_model.transcribe(audio, beam_size=5, language=language, vad_filter=False)
    result_segments = [
        {"start": float(segment.start), "end": float(segment.end), "text": str(segment.text).strip()}
        for segment in segments
    ]
    return {
        "language": getattr(info, "language", language or "en"),
        "segments": result_segments,
        "word_segments": [],
    }


def format_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = int(total % 60)
    millis = int(round((total - int(total)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_output_paths(video_path: Path, output_dir: Path, audio_dir: Path, overwrite: bool) -> tuple[Path, Path, Path | None, Path | None]:
    stem = video_path.stem
    json_path = output_dir / f"{stem}.json"
    audio_path = audio_dir / f"{stem}.wav"
    srt_path = output_dir / f"{stem}.srt"
    txt_path = output_dir / f"{stem}.txt"
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {json_path}")
    return json_path, audio_path, srt_path, txt_path


def has_audio_stream(video_path: Path, ffmpeg_bin: str = "ffmpeg") -> bool:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        ffprobe_bin = shutil.which(os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe"))

    if not ffprobe_bin:
        return True

    cmd = [ffprobe_bin, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return bool(result.stdout.strip())


def find_fallback_audio(video_path: Path) -> Path | None:
    stem = video_path.stem
    if stem.endswith("_face_narration"):
        candidate = video_path.with_name(stem.replace("_face_narration", "_audio_narration") + ".wav")
        if candidate.exists():
            return candidate
    elif stem.endswith("_face_scoring"):
        candidate = video_path.with_name(stem.replace("_face_scoring", "_audio_scoring") + ".wav")
        if candidate.exists():
            return candidate

    prefix = stem.split("_")[0]
    if "face_narration" in stem:
        candidate = video_path.parent / f"{prefix}_audio_narration.wav"
        if candidate.exists():
            return candidate
    if "face_scoring" in stem:
        candidate = video_path.parent / f"{prefix}_audio_scoring.wav"
        if candidate.exists():
            return candidate

    return None


def extract_audio(video_path: Path, audio_path: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

    ffmpeg_cmd = ffmpeg_bin
    if os.path.sep not in ffmpeg_bin and (os.path.altsep is None or os.path.altsep not in ffmpeg_bin):
        resolved = shutil.which(ffmpeg_bin)
        if resolved:
            ffmpeg_cmd = resolved

    if not os.path.exists(ffmpeg_cmd) and shutil.which(ffmpeg_cmd) is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH. Install ffmpeg and ensure it is available as 'ffmpeg'.")

    if not has_audio_stream(video_path, ffmpeg_cmd):
        fallback_audio = find_fallback_audio(video_path)
        if fallback_audio is not None:
            return fallback_audio
        raise RuntimeError(f"Video has no audio stream and no fallback audio found: {video_path.name}")

    cmd = [ffmpeg_cmd, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return audio_path


def transcribe_video(video_path: Path, args: argparse.Namespace, output_dir: Path, audio_dir: Path) -> Path:
    # Workaround: some pyannote/whisperx environments import OpenTelemetry
    # and expect TraceFlags.RANDOM_TRACE_ID to exist. Define it if missing
    # to avoid runtime AttributeError on Windows with mismatched opentelemetry versions.
    try:
        import opentelemetry.trace as _ot
        if not hasattr(_ot.TraceFlags, "RANDOM_TRACE_ID"):
            setattr(_ot.TraceFlags, "RANDOM_TRACE_ID", 1)
    except Exception:
        # ignore if opentelemetry isn't installed or another error occurs
        pass

    try:
        import whisperx
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("WhisperX is not installed. Install it with: pip install -r requirements-whisperx.txt") from exc

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed. Install it before running WhisperX") from exc

    json_path, _, srt_path, txt_path = build_output_paths(video_path, output_dir, audio_dir, args.overwrite)
    is_audio = video_path.suffix.lower() in AUDIO_EXTENSIONS
    if is_audio:
        audio_path = video_path
        audio_source = str(audio_path)
        result = transcribe_with_faster_whisper(audio_source, 16000, args.model, args.device, args.language)
    else:
        audio_path = audio_dir / f"{video_path.stem}.wav"
        audio_path = extract_audio(video_path, audio_path, args.ffmpeg_bin)
        audio = whisperx.load_audio(str(audio_path))
        model = whisperx.load_model(args.model, device=args.device, compute_type=args.compute_type)

        # Disable internal VAD to avoid pyannote/torchcodec/opentelemetry issues on Windows
        try:
            result = model.transcribe(audio, batch_size=args.batch_size, language=args.language, print_progress=False, vad=False)
        except TypeError:
            # Older whisperx versions may not accept the vad parameter; fall back
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
        "audio": str(audio_path),
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
        "audio": str(audio_path),
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

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 2

    if args.audio_only and args.face_only:
        print("Cannot use both --audio-only and --face-only together.")
        return 2

    video_files = find_video_files(input_dir, args.recursive, audio_only=args.audio_only, face_only=args.face_only)
    if not video_files:
        if args.audio_only:
            print(f"No supported audio files found in {input_dir}")
        elif args.face_only:
            print(f"No face narration or face scoring video files found in {input_dir}")
        else:
            print(f"No supported video files found in {input_dir}")
        return 0

    total_files = len(video_files)
    print(f"Found {total_files} file(s) in {input_dir}")
    for index, video_path in enumerate(video_files, start=1):
        print(f"[{index}/{total_files}] Starting transcription: {video_path.name}")
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else video_path.parent
        audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else video_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{video_path.stem}.json"
        if args.resume and json_path.exists():
            print(f"Skipping {video_path.name}: {json_path.name} already exists")
            continue
        try:
            output_path = transcribe_video(video_path, args, output_dir, audio_dir)
            print(f"[{index}/{total_files}] Completed {video_path.name} -> {output_path}")
        except FileExistsError as exc:
            print(str(exc))
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Failed for {video_path.name}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    main()
