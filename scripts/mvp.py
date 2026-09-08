#!/usr/bin/env python3
"""Staged MVP performance-review video pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}


@dataclass
class TrialAssets:
    trial_dir: str
    front_video: str | None
    side_video: str | None
    sync_manifest: str | None
    pause_timestamps: str | None
    narration_audio: str | None
    face_video: str | None
    heart_rate_csv: str | None
    subtitle_candidates: list[str]
    calibration_front: str | None
    calibration_side: str | None
    missing_required: list[str]


def first_matching(paths: list[Path], patterns: tuple[str, ...]) -> Path | None:
    for path in paths:
        lower_name = path.name.lower()
        if any(pattern in lower_name for pattern in patterns):
            return path
    return None


def discover_trial(trial_dir: Path, calibration_dir: Path | None = None) -> TrialAssets:
    trial_dir = trial_dir.resolve()
    all_files = [path for path in trial_dir.rglob("*") if path.is_file()]
    gopro_dir = trial_dir / "gopro_footage"
    gopro_files = sorted(
        path for path in gopro_dir.glob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ) if gopro_dir.is_dir() else []

    front = first_matching(gopro_files, ("front",))
    side = first_matching(gopro_files, ("side",))
    manifest = first_matching(all_files, ("sync_manifest",))
    pause_file = first_matching(all_files, ("review_timestamps",))
    narration_audio = first_matching(
        sorted(path for path in all_files if path.suffix.lower() in AUDIO_EXTENSIONS),
        ("audio_narration", "audio_commentary", "narration"),
    )
    face_video = first_matching(
        all_files,
        ("face_narration", "face_cam"),
    )
    heart_rate = first_matching(all_files, ("hr_full_session",))
    subtitle_candidates = sorted(
        path for path in all_files
        if path.suffix.lower() in SUBTITLE_EXTENSIONS
        or (
            path.suffix.lower() == ".json"
            and any(token in path.name.lower() for token in ("subtitle", "transcript", "whisper"))
        )
    )

    calibration_dir = calibration_dir.resolve() if calibration_dir else trial_dir
    calibration_front = calibration_dir / "front_calibration.json"
    calibration_side = calibration_dir / "side_calibration.json"

    missing = []
    for label, path in (
        ("front GoPro video", front),
        ("side GoPro video", side),
        ("sync manifest", manifest),
        ("review pause timestamps", pause_file),
        ("narration audio", narration_audio),
        ("face narration video", face_video),
        ("full-session heart-rate CSV", heart_rate),
    ):
        if path is None:
            missing.append(label)
    if not calibration_front.is_file():
        missing.append(f"front calibration ({calibration_front})")
    if not calibration_side.is_file():
        missing.append(f"side calibration ({calibration_side})")
    if not subtitle_candidates:
        missing.append("timestamped subtitle file")

    return TrialAssets(
        trial_dir=str(trial_dir),
        front_video=str(front) if front else None,
        side_video=str(side) if side else None,
        sync_manifest=str(manifest) if manifest else None,
        pause_timestamps=str(pause_file) if pause_file else None,
        narration_audio=str(narration_audio) if narration_audio else None,
        face_video=str(face_video) if face_video else None,
        heart_rate_csv=str(heart_rate) if heart_rate else None,
        subtitle_candidates=[str(path) for path in subtitle_candidates],
        calibration_front=str(calibration_front) if calibration_front.is_file() else None,
        calibration_side=str(calibration_side) if calibration_side.is_file() else None,
        missing_required=missing,
    )


def load_json(path: str | None) -> Any:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_timing(assets: TrialAssets) -> dict[str, Any]:
    manifest = load_json(assets.sync_manifest) or {}
    pause_events = load_json(assets.pause_timestamps) or []
    events = manifest.get("events", []) if isinstance(manifest, dict) else []
    event_times: dict[str, list[float]] = {}
    for event in events:
        if isinstance(event, dict) and "event" in event and "wall_time" in event:
            event_times.setdefault(event["event"], []).append(event["wall_time"])

    pauses = []
    active_pause = None
    for event in pause_events if isinstance(pause_events, list) else []:
        if event.get("type") == "pause":
            active_pause = event
        elif event.get("type") == "resume" and active_pause:
            pauses.append({
                "pause_wall_time": active_pause.get("wall_time"),
                "resume_wall_time": event.get("wall_time"),
                "performance_position_sec": active_pause.get("video_position_sec"),
                "duration_sec": event.get("wall_time", 0) - active_pause.get("wall_time", 0),
            })
            active_pause = None

    return {
        "manifest_events": event_times,
        "pause_intervals": pauses,
        "unpaired_pause": active_pause,
    }


def media_info(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        import cv2

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return {"path": path, "opened": False}
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        return {
            "path": path,
            "opened": True,
            "fps": fps,
            "frames": frames,
            "duration_sec": frames / fps if fps else None,
            "width": width,
            "height": height,
        }
    except ImportError:
        return {"path": path, "opened": None, "error": "opencv-python is unavailable"}


def build_timing_report(assets: TrialAssets) -> dict[str, Any]:
    timing = summarize_timing(assets)
    manifest_events = timing["manifest_events"]
    pause_events = load_json(assets.pause_timestamps) or []
    playback_start = next(
        (event.get("wall_time") for event in pause_events if event.get("type") in {"resume", "pause"}),
        None,
    )
    review_start = playback_start or (manifest_events.get("face_recorder_start") or [None])[0]
    review_player_events = load_json(assets.sync_manifest) or {}
    review_duration = next(
        (
            event.get("duration_sec")
            for event in review_player_events.get("events", [])
            if event.get("event") == "review_video_player_shown"
            and event.get("duration_sec") is not None
        ),
        None,
    )
    pause_duration = sum(
        interval["duration_sec"]
        for interval in timing["pause_intervals"]
        if interval.get("duration_sec") is not None
    )
    review_stop_event = next(
        (
            event.get("wall_time")
            for event in review_player_events.get("events", [])
            if event.get("event") == "review_playback_stop"
        ),
        None,
    )
    review_end = (
        review_start + review_duration + pause_duration
        if review_start is not None and review_duration is not None
        else review_stop_event
    )
    performance_start = (manifest_events.get("overhead_recorder_start") or [None])[0]
    performance_end = (manifest_events.get("overhead_recorder_stop") or [None])[-1]

    timing.update({
        "review_wall_start": review_start,
        "review_wall_end": review_end,
        "review_wall_duration_sec": review_end - review_start if review_start and review_end else None,
        "review_playback_duration_sec": review_duration,
        "review_pause_duration_sec": pause_duration,
        "performance_wall_start": performance_start,
        "performance_wall_end": performance_end,
        "performance_wall_duration_sec": performance_end - performance_start if performance_start and performance_end else None,
        "wall_clock_alignment_status": (
            "available"
            if performance_start is not None and review_start is not None
            else "incomplete_manifest"
        ),
        "source_media": {
            "front": media_info(assets.front_video),
            "side": media_info(assets.side_video),
            "face": media_info(assets.face_video),
        },
    })
    return timing


def build_pause_timeline(assets: TrialAssets) -> dict[str, Any]:
    timing = build_timing_report(assets)
    review_start = timing["review_wall_start"]
    review_end = timing["review_wall_end"]
    if review_start is None or review_end is None:
        raise ValueError("Manifest does not contain review start and end events")

    pause_events = load_json(assets.pause_timestamps) or []
    segments = []
    review_cursor = 0.0
    source_cursor = 0.0
    for index, event in enumerate(pause_events):
        if event.get("type") != "pause":
            continue
        pause_time = event.get("wall_time")
        pause_position = event.get("video_position_sec")
        if pause_time is None or pause_position is None:
            raise ValueError(f"Pause event {index} is missing wall_time or video_position_sec")
        pause_review_time = pause_time - review_start
        if pause_review_time < review_cursor:
            raise ValueError(f"Pause event {index} is out of order")
        if pause_position < source_cursor:
            raise ValueError(f"Pause event {index} moves backward in source video")
        segments.append({
            "type": "play",
            "review_start_sec": review_cursor,
            "review_end_sec": pause_review_time,
            "source_start_sec": source_cursor,
            "source_end_sec": pause_position,
        })
        resume = next(
            (candidate for candidate in pause_events[index + 1:] if candidate.get("type") == "resume"),
            None,
        )
        if resume is None:
            raise ValueError(f"Pause event {index} has no matching resume")
        resume_time = resume.get("wall_time")
        if resume_time is None or resume_time < pause_time:
            raise ValueError(f"Pause event {index} has an invalid resume")
        segments.append({
            "type": "hold",
            "review_start_sec": pause_review_time,
            "review_end_sec": resume_time - review_start,
            "source_start_sec": pause_position,
            "source_end_sec": pause_position,
        })
        review_cursor = resume_time - review_start
        source_cursor = pause_position

    segments.append({
        "type": "play",
        "review_start_sec": review_cursor,
        "review_end_sec": review_end - review_start,
        "source_start_sec": source_cursor,
        "source_end_sec": source_cursor + (review_end - review_start - review_cursor),
    })
    return {
        "review_duration_sec": review_end - review_start,
        "segments": segments,
        "note": "Source positions assume the reviewed performance timeline starts at source second zero.",
    }


def print_inspection(assets: TrialAssets, timing: dict[str, Any]) -> None:
    print(f"Trial: {assets.trial_dir}")
    for label, value in (
        ("Front GoPro", assets.front_video),
        ("Side GoPro", assets.side_video),
        ("Sync manifest", assets.sync_manifest),
        ("Pause timestamps", assets.pause_timestamps),
        ("Narration audio", assets.narration_audio),
        ("Face video", assets.face_video),
        ("Heart rate", assets.heart_rate_csv),
        ("Front calibration", assets.calibration_front),
        ("Side calibration", assets.calibration_side),
    ):
        print(f"{label}: {value or 'MISSING'}")
    print(f"Subtitle candidates: {len(assets.subtitle_candidates)}")
    for subtitle in assets.subtitle_candidates:
        print(f"  - {subtitle}")
    print(f"Pause intervals: {len(timing['pause_intervals'])}")
    if timing["unpaired_pause"]:
        print("WARNING: unpaired pause event")
    if assets.missing_required:
        print("Missing prerequisites:")
        for missing in assets.missing_required:
            print(f"  - {missing}")
    else:
        print("All discovered prerequisites are present.")


def inspect_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    timing = summarize_timing(assets)
    print_inspection(assets, timing)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"assets": asdict(assets), "timing": timing}, indent=2),
            encoding="utf-8",
        )
        print(f"Report: {report_path.resolve()}")
    return 0 if not assets.missing_required and not timing["unpaired_pause"] else 2


def timing_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    report = {
        "trial_dir": assets.trial_dir,
        "assets": asdict(assets),
        "timing": build_timing_report(assets),
    }
    output = args.report or args.trial / "MVP" / "timing_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    timing = report["timing"]
    print(f"Timing report: {output.resolve()}")
    print(f"Review duration: {timing['review_wall_duration_sec']}")
    print(f"Pause intervals: {len(timing['pause_intervals'])}")
    print(f"Wall-clock alignment: {timing['wall_clock_alignment_status']}")
    if timing["unpaired_pause"]:
        print("WARNING: unpaired pause event")
        return 2
    return 0


def pause_map_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    if not assets.pause_timestamps or not assets.sync_manifest:
        print("ERROR: sync manifest and pause timestamps are required", file=sys.stderr)
        return 2
    try:
        pause_timeline = build_pause_timeline(assets)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output = args.report or args.trial / "MVP" / "pause_timeline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pause_timeline, indent=2), encoding="utf-8")
    print(f"Pause timeline: {output.resolve()}")
    for segment in pause_timeline["segments"]:
        print(
            f"{segment['type']:>4} review {segment['review_start_sec']:.3f}-"
            f"{segment['review_end_sec']:.3f}s | source "
            f"{segment['source_start_sec']:.3f}-{segment['source_end_sec']:.3f}s"
        )
    return 0


def read_heart_rate(path: str) -> list[dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({"timestamp": float(row["timestamp"]), "bpm": float(row["bpm"])})
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def chart_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    if not assets.heart_rate_csv:
        print("ERROR: full-session heart-rate CSV was not found", file=sys.stderr)
        return 2

    timing = build_timing_report(assets)
    review_start = timing["review_wall_start"]
    review_end = timing["review_wall_end"]
    if review_start is None or review_end is None:
        print("ERROR: review wall-clock bounds are missing", file=sys.stderr)
        return 2

    rows = read_heart_rate(assets.heart_rate_csv)
    visible = [
        {"time_sec": row["timestamp"] - review_start, "bpm": row["bpm"]}
        for row in rows
        if review_start <= row["timestamp"] <= review_end
    ]
    if not visible:
        print("ERROR: no heart-rate readings overlap the review session", file=sys.stderr)
        return 2

    output_dir = args.output_dir or args.trial / "MVP" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_data = {
        "review_start_wall_time": review_start,
        "review_end_wall_time": review_end,
        "duration_sec": review_end - review_start,
        "min_bpm": min(row["bpm"] for row in visible),
        "max_bpm": max(row["bpm"] for row in visible),
        "readings": visible,
        "marker_time_sec": args.at_sec,
    }
    data_path = output_dir / "heart_rate_chart.json"
    data_path.write_text(json.dumps(chart_data, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Chart data written: {data_path}")
        print("WARNING: matplotlib is unavailable; PNG preview was not created")
        return 0

    x_values = [row["time_sec"] for row in visible]
    y_values = [row["bpm"] for row in visible]
    figure, axis = plt.subplots(figsize=(12, 2.2), dpi=150)
    axis.plot(x_values, y_values, color="#2b6cb0", linewidth=1.5)
    marker_time = max(0.0, min(args.at_sec, review_end - review_start))
    marker_value = min(visible, key=lambda row: abs(row["time_sec"] - marker_time))["bpm"]
    axis.axvline(marker_time, color="#c53030", linewidth=1.5)
    axis.scatter([marker_time], [marker_value], color="#c53030", zorder=3)
    axis.set_xlim(0, review_end - review_start)
    axis.set_ylim(max(0, chart_data["min_bpm"] - 5), chart_data["max_bpm"] + 5)
    axis.set_xlabel("Review time (seconds)")
    axis.set_ylabel("BPM")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    image_path = output_dir / "heart_rate_chart_preview.png"
    figure.savefig(image_path)
    plt.close(figure)
    print(f"Chart data: {data_path}")
    print(f"Chart preview: {image_path}")
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    if not assets.front_video or not assets.side_video:
        print("ERROR: front and side GoPro videos are required", file=sys.stderr)
        return 2
    output_dir = args.output_dir or args.trial / "MVP" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"Would undistort front: {assets.front_video}")
        print(f"Would undistort side: {assets.side_video}")
        print(f"Output directory: {output_dir}")
        print(f"Front calibration: {assets.calibration_front or 'MISSING'}")
        print(f"Side calibration: {assets.calibration_side or 'MISSING'}")
        return 0
    if not assets.calibration_front or not assets.calibration_side:
        print("ERROR: front_calibration.json and side_calibration.json are required", file=sys.stderr)
        return 2

    try:
        from undistort import undistort_video
    except ImportError as exc:
        print(f"ERROR: could not import undistort.py: {exc}", file=sys.stderr)
        return 1

    outputs = {}
    for camera, source, calibration in (
        ("front", assets.front_video, assets.calibration_front),
        ("side", assets.side_video, assets.calibration_side),
    ):
        output = output_dir / f"{Path(source).stem}_undistorted.mp4"
        print(f"Undistorting {camera}: {source}")
        frames = undistort_video(Path(source), output, Path(calibration), alpha=args.alpha)
        outputs[camera] = {"source": source, "output": str(output), "frames": frames}

    prepared_path = output_dir / "prepared_assets.json"
    prepared_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Prepared assets: {prepared_path}")
    return 0


def render_preview_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    prepared_path = args.prepared or args.trial / "MVP" / "generated" / "prepared_assets.json"
    if not prepared_path.is_file():
        print(f"ERROR: prepared asset report not found: {prepared_path}", file=sys.stderr)
        print("Run the prepare command after calibration first.", file=sys.stderr)
        return 2
    if not assets.face_video or not assets.narration_audio:
        print("ERROR: face video and narration audio are required", file=sys.stderr)
        return 2

    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    front = Path(prepared["front"]["output"])
    side = Path(prepared["side"]["output"])
    chart = args.chart or args.trial / "MVP" / "generated" / "heart_rate_chart_preview.png"
    if not front.is_file() or not side.is_file():
        print("ERROR: prepared front/side video is missing", file=sys.stderr)
        return 2
    if not chart.is_file():
        print(f"ERROR: chart preview not found: {chart}", file=sys.stderr)
        return 2

    output = args.output or args.trial / "MVP" / "preview.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("ERROR: FFmpeg is required for rendering", file=sys.stderr)
            return 1

    filter_parts = [
        "[0:v]scale=960:540,fps=30,setpts=PTS-STARTPTS[left]",
        "[1:v]scale=960:540,fps=30,setpts=PTS-STARTPTS[right]",
        "[left][right]hstack=inputs=2[performance]",
        "[2:v]scale=320:-1,setpts=PTS-STARTPTS[face]",
        "[performance][face]overlay=W-w-20:20[withface]",
        "[3:v]scale=1800:180[chart]",
        "[withface][chart]overlay=60:H-h-20[video]",
        "[0:a]volume=0.125[performance_audio]",
        "[4:a]volume=1.0[narration_audio]",
        "[performance_audio][narration_audio]amix=inputs=2:duration=longest[audio]",
    ]
    if args.subtitles:
        subtitle_path = str(args.subtitles.resolve()).replace("\\", "/").replace(":", r"\:")
        filter_parts.append(f"[video]subtitles='{subtitle_path}'[captioned]")
        video_label = "[captioned]"
    else:
        video_label = "[video]"

    command = [
        ffmpeg,
        "-y",
        "-ss", str(args.front_start_sec), "-i", str(front),
        "-ss", str(args.side_start_sec), "-i", str(side),
        "-stream_loop", "-1", "-ss", str(args.review_start_sec), "-i", assets.face_video,
        "-loop", "1", "-i", str(chart),
        "-ss", str(args.review_start_sec), "-i", assets.narration_audio,
        "-filter_complex", ";".join(filter_parts),
        "-map", video_label,
        "-map", "[audio]",
        "-t", str(args.duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-r", "30",
        "-c:a", "aac", "-shortest",
        str(output),
    ]
    print(f"Rendering preview: {output}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        return result.returncode
    print(f"Created: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Staged MVP review-video pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="discover trial inputs and validate metadata")
    inspect.add_argument("--trial", type=Path, required=True)
    inspect.add_argument("--calibration-dir", type=Path)
    inspect.add_argument("--report", type=Path)
    inspect.set_defaults(handler=inspect_command)
    timing = subparsers.add_parser("timing", help="write and validate trial timing metadata")
    timing.add_argument("--trial", type=Path, required=True)
    timing.add_argument("--calibration-dir", type=Path)
    timing.add_argument("--report", type=Path)
    timing.set_defaults(handler=timing_command)
    pause_map = subparsers.add_parser("pause-map", help="write the pause-aware source timeline")
    pause_map.add_argument("--trial", type=Path, required=True)
    pause_map.add_argument("--calibration-dir", type=Path)
    pause_map.add_argument("--report", type=Path)
    pause_map.set_defaults(handler=pause_map_command)
    chart = subparsers.add_parser("chart", help="generate a heart-rate chart preview and data")
    chart.add_argument("--trial", type=Path, required=True)
    chart.add_argument("--calibration-dir", type=Path)
    chart.add_argument("--output-dir", type=Path)
    chart.add_argument("--at-sec", type=float, default=0.0, help="marker position in review seconds")
    chart.set_defaults(handler=chart_command)
    prepare = subparsers.add_parser("prepare", help="undistort both GoPro inputs")
    prepare.add_argument("--trial", type=Path, required=True)
    prepare.add_argument("--calibration-dir", type=Path)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--alpha", type=float, default=1.0)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(handler=prepare_command)
    render = subparsers.add_parser("render-preview", help="render a short composition preview")
    render.add_argument("--trial", type=Path, required=True)
    render.add_argument("--calibration-dir", type=Path)
    render.add_argument("--prepared", type=Path)
    render.add_argument("--chart", type=Path)
    render.add_argument("--subtitles", type=Path)
    render.add_argument("--output", type=Path)
    render.add_argument("--duration", type=float, default=15.0)
    render.add_argument("--front-start-sec", type=float, default=0.0)
    render.add_argument("--side-start-sec", type=float, default=0.0)
    render.add_argument("--review-start-sec", type=float, default=0.0)
    render.set_defaults(handler=render_preview_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
