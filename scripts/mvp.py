#!/usr/bin/env python3
"""Staged MVP performance-review video pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}

# --- MVP composition layout (approved design) ---
# Front/side and face-cam sizes are exact 16:9 multiples so scaling never
# stretches or crops the source video; only the chart card and canvas
# background may be resized freely since they are generated graphics.
LAYOUT_MARGIN = 20
LAYOUT_GAP = 10
LAYOUT_RADIUS = 0
LAYOUT_VIDEO_W, LAYOUT_VIDEO_H = 944, 531   # 944=16*59, 531=9*59 -> exact 16:9
LAYOUT_FACE_W, LAYOUT_FACE_H = 544, 306     # 544=16*34, 306=9*34 -> exact 16:9
LAYOUT_SUB_H = 60
LAYOUT_BG_HEX = "#121315"
LAYOUT_SUB_BG = (36, 38, 43)
LAYOUT_CARD_LIGHT_HEX = "#fcfcfb"
LAYOUT_CARD_LIGHT = (252, 252, 251)
LAYOUT_CHART_COLOR = "#e34948"  # red: dataviz skill categorical slot 8
LAYOUT_CHART_CARD_PAD = 10
# matplotlib subplots_adjust fractions for the chart card's plotted axes area
# (shared with render commands so the animated marker overlay lines up exactly
# with the static line/fill baked into the chart image).
CHART_PLOT_LEFT_FRAC = 0.065
CHART_PLOT_RIGHT_FRAC = 0.99
CHART_PLOT_TOP_FRAC = 0.96
CHART_PLOT_BOTTOM_FRAC = 0.24


def chart_plot_pixel_bounds(chart_x: int, chart_y: int) -> tuple[float, float, float, float]:
    """Pixel bounds (left, right, top, bottom) of the chart's plotted axes area
    within the full composited canvas, given the chart card's top-left position."""
    inner_w = LAYOUT_CHART_W - LAYOUT_CHART_CARD_PAD * 2
    inner_h = LAYOUT_FACE_H - LAYOUT_CHART_CARD_PAD * 2
    base_x = chart_x + LAYOUT_CHART_CARD_PAD
    base_y = chart_y + LAYOUT_CHART_CARD_PAD
    left = base_x + CHART_PLOT_LEFT_FRAC * inner_w
    right = base_x + CHART_PLOT_RIGHT_FRAC * inner_w
    top = base_y + (1 - CHART_PLOT_TOP_FRAC) * inner_h
    bottom = base_y + (1 - CHART_PLOT_BOTTOM_FRAC) * inner_h
    return left, right, top, bottom

LAYOUT_CANVAS_W = LAYOUT_MARGIN * 2 + LAYOUT_VIDEO_W * 2 + LAYOUT_GAP
LAYOUT_CHART_W = LAYOUT_CANVAS_W - LAYOUT_MARGIN * 2 - LAYOUT_FACE_W - LAYOUT_GAP
LAYOUT_CANVAS_H = LAYOUT_MARGIN * 3 + LAYOUT_VIDEO_H + LAYOUT_SUB_H + LAYOUT_FACE_H
# libx264 requires even width/height; pad with an imperceptible extra row/col.
if LAYOUT_CANVAS_W % 2:
    LAYOUT_CANVAS_W += 1
if LAYOUT_CANVAS_H % 2:
    LAYOUT_CANVAS_H += 1


def _rounded_mask(size: tuple[int, int], radius: int):
    from PIL import Image, ImageDraw

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return mask


def _make_shadow(size: tuple[int, int], radius: int, pad: int = 12, blur: int = 8):
    from PIL import Image, ImageDraw, ImageFilter

    layer = Image.new("RGBA", (size[0] + pad * 2, size[1] + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([pad, pad + 2, pad + size[0], pad + 2 + size[1]], radius=radius, fill=(0, 0, 0, 110))
    return layer.filter(ImageFilter.GaussianBlur(blur))


def _make_subtitle_bg(width: int, height: int, radius: int, color: tuple[int, int, int]):
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius, fill=(*color, 255))
    return layer


def generate_layout_assets(output_dir: Path) -> dict[str, Path]:
    """Generate the static rounded-corner masks, drop shadows, and subtitle-bar
    background used to composite the approved MVP layout. Cheap and
    deterministic, so regenerating every render is simpler than caching."""
    paths: dict[str, Path] = {}

    mask_video_path = output_dir / "_mask_video.png"
    _rounded_mask((LAYOUT_VIDEO_W, LAYOUT_VIDEO_H), LAYOUT_RADIUS).save(mask_video_path)
    paths["mask_video"] = mask_video_path

    mask_face_path = output_dir / "_mask_face.png"
    _rounded_mask((LAYOUT_FACE_W, LAYOUT_FACE_H), LAYOUT_RADIUS).save(mask_face_path)
    paths["mask_face"] = mask_face_path

    shadow_video_path = output_dir / "_shadow_video.png"
    _make_shadow((LAYOUT_VIDEO_W, LAYOUT_VIDEO_H), LAYOUT_RADIUS).save(shadow_video_path)
    paths["shadow_video"] = shadow_video_path

    shadow_face_path = output_dir / "_shadow_face.png"
    _make_shadow((LAYOUT_FACE_W, LAYOUT_FACE_H), LAYOUT_RADIUS).save(shadow_face_path)
    paths["shadow_face"] = shadow_face_path

    subtitle_bg_path = output_dir / "_subtitle_bg.png"
    _make_subtitle_bg(LAYOUT_CANVAS_W - LAYOUT_MARGIN * 2, LAYOUT_SUB_H, LAYOUT_RADIUS, LAYOUT_SUB_BG).save(subtitle_bg_path)
    paths["subtitle_bg"] = subtitle_bg_path

    # Heart-rate marker (moving vertical line + dot). Positioned dynamically at
    # render time via the `overlay` filter's per-frame `t` expression -- NOT
    # `drawbox`: in this ffmpeg build drawbox's x/y are evaluated once at init
    # (before `t` exists), not per frame, so a "moving" drawbox silently
    # renders at a nonsensical fixed position instead of animating.
    from PIL import Image, ImageDraw

    chart_x = LAYOUT_MARGIN + LAYOUT_FACE_W + LAYOUT_GAP
    bottom_y = LAYOUT_MARGIN + LAYOUT_VIDEO_H + LAYOUT_MARGIN + LAYOUT_SUB_H + LAYOUT_MARGIN
    _, _, plot_top, plot_bottom = chart_plot_pixel_bounds(chart_x, bottom_y)
    line_height = max(1, int(round(plot_bottom - plot_top)))

    marker_line_path = output_dir / "_marker_line.png"
    Image.new("RGBA", (2, line_height), (0x52, 0x51, 0x4e, 130)).save(marker_line_path)
    paths["marker_line"] = marker_line_path

    dot_size = 12
    accent_hex = LAYOUT_CHART_COLOR.lstrip("#")
    accent_rgb = tuple(int(accent_hex[i:i + 2], 16) for i in (0, 2, 4))
    dot = Image.new("RGBA", (dot_size, dot_size), (0, 0, 0, 0))
    ImageDraw.Draw(dot).ellipse([0, 0, dot_size - 1, dot_size - 1], fill=(*accent_rgb, 255))
    marker_dot_path = output_dir / "_marker_dot.png"
    dot.save(marker_dot_path)
    paths["marker_dot"] = marker_dot_path

    return paths


SUBTITLE_FONT_PATH = "C:/Windows/Fonts/arial.ttf"


def parse_srt(path: Path) -> list[dict[str, Any]]:
    """Parse an .srt file into a list of {start, end, text} cues (seconds)."""
    text = path.read_text(encoding="utf-8-sig")
    time_re = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    cues: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        timing_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_idx is None:
            continue
        match = time_re.search(lines[timing_idx])
        if not match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
        cue_text = " ".join(lines[timing_idx + 1:]).strip()
        if cue_text:
            cues.append({"start": start, "end": end, "text": cue_text})
    return cues


def _fit_subtitle_line(text: str, max_width_px: int, font_path: str, base_size: int = 28, min_size: int = 16) -> tuple[str, int]:
    """Shrink font size to fit one line; truncate with an ellipsis as a last resort."""
    from PIL import ImageFont

    size = base_size
    while size >= min_size:
        font = ImageFont.truetype(font_path, size)
        if font.getlength(text) <= max_width_px:
            return text, size
        size -= 1
    font = ImageFont.truetype(font_path, min_size)
    while len(text) > 1 and font.getlength(text + "\u2026") > max_width_px:
        text = text[:-1]
    return text.rstrip() + "\u2026", min_size


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
    heart_rate_csvs: list[str]
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


def find_gopro_video(gopro_files: list[Path], camera: str) -> Path | None:
    """Find the front/side GoPro file for a camera, preferring a '_COMBINED'
    variant when present. The GoPro splits recordings into ~4GB chapters;
    some trials only got the first chapter, with a separate '<camera>_COMBINED'
    file created later holding the full merged recording. Plain alphabetical
    matching picks the truncated original ('.' sorts before '_'), silently
    cutting the performance short for every trial that hit the chapter split."""
    candidates = [f for f in gopro_files if camera in f.name.lower() and "sync_full" not in f.name.lower()]
    combined = [f for f in candidates if "combined" in f.name.lower()]
    if combined:
        return sorted(combined)[0]
    return sorted(candidates)[0] if candidates else None


def find_heart_rate_csvs(all_files: list[Path]) -> list[Path]:
    """All hr_full_session CSV files for a trial, in the order their rows
    should be concatenated. Two real cases this covers (found via P20 and
    P26 failing to render with 'no heart-rate readings overlap the review
    session'):
      - a session split into '..._1of2.csv' / '..._2of2.csv' parts (same
        class of issue as the GoPro chapter split -- plain first-match
        picked only the first part, missing the data the review window
        actually falls in);
      - a separate '..._retry.csv' alongside the original (the original
        session ended early/failed, and the retry covers what it missed).
    Rather than encode which one 'wins', every matching file is returned and
    the caller merges + time-sorts all their rows -- a retry or extra part's
    out-of-window rows are harmless, since the chart's window filter drops
    them anyway."""
    candidates = [
        path for path in all_files
        if path.suffix.lower() == ".csv" and "hr_full_session" in path.stem.lower()
    ]

    def sort_key(path: Path) -> tuple[int, int]:
        match = re.search(r"(\d+)of\d+", path.stem, re.IGNORECASE)
        part = int(match.group(1)) if match else 1
        is_retry = 1 if "retry" in path.stem.lower() else 0
        return (part, is_retry)

    return sorted(candidates, key=sort_key)


SYNC_OFFSETS_FILENAME = "sync_offsets.json"


def sync_offsets_path(trial_dir: Path) -> Path:
    return trial_dir / "MVP" / SYNC_OFFSETS_FILENAME


def load_sync_offsets(trial_dir: Path) -> dict[str, Any]:
    path = sync_offsets_path(trial_dir)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_sync_offsets(trial_dir: Path, front_offset_sec: float, side_offset_sec: float, note: str = "") -> Path:
    path = sync_offsets_path(trial_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "front_offset_sec": front_offset_sec,
        "side_offset_sec": side_offset_sec,
        "note": note,
        "meaning": (
            "Seconds into the undistorted front/side GoPro file where the review "
            "timeline's t=0 (review start) occurs. Set manually with set-offsets "
            "after visually confirming alignment; render commands refuse to run "
            "without this file unless --front-start-sec/--side-start-sec are passed explicitly."
        ),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def get_video_creation_time(path: str) -> float | None:
    """Read a video's embedded creation_time and interpret it as a naive
    local timestamp (matching the experiment manifest's wall_time domain).

    Empirically the 'Z' (UTC) suffix does not reflect a true UTC conversion:
    cross-validated on P01 against the known-accurate front/side offset from
    the audio-based sync_info.json (9.68s), the local-time interpretation of
    creation_time gives 11.0s -- within ~1.3s, consistent with creation_time's
    whole-second resolution rather than a real UTC/local mismatch.
    """
    ffprobe = None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        candidate = Path(ffmpeg).with_name("ffprobe.exe")
        if candidate.is_file():
            ffprobe = str(candidate)
    if not ffprobe:
        ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format_tags=creation_time", "-of", "json", str(path)],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout or "{}")
        raw = data.get("format", {}).get("tags", {}).get("creation_time")
        if not raw:
            return None
        return datetime.datetime.fromisoformat(raw.replace("Z", "")).timestamp()
    except Exception:
        return None


def compute_gopro_offsets(assets: "TrialAssets", timing: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return (front_offset, side_offset, overhead_start) such that
    front_position(overhead_position) = overhead_position + front_offset, or
    None if required metadata is unavailable. See suggest_offsets_command for
    why this is anchored to overhead_recorder_start, not review playback."""
    overhead_start = timing.get("performance_wall_start")
    if overhead_start is None or not assets.front_video or not assets.side_video:
        return None
    front_ct = get_video_creation_time(assets.front_video)
    side_ct = get_video_creation_time(assets.side_video)
    if front_ct is None or side_ct is None:
        return None
    return overhead_start - front_ct, overhead_start - side_ct, overhead_start


def suggest_offsets_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    if not assets.front_video or not assets.side_video:
        print("ERROR: front and side GoPro videos are required", file=sys.stderr)
        return 2

    try:
        timing = build_timing_report(assets)
    except Exception as exc:
        print(f"ERROR: could not compute timing report: {exc}", file=sys.stderr)
        return 2

    result = compute_gopro_offsets(assets, timing)
    if result is None:
        print("ERROR: overhead recorder start time or GoPro creation_time metadata is unavailable", file=sys.stderr)
        return 2
    front_offset, side_offset, _ = result
    print(f"Suggested front_offset_sec (add to an overhead video_position_sec): {front_offset:.2f}")
    print(f"Suggested side_offset_sec  (add to an overhead video_position_sec): {side_offset:.2f}")
    print(
        "Precision note: GoPro creation_time has whole-second resolution, so this "
        "estimate is accurate to roughly +/-1-2s -- good enough for a review "
        "video's freeze-frame timing, but worth a quick visual spot check before "
        "trusting it across the full batch."
    )

    front_duration = media_info(assets.front_video).get("duration_sec")
    side_duration = media_info(assets.side_video).get("duration_sec")
    pause_events = load_json(assets.pause_timestamps) or []
    positions = sorted({e["video_position_sec"] for e in pause_events if "video_position_sec" in e})
    if positions:
        print(f"\nChecking {len(positions)} pause/resume position(s) against front/side recorded duration:")
        for pos in positions:
            front_pos = pos + front_offset
            side_pos = pos + side_offset
            front_ok = front_duration is not None and 0 <= front_pos <= front_duration
            side_ok = side_duration is not None and 0 <= side_pos <= side_duration
            flag = "OK" if front_ok and side_ok else "OUT OF RANGE"
            print(f"  overhead={pos:.1f}s -> front={front_pos:.1f}s side={side_pos:.1f}s [{flag}]")

    if args.save:
        path = save_sync_offsets(
            args.trial, front_offset, side_offset,
            note="auto-suggested from GoPro creation_time metadata (offset vs. overhead_recorder_start)",
        )
        print(f"Saved: {path}")
    return 0


def _resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _extract_thumbnail(ffmpeg: str, video_path: str, position_sec: float, out_path: Path, width: int = 320) -> bool:
    command = [
        ffmpeg, "-y", "-v", "error",
        "-ss", f"{max(position_sec, 0):.3f}", "-i", video_path,
        "-frames:v", "1", "-update", "1", "-vf", f"scale={width}:-1",
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and out_path.is_file()


def _parse_timecode(value: str) -> float:
    """Accept plain seconds ('332', '332.5') or 'MM:SS'/'H:MM:SS'."""
    if ":" not in value:
        return float(value)
    parts = [float(p) for p in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def record_match_command(args: argparse.Namespace) -> int:
    """Compute front/side offsets from one or two manually-found matching
    timestamps (overhead vs. front, optionally also overhead vs. side) --
    the reliable alternative to creation_time/motion-correlation, both of
    which proved too imprecise or outright unreliable across the dataset."""
    overhead_pos = _parse_timecode(args.overhead_time)
    front_pos = _parse_timecode(args.front_time)
    front_offset = front_pos - overhead_pos

    if args.side_time is not None:
        side_pos = _parse_timecode(args.side_time)
        side_offset = side_pos - overhead_pos
        note = (
            f"Manually verified by user: overhead {overhead_pos:.2f}s matches "
            f"front {front_pos:.2f}s AND side {side_pos:.2f}s directly."
        )
    else:
        sync_info_path = args.trial / "gopro_footage" / "sync_info.json"
        if not sync_info_path.is_file():
            print(f"ERROR: {sync_info_path} not found -- pass --side-time to set side directly", file=sys.stderr)
            return 2
        sync_info = json.loads(sync_info_path.read_text(encoding="utf-8"))
        ref = sync_info.get("reference_video", "")
        tgt = sync_info.get("target_video", "")
        offset_seconds = sync_info.get("offset_seconds")
        if offset_seconds is None or "front" not in ref.lower() or "side" not in tgt.lower():
            print(f"ERROR: unexpected sync_info.json shape (reference={ref!r} target={tgt!r})", file=sys.stderr)
            return 2
        # tgt(side)_start = ref(front)_start + offset_seconds  =>
        # side_offset = overhead_start - side_start = front_offset - offset_seconds
        side_offset = front_offset - offset_seconds
        note = (
            f"Manually verified by user: overhead {overhead_pos:.2f}s matches "
            f"front {front_pos:.2f}s. side derived via this trial's own "
            f"gopro_footage/sync_info.json (front/side gap {offset_seconds:.4f}s)."
        )

    print(f"front_offset_sec: {front_offset:.3f}")
    print(f"side_offset_sec:  {side_offset:.3f}")

    if args.save:
        path = save_sync_offsets(args.trial, front_offset, side_offset, note=note)
        print(f"Saved: {path}")
    return 0


def verify_offsets_command(args: argparse.Namespace) -> int:
    """Build a row-by-row comparison grid (overhead | front | side) at the
    predicted-matching timestamps for a candidate offset, so a person can
    visually confirm whether the same real-world moment appears in each
    column throughout -- the practical check for whether creation_time-based
    offsets (of unverified absolute accuracy) are actually correct."""
    assets = discover_trial(args.trial, args.calibration_dir)
    if not assets.front_video or not assets.side_video:
        print("ERROR: front and side GoPro videos are required", file=sys.stderr)
        return 2
    overhead_video = str(args.trial / "performance" / next(
        (p.name for p in (args.trial / "performance").glob("*overhead*") if p.is_file()), ""
    )) if (args.trial / "performance").is_dir() else None
    if not overhead_video or not Path(overhead_video).is_file():
        print("ERROR: could not find the overhead camera video under performance/", file=sys.stderr)
        return 2

    try:
        timing = build_timing_report(assets)
    except Exception as exc:
        print(f"ERROR: could not compute timing report: {exc}", file=sys.stderr)
        return 2

    if args.front_offset_sec is not None and args.side_offset_sec is not None:
        front_offset, side_offset = args.front_offset_sec, args.side_offset_sec
    else:
        result = compute_gopro_offsets(assets, timing)
        if result is None:
            print("ERROR: could not auto-compute offsets; pass --front-offset-sec/--side-offset-sec", file=sys.stderr)
            return 2
        front_offset, side_offset, _ = result
    print(f"Using front_offset_sec={front_offset:.2f} side_offset_sec={side_offset:.2f}")

    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        print("ERROR: FFmpeg is required", file=sys.stderr)
        return 1

    front_duration = (media_info(assets.front_video) or {}).get("duration_sec")
    side_duration = (media_info(assets.side_video) or {}).get("duration_sec")
    overhead_duration = (media_info(overhead_video) or {}).get("duration_sec")

    output_dir = args.output_dir or args.trial / "MVP" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "_offset_verify_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    thumb_w = 320
    rows = []
    t = -args.span_before
    while t <= args.span_after + 1e-9:
        front_pos = t
        overhead_pos = t - front_offset
        side_pos = t - front_offset + side_offset
        row_cells = []
        for label, video, pos, duration in (
            ("overhead", overhead_video, overhead_pos, overhead_duration),
            ("front", assets.front_video, front_pos, front_duration),
            ("side", assets.side_video, side_pos, side_duration),
        ):
            valid = duration is None or (0 <= pos <= duration)
            cell_path = frames_dir / f"{label}_{t:+.1f}.png"
            if valid and video and _extract_thumbnail(ffmpeg, str(video), pos, cell_path, width=thumb_w):
                img = Image.open(cell_path).convert("RGB")
            else:
                img = Image.new("RGB", (thumb_w, int(thumb_w * 9 / 16)), (40, 20, 20))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
            draw.text((4, 2), f"{label} {pos:.1f}s", font=font, fill=(255, 255, 0))
            row_cells.append(img)
        row_height = max(cell.height for cell in row_cells)
        row_img = Image.new("RGB", (thumb_w * 3 + 8, row_height), (10, 10, 10))
        for i, cell in enumerate(row_cells):
            row_img.paste(cell, (i * (thumb_w + 4), 0))
        rows.append(row_img)
        t += args.interval

    grid_height = sum(r.height for r in rows) + 4 * (len(rows) - 1)
    grid = Image.new("RGB", (thumb_w * 3 + 8, grid_height), (10, 10, 10))
    y = 0
    for r in rows:
        grid.paste(r, (0, y))
        y += r.height + 4

    grid_path = output_dir / "offset_verify_grid.png"
    grid.save(grid_path)
    print(f"Comparison grid ({len(rows)} rows): {grid_path}")
    print("Each row should show the SAME real-world moment in all three columns")
    print("if the offset is correct; consistent drift across rows means the")
    print("offset needs adjusting by that amount.")
    return 0


def discover_trial(trial_dir: Path, calibration_dir: Path | None = None) -> TrialAssets:
    trial_dir = trial_dir.resolve()
    all_files = [path for path in trial_dir.rglob("*") if path.is_file()]
    gopro_dir = trial_dir / "gopro_footage"
    gopro_files = sorted(
        path for path in gopro_dir.glob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ) if gopro_dir.is_dir() else []

    front = find_gopro_video(gopro_files, "front")
    side = find_gopro_video(gopro_files, "side")
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
    heart_rate_csvs = find_heart_rate_csvs(all_files)
    heart_rate = heart_rate_csvs[0] if heart_rate_csvs else None
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
        heart_rate_csvs=[str(path) for path in heart_rate_csvs],
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
        "note": (
            "source_start_sec/source_end_sec are positions in the OVERHEAD "
            "camera video (what video_position_sec in review_timestamps.json "
            "actually references) -- not front/side GoPro positions. Convert "
            "via a trial's sync_offsets.json (front/side offset relative to "
            "overhead_recorder_start) to get the corresponding front/side "
            "position; see compute_review_window()."
        ),
    }


def overhead_position_at(t: float, segments: list[dict[str, Any]]) -> float:
    """Overhead-camera position at review/output time t, per the pause
    timeline (constant during a 'hold' segment, linear during 'play')."""
    for seg in segments:
        if seg["review_start_sec"] <= t <= seg["review_end_sec"]:
            if seg["type"] == "hold":
                return seg["source_start_sec"]
            span = seg["review_end_sec"] - seg["review_start_sec"]
            if span <= 0:
                return seg["source_start_sec"]
            frac = (t - seg["review_start_sec"]) / span
            return seg["source_start_sec"] + frac * (seg["source_end_sec"] - seg["source_start_sec"])
    # past the last segment
    last = segments[-1]
    return last["source_end_sec"]


def compute_review_window(
    pause_timeline: dict[str, Any],
    front_offset: float,
    side_offset: float,
    front_duration: float,
    side_duration: float,
) -> tuple[float, float]:
    """Return (t_start, t_end) in output/review time: the sub-range where
    BOTH front and side have a valid recorded frame for whatever overhead
    position the pause timeline calls for at that instant. This is the
    "intersection" MVP_SPEC.md describes -- the render must not start before
    front/side have footage, or run past when either one runs out.

    Overhead position is non-decreasing in t (flat during a hold, 1:1 during
    play), so the valid t range is contiguous and can be found by walking
    the segments once from each end.
    """
    segments = pause_timeline["segments"]
    review_duration = pause_timeline["review_duration_sec"]

    # overhead-position range where BOTH cameras are valid
    lo = max(-front_offset, -side_offset)
    hi = min(front_duration - front_offset, side_duration - side_offset)
    if lo > hi:
        raise ValueError(
            f"front and side never overlap in valid recorded range "
            f"(need overhead position in [{lo:.1f}, {hi:.1f}])"
        )

    def t_entering(target_pos: float) -> float:
        """First t (ascending) at which overhead_position(t) >= target_pos."""
        for seg in segments:
            if seg["type"] == "hold":
                if seg["source_start_sec"] >= target_pos:
                    return seg["review_start_sec"]
                continue
            if seg["source_end_sec"] < target_pos:
                continue
            if seg["source_start_sec"] >= target_pos:
                return seg["review_start_sec"]
            span = seg["source_end_sec"] - seg["source_start_sec"]
            frac = (target_pos - seg["source_start_sec"]) / span if span > 0 else 0.0
            return seg["review_start_sec"] + frac * (seg["review_end_sec"] - seg["review_start_sec"])
        return review_duration

    def t_exiting(target_pos: float) -> float:
        """Last t (descending) at which overhead_position(t) <= target_pos."""
        for seg in reversed(segments):
            if seg["type"] == "hold":
                if seg["source_start_sec"] <= target_pos:
                    return seg["review_end_sec"]
                continue
            if seg["source_start_sec"] > target_pos:
                continue
            if seg["source_end_sec"] <= target_pos:
                return seg["review_end_sec"]
            span = seg["source_end_sec"] - seg["source_start_sec"]
            frac = (target_pos - seg["source_start_sec"]) / span if span > 0 else 0.0
            return seg["review_start_sec"] + frac * (seg["review_end_sec"] - seg["review_start_sec"])
        return 0.0

    t_start = max(0.0, t_entering(lo))
    t_end = min(review_duration, t_exiting(hi))
    return t_start, t_end


def _extract_frame_fullres(ffmpeg: str, video_path: str, position_sec: float, out_path: Path) -> bool:
    command = [
        ffmpeg, "-y", "-v", "error",
        "-ss", f"{max(position_sec, 0):.3f}", "-i", video_path,
        "-frames:v", "1", "-update", "1",
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and out_path.is_file()


def _probe_audio_params(source: Path) -> tuple[int, str]:
    """(sample_rate, channel_layout) of source's first audio stream, falling
    back to 48kHz stereo (GoPro's usual format) if ffprobe/the stream isn't
    available -- used so freeze-segment silence exactly matches play-segment
    audio, letting the final concat use `-c copy` without a parameter clash."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ffmpeg_path = _resolve_ffmpeg()
        candidate = Path(ffmpeg_path).with_name("ffprobe.exe") if ffmpeg_path else None
        ffprobe = str(candidate) if candidate and candidate.is_file() else None
    if not ffprobe:
        return 48000, "stereo"
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "csv=p=0", str(source)],
        capture_output=True, text=True,
    )
    try:
        rate_str, channels_str = result.stdout.strip().split(",")
        return int(rate_str), ("mono" if int(channels_str) == 1 else "stereo")
    except Exception:
        return 48000, "stereo"


def build_pause_aware_video(
    ffmpeg: str,
    source: Path,
    offset: float,
    pause_timeline: dict[str, Any],
    t_start: float,
    t_end: float,
    output: Path,
    fps: int = 30,
) -> Path:
    """Build a pause-aware version of a front/side video covering review-time
    window [t_start, t_end): plays back normally during 'play' segments
    (mapped into source-video time via `offset`) and freezes on the paused
    frame during 'hold' segments. Output starts at 0 and runs for exactly
    (t_end - t_start) seconds so it stays frame-accurate against the
    continuous face/narration/marker tracks the final render composites
    alongside it -- each clip is retimed (setpts) to its exact target
    duration rather than trusting the source trim to land exactly on it,
    since review-time spans (from wall-clock pause events) and source-time
    spans (from a human-entered video_position_sec) can drift by a frame
    or two even though both nominally advance 1:1.
    """
    segments = pause_timeline["segments"]
    audio_sample_rate, audio_channel_layout = _probe_audio_params(source)
    work_dir = output.parent / f"_clips_{output.stem}"
    if work_dir.is_dir():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []
    clip_index = 0
    for seg in segments:
        clip_review_start = max(seg["review_start_sec"], t_start)
        clip_review_end = min(seg["review_end_sec"], t_end)
        if clip_review_end <= clip_review_start:
            continue
        target_duration = clip_review_end - clip_review_start
        clip_path = work_dir / f"clip_{clip_index:04d}.mp4"

        if seg["type"] == "hold":
            # Silence during a freeze, not looped/duplicated audio -- the
            # performance is paused, so there's no new ambient sound to show;
            # sample rate/channels/codec match the source exactly so the
            # final `-c copy` concat across mixed play/hold clips doesn't
            # choke on mismatched audio stream parameters.
            video_pos = seg["source_start_sec"] + offset
            frame_path = work_dir / f"clip_{clip_index:04d}_frame.png"
            if not _extract_frame_fullres(ffmpeg, str(source), video_pos, frame_path):
                raise RuntimeError(f"failed to extract hold frame at {video_pos:.2f}s from {source}")
            command = [
                ffmpeg, "-y", "-v", "error",
                "-loop", "1", "-i", str(frame_path),
                "-f", "lavfi", "-i", f"anullsrc=channel_layout={audio_channel_layout}:sample_rate={audio_sample_rate}",
                "-t", f"{target_duration:.3f}",
                "-r", str(fps),
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-ar", str(audio_sample_rate),
                str(clip_path),
            ]
        else:
            seg_review_span = seg["review_end_sec"] - seg["review_start_sec"]
            seg_source_span = seg["source_end_sec"] - seg["source_start_sec"]
            frac_start = (clip_review_start - seg["review_start_sec"]) / seg_review_span if seg_review_span > 0 else 0.0
            frac_end = (clip_review_end - seg["review_start_sec"]) / seg_review_span if seg_review_span > 0 else 1.0
            video_start = seg["source_start_sec"] + frac_start * seg_source_span + offset
            video_end = seg["source_start_sec"] + frac_end * seg_source_span + offset
            video_start = max(video_start, 0.0)
            actual_span = max(video_end - video_start, 1e-3)
            speed = actual_span / target_duration if target_duration > 0 else 1.0
            # atempo mirrors the video's setpts correction so audio/video
            # land on the identical target_duration -- speed corrections here
            # are tiny (source and review time both nominally advance 1:1),
            # well inside atempo's valid [0.5, 2.0] range.
            command = [
                ffmpeg, "-y", "-v", "error",
                "-ss", f"{video_start:.3f}", "-t", f"{actual_span:.3f}", "-i", str(source),
                "-vf", f"setpts=PTS/{speed:.6f},fps={fps}",
                "-af", f"atempo={speed:.6f}",
                "-t", f"{target_duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-ar", str(audio_sample_rate),
                str(clip_path),
            ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"failed to build clip {clip_index} ({seg['type']}) for {output}: {result.stderr[-2000:]}")
        clip_paths.append(clip_path)
        clip_index += 1

    if not clip_paths:
        raise ValueError(f"no segments overlap review window [{t_start}, {t_end}] for {output}")

    concat_list = work_dir / "concat_list.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in clip_paths) + "\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"concat failed for {output}: {result.stderr[-4000:]}")

    shutil.rmtree(work_dir, ignore_errors=True)
    return output


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


def read_heart_rate(paths: str | list[str]) -> list[dict[str, float]]:
    if isinstance(paths, str):
        paths = [paths]
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append({"timestamp": float(row["timestamp"]), "bpm": float(row["bpm"])})
                except (KeyError, TypeError, ValueError):
                    continue
    # merging multiple files (chapter-split parts, or a '_retry' alongside
    # the original) means rows aren't necessarily in time order on input
    rows.sort(key=lambda r: r["timestamp"])
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

    rows = read_heart_rate(assets.heart_rate_csvs)
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

    series_color = LAYOUT_CHART_COLOR
    text_secondary = "#52514e"
    grid_color = "#e3e2dd"

    # Render at the exact pixel size of the chart's card interior so it fills
    # the card with no letterboxing, then composite onto a rounded light card
    # (transparent outside the rounded rect) to match the approved layout.
    # No marker is drawn here: render commands draw it dynamically (driven by
    # playback time) using chart_plot_pixel_bounds(), so it actually moves
    # instead of sitting frozen at whichever position this command ran at.
    card_pad = LAYOUT_CHART_CARD_PAD
    inner_w = LAYOUT_CHART_W - card_pad * 2
    inner_h = LAYOUT_FACE_H - card_pad * 2
    dpi = 150

    figure, axis = plt.subplots(figsize=(inner_w / dpi, inner_h / dpi), dpi=dpi)
    figure.patch.set_facecolor(LAYOUT_CARD_LIGHT_HEX)
    axis.set_facecolor(LAYOUT_CARD_LIGHT_HEX)
    axis.plot(x_values, y_values, color=series_color, linewidth=2.0, solid_joinstyle="round", solid_capstyle="round")
    axis.fill_between(x_values, y_values, chart_data["min_bpm"] - 5, color=series_color, alpha=0.10, linewidth=0)

    axis.set_xlim(0, review_end - review_start)
    axis.set_ylim(max(0, chart_data["min_bpm"] - 5), chart_data["max_bpm"] + 5)
    axis.set_xlabel("Review time (seconds)", color=text_secondary, fontsize=11)
    axis.set_ylabel("BPM", color=text_secondary, fontsize=11)
    axis.tick_params(colors=text_secondary, labelsize=10)
    axis.grid(True, color=grid_color, linewidth=1.0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(grid_color)
    figure.subplots_adjust(
        left=CHART_PLOT_LEFT_FRAC, right=CHART_PLOT_RIGHT_FRAC,
        top=CHART_PLOT_TOP_FRAC, bottom=CHART_PLOT_BOTTOM_FRAC,
    )

    raw_chart_path = output_dir / "_chart_raw.png"
    figure.savefig(raw_chart_path, facecolor=figure.get_facecolor(), dpi=dpi)
    plt.close(figure)

    from PIL import Image

    chart_img = Image.open(raw_chart_path).convert("RGB")
    if chart_img.size != (inner_w, inner_h):
        chart_img = chart_img.resize((inner_w, inner_h), Image.LANCZOS)

    card_bg = Image.new("RGB", (LAYOUT_CHART_W, LAYOUT_FACE_H), LAYOUT_CARD_LIGHT)
    card_bg.paste(chart_img, (card_pad, card_pad))
    card = Image.new("RGBA", (LAYOUT_CHART_W, LAYOUT_FACE_H), (0, 0, 0, 0))
    card.paste(card_bg, (0, 0), _rounded_mask((LAYOUT_CHART_W, LAYOUT_FACE_H), LAYOUT_RADIUS))

    image_path = output_dir / "heart_rate_chart_preview.png"
    card.save(image_path)
    raw_chart_path.unlink(missing_ok=True)
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


def compose_review_video(
    ffmpeg: str,
    assets: "TrialAssets",
    front: Path,
    side: Path,
    chart: Path,
    review_start_sec: float,
    duration: float,
    front_start_sec: float,
    side_start_sec: float,
    output: Path,
    subtitles: Path | None,
    review_duration_sec: float,
) -> int:
    """Composite front/side/face/chart/subtitles into the approved MVP layout
    via one ffmpeg filter_complex. Shared by render-preview (a short clip at
    a fixed front/side offset, for spot-checking layout/marker/subtitles
    without building pause-aware video first) and the full render (front/side
    are pre-built pause-aware clips that already start at review time
    t_start, so front_start_sec/side_start_sec are 0 and review_start_sec is
    that window's t_start -- everything else about the composite is
    identical between the two)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = chart.parent
    layout = generate_layout_assets(assets_dir)

    front_x, front_y = LAYOUT_MARGIN, LAYOUT_MARGIN
    side_x, side_y = LAYOUT_MARGIN + LAYOUT_VIDEO_W + LAYOUT_GAP, LAYOUT_MARGIN
    sub_x = LAYOUT_MARGIN
    sub_y = LAYOUT_MARGIN + LAYOUT_VIDEO_H + LAYOUT_MARGIN
    bottom_y = sub_y + LAYOUT_SUB_H + LAYOUT_MARGIN
    face_x, face_y = LAYOUT_MARGIN, bottom_y
    chart_x = LAYOUT_MARGIN + LAYOUT_FACE_W + LAYOUT_GAP
    shadow_pad = 12

    filter_parts = [
        f"color=c=0x{LAYOUT_BG_HEX.lstrip('#')}:s={LAYOUT_CANVAS_W}x{LAYOUT_CANVAS_H}[bg0]",
        f"[bg0][6:v]overlay={front_x - shadow_pad}:{front_y - shadow_pad}[bg1]",
        f"[bg1][6:v]overlay={side_x - shadow_pad}:{front_y - shadow_pad}[bg2]",
        f"[0:v]scale={LAYOUT_VIDEO_W}:{LAYOUT_VIDEO_H},fps=30,setpts=PTS-STARTPTS,format=rgba[frontv]",
        "[4:v]format=gray[maskv_f]",
        "[frontv][maskv_f]alphamerge[frontm]",
        f"[bg2][frontm]overlay={front_x}:{front_y}[bg3]",
        f"[1:v]scale={LAYOUT_VIDEO_W}:{LAYOUT_VIDEO_H},fps=30,setpts=PTS-STARTPTS,format=rgba[sidev]",
        "[4:v]format=gray[maskv_s]",
        "[sidev][maskv_s]alphamerge[sidem]",
        f"[bg3][sidem]overlay={side_x}:{side_y}[bg4]",
        f"[bg4][8:v]overlay={sub_x}:{sub_y}[bg5]",
        f"[bg5][7:v]overlay={face_x - shadow_pad}:{face_y - shadow_pad}[bg6]",
        f"[2:v]scale={LAYOUT_FACE_W}:{LAYOUT_FACE_H},fps=30,setpts=PTS-STARTPTS,format=rgba[facev]",
        "[5:v]format=gray[maskv_face]",
        "[facev][maskv_face]alphamerge[facem]",
        f"[bg6][facem]overlay={face_x}:{face_y}[bg7]",
        f"[bg7][3:v]overlay={chart_x}:{bottom_y}[bg8]",
        "[0:a]volume=0.125[performance_audio]",
        "[11:a]volume=1.0[narration_audio]",
        "[performance_audio][narration_audio]amix=inputs=2:duration=longest[audio]",
    ]

    # Animated heart-rate marker: composited here via `overlay` (not
    # `drawbox` -- its x/y are evaluated once at filter init in this ffmpeg
    # build, before `t` exists, so a "moving" drawbox silently renders at a
    # nonsensical fixed position instead of animating; overlay's x/y support
    # per-frame `t` expressions properly, confirmed empirically).
    # review time = review_start_sec + t, since review_start_sec is how far
    # into the review timeline this render's t=0 begins.
    plot_left, plot_right, plot_top, plot_bottom = chart_plot_pixel_bounds(chart_x, bottom_y)
    plot_width = plot_right - plot_left
    marker_x_expr = f"{plot_left:.2f}+(({review_start_sec}+t)/{review_duration_sec:.4f})*{plot_width:.2f}"
    dot_size = 12
    dot_y = (plot_top + plot_bottom) / 2 - dot_size / 2
    filter_parts.append(f"[bg8][9:v]overlay=x='{marker_x_expr}-1':y={plot_top:.2f}[bg9]")
    filter_parts.append(f"[bg9][10:v]overlay=x='{marker_x_expr}-{dot_size / 2:.1f}':y={dot_y:.2f}[bg10]")
    marker_label = "bg10"

    if subtitles:
        # Burn subtitles via chained drawtext (one per cue), positioned inside
        # the subtitle bar we composited above. Not using ffmpeg's `subtitles=`
        # filter: it needs libass, which no available ffmpeg build here has,
        # and even a working one would place text by its own default styling
        # rather than fitting our specific bar.
        #
        # Text goes through `textfile=`, not an inline `text='...'` value:
        # ffmpeg's filtergraph quoting for an embedded literal single-quote
        # (close-quote, backslash-quote, reopen-quote) did not survive real
        # narration text containing apostrophes -- those cues silently
        # rendered blank. A file path only needs colon-escaping, which is
        # simple and already used elsewhere for Windows drive letters.
        font_path_escaped = SUBTITLE_FONT_PATH.replace(":", "\\:")
        bar_width = LAYOUT_CANVAS_W - LAYOUT_MARGIN * 2
        max_text_width = bar_width - 80
        subtitle_cues_dir = assets_dir / "_subtitle_cues"
        subtitle_cues_dir.mkdir(parents=True, exist_ok=True)
        current_label = marker_label
        cue_index = 0
        for cue in parse_srt(subtitles):
            start = cue["start"] - review_start_sec
            end = cue["end"] - review_start_sec
            if end <= 0 or start >= duration:
                continue
            start = max(start, 0.0)
            cue_text = cue["text"].replace("\n", " ").strip()
            if not cue_text:
                continue
            fitted_text, font_size = _fit_subtitle_line(cue_text, max_text_width, SUBTITLE_FONT_PATH)
            cue_file = subtitle_cues_dir / f"cue_{cue_index}.txt"
            cue_file.write_text(fitted_text, encoding="utf-8")
            cue_file_escaped = str(cue_file).replace("\\", "/").replace(":", "\\:")
            next_label = f"sub{cue_index}"
            filter_parts.append(
                f"[{current_label}]drawtext=fontfile='{font_path_escaped}':textfile='{cue_file_escaped}':"
                f"fontcolor=white:fontsize={font_size}:x=(w-text_w)/2:"
                f"y={sub_y}+(({LAYOUT_SUB_H}-text_h)/2):"
                f"enable='between(t,{start:.3f},{end:.3f})'[{next_label}]"
            )
            current_label = next_label
            cue_index += 1
        video_label = f"[{current_label}]"
    else:
        video_label = f"[{marker_label}]"

    command = [
        ffmpeg,
        "-y",
        "-ss", str(front_start_sec), "-i", str(front),
        "-ss", str(side_start_sec), "-i", str(side),
        "-stream_loop", "-1", "-ss", str(review_start_sec), "-i", assets.face_video,
        # -framerate 30 on every looped PNG: image2/png_pipe defaults to 25fps
        # when unspecified, so these were being fed into alphamerge/overlay
        # alongside 30fps video -- that mismatch produced a reproducible
        # single-frame content jump partway through a long render (confirmed
        # independent of encoder preset, and absent when the front/side
        # stream was tested in isolation without this chain), consistent
        # with a periodic 25-vs-30fps resync beat rather than a logic bug.
        "-framerate", "30", "-loop", "1", "-i", str(chart),
        "-framerate", "30", "-loop", "1", "-i", str(layout["mask_video"]),
        "-framerate", "30", "-loop", "1", "-i", str(layout["mask_face"]),
        "-framerate", "30", "-loop", "1", "-i", str(layout["shadow_video"]),
        "-framerate", "30", "-loop", "1", "-i", str(layout["shadow_face"]),
        "-framerate", "30", "-loop", "1", "-i", str(layout["subtitle_bg"]),
        "-framerate", "30", "-loop", "1", "-i", str(layout["marker_line"]),
        "-framerate", "30", "-loop", "1", "-i", str(layout["marker_dot"]),
        "-ss", str(review_start_sec), "-i", assets.narration_audio,
        "-filter_complex", ";".join(filter_parts),
        "-map", video_label,
        "-map", "[audio]",
        "-t", str(duration),
        # "medium" (x264 default), not "ultrafast", plus a huge -g: a
        # full-length render showed a reproducible single-frame content jump
        # exactly at frame ~12500 -- 50x libx264's default 250-frame GOP
        # length -- regardless of preset or the PNG-framerate fix above, and
        # the pause-aware source was independently proven pixel-identical
        # throughout at that exact point. That signature (exact GOP-length
        # multiple, preset-independent) points to an I-frame requantization
        # quirk on 100%-static content at a periodic keyframe boundary, not
        # a logic bug -- so -g is set past this video's total frame count,
        # forcing a single keyframe and removing the periodic boundary
        # entirely. Batch renders run unattended overnight, so giving up
        # mid-file seek efficiency for this is a non-issue.
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-r", "30", "-g", "999999",
        "-c:a", "aac", "-shortest",
        str(output),
    ]
    print(f"Rendering: {output}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        return result.returncode
    print(f"Created: {output}")
    return 0


def _resolve_ffmpeg_or_error() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


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
    ffmpeg = _resolve_ffmpeg_or_error()
    if not ffmpeg:
        print("ERROR: FFmpeg is required for rendering", file=sys.stderr)
        return 1

    try:
        timing_report = build_timing_report(assets)
    except Exception as exc:
        print(f"ERROR: could not compute timing report: {exc}", file=sys.stderr)
        return 2
    review_duration_sec = timing_report.get("review_wall_duration_sec")
    if not review_duration_sec:
        print("ERROR: review wall-clock duration is unavailable; cannot position the heart-rate marker", file=sys.stderr)
        return 2

    return compose_review_video(
        ffmpeg=ffmpeg,
        assets=assets,
        front=front,
        side=side,
        chart=chart,
        review_start_sec=args.review_start_sec,
        duration=args.duration,
        front_start_sec=args.front_start_sec,
        side_start_sec=args.side_start_sec,
        output=output,
        subtitles=args.subtitles,
        review_duration_sec=review_duration_sec,
    )


def ensure_prepared_assets(trial: Path, assets: "TrialAssets", calibration_dir: Path | None) -> Path:
    """Return prepared_assets.json's path, writing it first if missing. Most
    trials already have <PID>_front_undistorted.mp4 / <PID>_side_undistorted.mp4
    in MVP/generated from an earlier bulk undistortion pass that didn't go
    through `prepare` (so it never wrote this manifest) -- if those files are
    there, point the manifest at them instead of re-undistorting from scratch.
    Only actually undistorts when neither the manifest nor those files exist."""
    prepared_path = trial / "MVP" / "generated" / "prepared_assets.json"
    if prepared_path.is_file():
        return prepared_path

    output_dir = trial / "MVP" / "generated"
    pid = trial.name.split("_")[0]
    existing_front = output_dir / f"{pid}_front_undistorted.mp4"
    existing_side = output_dir / f"{pid}_side_undistorted.mp4"
    if existing_front.is_file() and existing_side.is_file():
        outputs = {
            "front": {"source": assets.front_video, "output": str(existing_front)},
            "side": {"source": assets.side_video, "output": str(existing_side)},
        }
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        return prepared_path

    if not assets.calibration_front or not assets.calibration_side:
        raise RuntimeError(
            f"no prepared_assets.json, no existing undistorted files, and no calibration for {trial.name}"
        )
    from undistort import undistort_video
    outputs = {}
    for camera, source, calibration in (
        ("front", assets.front_video, assets.calibration_front),
        ("side", assets.side_video, assets.calibration_side),
    ):
        output = output_dir / f"{pid}_{camera}_undistorted.mp4"
        frames = undistort_video(Path(source), output, Path(calibration), alpha=1.0)
        outputs[camera] = {"source": source, "output": str(output), "frames": frames}
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return prepared_path


def ensure_chart(trial: Path, assets: "TrialAssets") -> Path:
    """Return heart_rate_chart_preview.png's path, generating it first if missing."""
    chart_path = trial / "MVP" / "generated" / "heart_rate_chart_preview.png"
    if chart_path.is_file():
        return chart_path
    rc = chart_command(argparse.Namespace(trial=trial, calibration_dir=None, output_dir=None))
    if rc != 0 or not chart_path.is_file():
        raise RuntimeError(f"chart generation failed for {trial.name} (rc={rc})")
    return chart_path


def render_one_trial(trial: Path) -> Path:
    """Full per-trial pipeline: ensure prepared assets + chart exist, then
    render. Shared by render_command (single trial, real argparse args) and
    batch_render_command (all trials, error-isolated)."""
    assets = discover_trial(trial)
    if not assets.face_video or not assets.narration_audio:
        raise RuntimeError("face video and narration audio are required")

    ensure_prepared_assets(trial, assets, None)
    ensure_chart(trial, assets)

    prepared = json.loads((trial / "MVP" / "generated" / "prepared_assets.json").read_text(encoding="utf-8"))
    front_undistorted = Path(prepared["front"]["output"])
    side_undistorted = Path(prepared["side"]["output"])
    if not front_undistorted.is_file() or not side_undistorted.is_file():
        raise RuntimeError("prepared front/side video is missing")

    offsets = load_sync_offsets(trial)
    if not offsets:
        raise RuntimeError(f"no sync offsets found ({sync_offsets_path(trial)})")
    front_offset = offsets["front_offset_sec"]
    side_offset = offsets["side_offset_sec"]

    chart = trial / "MVP" / "generated" / "heart_rate_chart_preview.png"
    subtitles = Path(assets.subtitle_candidates[0]) if assets.subtitle_candidates else None

    pause_timeline = build_pause_timeline(assets)
    front_duration = (media_info(str(front_undistorted)) or {}).get("duration_sec")
    side_duration = (media_info(str(side_undistorted)) or {}).get("duration_sec")
    if not front_duration or not side_duration:
        raise RuntimeError("could not read front/side undistorted video duration")

    t_start, t_end = compute_review_window(pause_timeline, front_offset, side_offset, front_duration, side_duration)
    duration = t_end - t_start

    ffmpeg = _resolve_ffmpeg_or_error()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for rendering")

    output_dir = trial / "MVP" / "generated"
    front_pa = output_dir / "front_pause_aware.mp4"
    side_pa = output_dir / "side_pause_aware.mp4"
    build_pause_aware_video(ffmpeg, front_undistorted, front_offset, pause_timeline, t_start, t_end, front_pa)
    build_pause_aware_video(ffmpeg, side_undistorted, side_offset, pause_timeline, t_start, t_end, side_pa)

    output = trial / "MVP" / "review.mp4"
    rc = compose_review_video(
        ffmpeg=ffmpeg,
        assets=assets,
        front=front_pa,
        side=side_pa,
        chart=chart,
        review_start_sec=t_start,
        duration=duration,
        front_start_sec=0.0,
        side_start_sec=0.0,
        output=output,
        subtitles=subtitles,
        review_duration_sec=pause_timeline["review_duration_sec"],
    )
    if rc != 0:
        raise RuntimeError(f"compose_review_video failed (rc={rc})")
    return output


def batch_render_command(args: argparse.Namespace) -> int:
    data_root = args.data_root
    trials = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("P"))
    if args.start:
        trials = [t for t in trials if t.name >= args.start]
    if args.skip_existing:
        trials = [t for t in trials if not (t / "MVP" / "review.mp4").is_file()]

    print(f"Batch rendering {len(trials)} trial(s).")
    results = {"ok": [], "failed": []}
    for index, trial in enumerate(trials, 1):
        print(f"\n[{index}/{len(trials)}] {trial.name}")
        try:
            output = render_one_trial(trial)
            print(f"  OK: {output}")
            results["ok"].append(trial.name)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            results["failed"].append({"trial": trial.name, "error": str(exc)})

    print(f"\nDone. {len(results['ok'])}/{len(trials)} succeeded.")
    if results["failed"]:
        print("Failed trials:")
        for failure in results["failed"]:
            print(f"  {failure['trial']}: {failure['error']}")
    report_path = data_root.parent / "batch_render_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0 if not results["failed"] else 1


def render_command(args: argparse.Namespace) -> int:
    assets = discover_trial(args.trial, args.calibration_dir)
    if not assets.face_video or not assets.narration_audio:
        print("ERROR: face video and narration audio are required", file=sys.stderr)
        return 2

    prepared_path = args.prepared
    if not prepared_path:
        try:
            prepared_path = ensure_prepared_assets(args.trial, assets, args.calibration_dir)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    if not prepared_path.is_file():
        print(f"ERROR: prepared asset report not found: {prepared_path}", file=sys.stderr)
        print("Run the prepare command after calibration first.", file=sys.stderr)
        return 2
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    front_undistorted = Path(prepared["front"]["output"])
    side_undistorted = Path(prepared["side"]["output"])
    if not front_undistorted.is_file() or not side_undistorted.is_file():
        print("ERROR: prepared front/side video is missing", file=sys.stderr)
        return 2

    offsets = load_sync_offsets(args.trial)
    if not offsets:
        print(f"ERROR: no sync offsets found ({sync_offsets_path(args.trial)}); run record-match first", file=sys.stderr)
        return 2
    front_offset = offsets["front_offset_sec"]
    side_offset = offsets["side_offset_sec"]

    chart = args.chart
    if not chart:
        try:
            chart = ensure_chart(args.trial, assets)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    if not chart.is_file():
        print(f"ERROR: chart preview not found: {chart}; run the chart command first", file=sys.stderr)
        return 2

    subtitles = args.subtitles
    if not subtitles and assets.subtitle_candidates:
        subtitles = Path(assets.subtitle_candidates[0])

    try:
        pause_timeline = build_pause_timeline(assets)
    except Exception as exc:
        print(f"ERROR: could not build pause timeline: {exc}", file=sys.stderr)
        return 2

    front_duration = (media_info(str(front_undistorted)) or {}).get("duration_sec")
    side_duration = (media_info(str(side_undistorted)) or {}).get("duration_sec")
    if not front_duration or not side_duration:
        print("ERROR: could not read front/side undistorted video duration", file=sys.stderr)
        return 2

    try:
        t_start, t_end = compute_review_window(pause_timeline, front_offset, side_offset, front_duration, side_duration)
    except Exception as exc:
        print(f"ERROR: could not compute review window: {exc}", file=sys.stderr)
        return 2
    duration = t_end - t_start
    print(f"Review window: [{t_start:.2f}, {t_end:.2f}] ({duration:.2f}s of {pause_timeline['review_duration_sec']:.2f}s total)")

    ffmpeg = _resolve_ffmpeg_or_error()
    if not ffmpeg:
        print("ERROR: FFmpeg is required for rendering", file=sys.stderr)
        return 1

    output_dir = args.trial / "MVP" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    front_pa = output_dir / "front_pause_aware.mp4"
    side_pa = output_dir / "side_pause_aware.mp4"
    print(f"Building pause-aware front video: {front_pa}")
    build_pause_aware_video(ffmpeg, front_undistorted, front_offset, pause_timeline, t_start, t_end, front_pa)
    print(f"Building pause-aware side video: {side_pa}")
    build_pause_aware_video(ffmpeg, side_undistorted, side_offset, pause_timeline, t_start, t_end, side_pa)

    output = args.output or args.trial / "MVP" / "review.mp4"
    return compose_review_video(
        ffmpeg=ffmpeg,
        assets=assets,
        front=front_pa,
        side=side_pa,
        chart=chart,
        review_start_sec=t_start,
        duration=duration,
        front_start_sec=0.0,
        side_start_sec=0.0,
        output=output,
        subtitles=subtitles,
        review_duration_sec=pause_timeline["review_duration_sec"],
    )


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
    chart.set_defaults(handler=chart_command)
    suggest_offsets = subparsers.add_parser(
        "suggest-offsets",
        help="estimate front/side GoPro-to-review-timeline offsets from embedded creation_time metadata",
    )
    suggest_offsets.add_argument("--trial", type=Path, required=True)
    suggest_offsets.add_argument("--calibration-dir", type=Path)
    suggest_offsets.add_argument("--save", action="store_true", help="write the suggestion to sync_offsets.json")
    suggest_offsets.set_defaults(handler=suggest_offsets_command)
    verify_offsets = subparsers.add_parser(
        "verify-offsets",
        help="build a visual overhead|front|side comparison grid to spot-check a GoPro sync offset",
    )
    verify_offsets.add_argument("--trial", type=Path, required=True)
    verify_offsets.add_argument("--calibration-dir", type=Path)
    verify_offsets.add_argument("--output-dir", type=Path)
    verify_offsets.add_argument("--front-offset-sec", type=float)
    verify_offsets.add_argument("--side-offset-sec", type=float)
    verify_offsets.add_argument("--span-before", type=float, default=5.0)
    verify_offsets.add_argument("--span-after", type=float, default=20.0)
    verify_offsets.add_argument("--interval", type=float, default=2.0)
    verify_offsets.set_defaults(handler=verify_offsets_command)
    record_match = subparsers.add_parser(
        "record-match",
        help="compute+save front/side offsets from a manually-found overhead<->front(/side) matching timestamp",
    )
    record_match.add_argument("--trial", type=Path, required=True)
    record_match.add_argument("--overhead-time", required=True, help="seconds or MM:SS in the overhead video")
    record_match.add_argument("--front-time", required=True, help="seconds or MM:SS in the front video, same moment")
    record_match.add_argument("--side-time", help="optional: seconds or MM:SS in the side video, same moment")
    record_match.add_argument("--save", action="store_true", default=True, help="write to sync_offsets.json (default on)")
    record_match.add_argument("--no-save", dest="save", action="store_false")
    record_match.set_defaults(handler=record_match_command)
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
    full_render = subparsers.add_parser(
        "render",
        help="full pause-aware render: builds frozen-during-pause front/side video then composites the final review video",
    )
    full_render.add_argument("--trial", type=Path, required=True)
    full_render.add_argument("--calibration-dir", type=Path)
    full_render.add_argument("--prepared", type=Path)
    full_render.add_argument("--chart", type=Path)
    full_render.add_argument("--subtitles", type=Path)
    full_render.add_argument("--output", type=Path)
    full_render.set_defaults(handler=render_command)
    batch = subparsers.add_parser(
        "batch-render",
        help="render every trial under a data root, continuing past per-trial failures",
    )
    batch.add_argument("--data-root", type=Path, required=True)
    batch.add_argument("--start", help="trial folder name prefix to start from, e.g. P10")
    batch.add_argument("--skip-existing", action="store_true", help="skip trials that already have MVP/review.mp4")
    batch.set_defaults(handler=batch_render_command)
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
