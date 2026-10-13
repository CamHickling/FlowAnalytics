#!/usr/bin/env python3
"""Narration-view and performance-view videos for each trial.

narration view    The whole review session. Front, side and overhead cameras
                  play as the participant watched them and freeze while they
                  paused to talk; the narrating face, narration audio (over
                  quiet performance sound), subtitles and review-phase heart
                  rate run continuously.  -> MVP/narration_view.mp4

performance view  The performance phase in real time. Front, side and overhead
                  cameras, the face from when the participant watched each
                  moment during review (talking pauses cut out), GoPro sound,
                  and performance-phase heart rate.  -> MVP/performance_view.mp4

Both share one 1920x1080 layout: three cameras across the top, a subtitle /
caption bar, then face and heart-rate chart. Heart rate uses one fixed axis
(HR_MIN_BPM..HR_MAX_BPM) so videos compare across trials and views.

Usage:
    python views.py narration   --trial <trial dir>
    python views.py performance --trial <trial dir>
    python views.py batch --data-root <dir> [--view both|narration|performance] [--start P10] [--skip-existing]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mvp

CANVAS_W, CANVAS_H = 1920, 1080
CAM_Y, CAM_H, CAM_GAP = 30, 360, 10
FRONT_W = SIDE_W = 640                 # GoPros are 16:9
OVERHEAD_W = 576                       # overhead cam is 16:10 (1600x1000)
CAM_X0 = (CANVAS_W - (FRONT_W + SIDE_W + OVERHEAD_W + 2 * CAM_GAP)) // 2
SIDE_X = CAM_X0 + FRONT_W + CAM_GAP
OVERHEAD_X = SIDE_X + SIDE_W + CAM_GAP
BAR_X, BAR_Y, BAR_H = 20, 410, 60
BAR_W = CANVAS_W - 2 * BAR_X
BOTTOM_Y = 490
FACE_X, FACE_W, FACE_H = 20, 992, 558  # 16:9
CHART_X = FACE_X + FACE_W + 10
CHART_W, CHART_H = CANVAS_W - 20 - CHART_X, FACE_H

HR_MIN_BPM, HR_MAX_BPM = 60, 210       # covers every trial's review and performance readings
HR_GAP_SEC = 3.0                       # readings further apart than this are a sensor dropout
# Plotted area of the chart card, as fractions (matplotlib axes position).
# The marker track uses the same fractions so it lines up with the line.
PLOT_LEFT, PLOT_RIGHT, PLOT_BOTTOM, PLOT_TOP = 0.075, 0.975, 0.13, 0.86

PERFORMANCE_AUDIO_VOLUME = 0.125       # under the narration, as in review.mp4
BOLD_FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
TRANSCRIPTS_DIR = Path(os.environ.get(
    "FLOW_ANALYTICS_TRANSCRIPTS", Path(__file__).resolve().parents[1] / "flow transcripts"
))


@dataclass
class TrialMedia:
    trial: Path
    assets: mvp.TrialAssets
    front: Path
    side: Path
    overhead: Path
    front_offset: float
    side_offset: float
    ffmpeg: str
    work_dir: Path


def _escape_path(path: str | Path) -> str:
    """A path as a quoted ffmpeg filter option value."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def find_overhead(trial: Path) -> Path:
    files = sorted(
        p for p in (trial / "performance").glob("*overhead*")
        if p.is_file() and p.suffix.lower() in mvp.VIDEO_EXTENSIONS
    ) if (trial / "performance").is_dir() else []
    if not files:
        raise RuntimeError("no overhead video under performance/")
    # Split recordings (_1of2, _2of2) are separate takes; the review player
    # showed the last one, which is what the pause positions refer to.
    return files[-1]


def find_subtitles(trial: Path, assets: mvp.TrialAssets) -> Path | None:
    """Timed narration subtitles: an .srt in the trial folder, else the
    cleaned-and-timed synthesis in the transcripts folder."""
    for candidate in assets.subtitle_candidates:
        if candidate.lower().endswith(".srt") and "scoring" not in Path(candidate).name.lower():
            return Path(candidate)
    pid = trial.name.split("_")[0]
    synthesis = TRANSCRIPTS_DIR / "synthesis" / pid / f"{pid}_audio_narration_synthesis.srt"
    return synthesis if synthesis.is_file() else None


def load_trial_media(trial: Path) -> TrialMedia:
    trial = trial.resolve()
    assets = mvp.discover_trial(trial)
    if not assets.face_video:
        raise RuntimeError("face video is required")
    prepared = json.loads(mvp.ensure_prepared_assets(trial, assets, None).read_text(encoding="utf-8"))
    front, side = Path(prepared["front"]["output"]), Path(prepared["side"]["output"])
    if not front.is_file() or not side.is_file():
        raise RuntimeError("prepared front/side video is missing")
    offsets = mvp.load_sync_offsets(trial)
    if not offsets:
        raise RuntimeError(f"no sync offsets found ({mvp.sync_offsets_path(trial)})")
    ffmpeg = mvp._resolve_ffmpeg_or_error()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for rendering")
    work_dir = trial / "MVP" / "generated" / "views"
    work_dir.mkdir(parents=True, exist_ok=True)
    return TrialMedia(
        trial=trial, assets=assets, front=front, side=side, overhead=find_overhead(trial),
        front_offset=offsets["front_offset_sec"], side_offset=offsets["side_offset_sec"],
        ffmpeg=ffmpeg, work_dir=work_dir,
    )


def phase_bounds(manifest: dict, phase_id: str) -> tuple[float, float | None] | None:
    """(start, end) wall time of the last run of a phase; end is the next phase's start."""
    starts = [e for e in manifest.get("events", []) if e.get("event") == "phase_start"]
    for index in range(len(starts) - 1, -1, -1):
        if starts[index].get("phase_id") == phase_id:
            end = starts[index + 1]["wall_time"] if index + 1 < len(starts) else None
            return starts[index]["wall_time"], end
    return None


# --- heart rate --------------------------------------------------------------

def read_hr(assets: mvp.TrialAssets, wall_start: float, wall_end: float) -> list[tuple[float, float]]:
    """(seconds from wall_start, bpm) for readings with sensor contact."""
    import csv

    readings = []
    for path in assets.heart_rate_csvs:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    stamp, bpm = float(row["timestamp"]), float(row["bpm"])
                except (KeyError, TypeError, ValueError):
                    continue
                if bpm <= 0 or str(row.get("sensor_contact", "True")).strip().lower() == "false":
                    continue
                if wall_start - HR_GAP_SEC <= stamp <= wall_end + HR_GAP_SEC:
                    readings.append((stamp - wall_start, bpm))
    readings.sort()
    return readings


def bpm_at(readings: list[tuple[float, float]], t: float) -> float | None:
    """Interpolated bpm at t, or None inside a sensor dropout."""
    if not readings:
        return None
    times = [r[0] for r in readings]
    i = bisect.bisect_left(times, t)
    if i < len(times) and times[i] == t:
        return readings[i][1]
    if i == 0:
        return readings[0][1] if times[0] - t <= HR_GAP_SEC / 2 else None
    if i == len(times):
        return readings[-1][1] if t - times[-1] <= HR_GAP_SEC / 2 else None
    (t0, b0), (t1, b1) = readings[i - 1], readings[i]
    if t1 - t0 > HR_GAP_SEC:
        return None
    return b0 + (b1 - b0) * (t - t0) / (t1 - t0)


def plot_pixel_bounds() -> tuple[float, float, float, float]:
    """(left, right, top, bottom) of the plotted area within the chart card."""
    return (PLOT_LEFT * CHART_W, PLOT_RIGHT * CHART_W,
            (1 - PLOT_TOP) * CHART_H, (1 - PLOT_BOTTOM) * CHART_H)


def render_hr_chart(readings, duration: float, title: str, out: Path,
                    shaded: list[tuple[float, float]] = ()) -> Path:
    """Static chart card: the whole window's bpm line on the fixed axis.
    `shaded` spans (e.g. talking pauses) get a grey band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    text_secondary, grid_color = "#52514e", "#e3e2dd"
    dpi = 100
    figure = plt.figure(figsize=(CHART_W / dpi, CHART_H / dpi), dpi=dpi)
    figure.patch.set_facecolor(mvp.LAYOUT_CARD_LIGHT_HEX)
    axis = figure.add_axes([PLOT_LEFT, PLOT_BOTTOM, PLOT_RIGHT - PLOT_LEFT, PLOT_TOP - PLOT_BOTTOM])
    axis.set_facecolor(mvp.LAYOUT_CARD_LIGHT_HEX)

    for start, end in shaded:
        axis.axvspan(start, end, color="#d6d4cd", alpha=0.6, linewidth=0)
    # Break the line at sensor dropouts rather than bridging them
    runs, current = [], []
    for t, bpm in readings:
        if current and t - current[-1][0] > HR_GAP_SEC:
            runs.append(current)
            current = []
        current.append((t, bpm))
    if current:
        runs.append(current)
    for run in runs:
        xs, ys = [r[0] for r in run], [r[1] for r in run]
        axis.plot(xs, ys, color=mvp.LAYOUT_CHART_COLOR, linewidth=2.0, solid_joinstyle="round", solid_capstyle="round")
        axis.fill_between(xs, ys, HR_MIN_BPM, color=mvp.LAYOUT_CHART_COLOR, alpha=0.10, linewidth=0)

    axis.set_xlim(0, duration)
    axis.set_ylim(HR_MIN_BPM, HR_MAX_BPM)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v // 60)}:{int(v % 60):02d}"))
    axis.set_ylabel("BPM", color=text_secondary, fontsize=11)
    axis.tick_params(colors=text_secondary, labelsize=10)
    axis.grid(True, color=grid_color, linewidth=1.0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(grid_color)
    figure.text(PLOT_LEFT, 0.93, title, color="#2b2a28", fontsize=14, fontweight="bold", va="center")
    if shaded:
        figure.text(PLOT_LEFT + 0.45, 0.93, "grey = paused to talk", color=text_secondary, fontsize=11, va="center")
    if not readings:
        axis.text(duration / 2, (HR_MIN_BPM + HR_MAX_BPM) / 2, "no heart-rate signal",
                  ha="center", va="center", color=text_secondary, fontsize=14)
    figure.savefig(out, facecolor=figure.get_facecolor(), dpi=dpi)
    plt.close(figure)
    return out


def build_hr_marker_track(readings, duration: float, out_dir: Path) -> Path:
    """Transparent overlay frames for the chart: a moving time line, a dot on
    the bpm line and the live bpm number. Returned as an ffmpeg concat list
    (one PNG per step) -- a per-frame overlay expression can't look up data.
    Steps are short enough that the line moves at most ~2 px per step."""
    from PIL import Image, ImageDraw, ImageFont

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("marker_*.png"):
        old.unlink()
    left, right, top, bottom = plot_pixel_bounds()
    px_per_sec = (right - left) / duration
    step = min(1.0, max(1 / 30, 2.0 / px_per_sec))
    font = ImageFont.truetype(BOLD_FONT_PATH, 30)
    accent = mvp.LAYOUT_CHART_COLOR.lstrip("#")
    accent_rgb = tuple(int(accent[i:i + 2], 16) for i in (0, 2, 4))
    dot = 14

    lines = []
    frame_count = max(1, math.ceil(duration / step))
    for index in range(frame_count):
        t = min(index * step, duration)
        x = left + t * px_per_sec
        bpm = bpm_at(readings, t)
        image = Image.new("RGBA", (CHART_W, CHART_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle([x - 1, top, x, bottom], fill=(0x52, 0x51, 0x4e, 150))
        if bpm is not None:
            y = bottom - (min(max(bpm, HR_MIN_BPM), HR_MAX_BPM) - HR_MIN_BPM) / (HR_MAX_BPM - HR_MIN_BPM) * (bottom - top)
            draw.ellipse([x - dot / 2, y - dot / 2, x + dot / 2, y + dot / 2], fill=(*accent_rgb, 255),
                         outline=(255, 255, 255, 255), width=2)
            label, color = f"{bpm:.0f} bpm", (*accent_rgb, 255)
        else:
            label, color = "no signal", (0x8a, 0x88, 0x82, 255)
        text_w = draw.textlength(label, font=font)
        draw.text((right - text_w, 0.07 * CHART_H - 17), label, font=font, fill=color)
        frame = out_dir / f"marker_{index:05d}.png"
        image.save(frame)
        lines += [f"file '{frame.resolve().as_posix()}'", f"duration {step:.4f}"]
    lines.append(lines[-2])  # concat demuxer ignores the last entry's duration unless repeated
    concat_list = out_dir / "markers.txt"
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return concat_list


# --- composition -------------------------------------------------------------

def compose_view(media: TrialMedia, output: Path, duration: float, front: Path, side: Path, overhead: Path,
                 face: Path, face_seek: float, face_caption: str, chart: Path, markers: Path,
                 narration_audio: str | None = None, audio_seek: float = 0.0,
                 subtitles: Path | None = None, caption: str | None = None) -> None:
    """Composite one view. front/side/overhead are pre-built clips that start
    at the view's t=0; face is seeked by face_seek. With narration_audio the
    soundtrack is narration over quiet front-GoPro sound, else GoPro sound only."""
    font = mvp.SUBTITLE_FONT_PATH.replace(":", "\\:")
    bold = BOLD_FONT_PATH.replace(":", "\\:")
    parts = [
        f"color=c=0x{mvp.LAYOUT_BG_HEX.lstrip('#')}:s={CANVAS_W}x{CANVAS_H}:r=30[bg]",
        f"[0:v]scale={FRONT_W}:{CAM_H},fps=30,setpts=PTS-STARTPTS[frontv]",
        f"[1:v]scale={SIDE_W}:{CAM_H},fps=30,setpts=PTS-STARTPTS[sidev]",
        f"[2:v]scale={OVERHEAD_W}:{CAM_H}:force_original_aspect_ratio=decrease,"
        f"pad={OVERHEAD_W}:{CAM_H}:(ow-iw)/2:(oh-ih)/2,fps=30,setpts=PTS-STARTPTS[overheadv]",
        f"[3:v]scale={FACE_W}:{FACE_H},fps=30,setpts=PTS-STARTPTS[facev]",
        "[5:v]fps=30,format=rgba[markerv]",
        f"[bg][frontv]overlay={CAM_X0}:{CAM_Y}[c1]",
        f"[c1][sidev]overlay={SIDE_X}:{CAM_Y}[c2]",
        f"[c2][overheadv]overlay={OVERHEAD_X}:{CAM_Y}[c3]",
        f"[c3]drawbox=x={BAR_X}:y={BAR_Y}:w={BAR_W}:h={BAR_H}:color=0x{bytes(mvp.LAYOUT_SUB_BG).hex()}:t=fill[c4]",
        f"[c4][facev]overlay={FACE_X}:{BOTTOM_Y}[c5]",
        f"[c5][4:v]overlay={CHART_X}:{BOTTOM_Y}[c6]",
        f"[c6][markerv]overlay={CHART_X}:{BOTTOM_Y}[c7]",
    ]
    label = "c7"

    def add_text(text_file: Path, text: str, x: str, y: str, size: int, fontfile: str, extra: str = "") -> None:
        nonlocal label
        text_file.write_text(text, encoding="utf-8")
        next_label = f"t{len(parts)}"
        parts.append(
            f"[{label}]drawtext=fontfile='{fontfile}':textfile='{_escape_path(text_file)}':"
            f"fontcolor=white:fontsize={size}:x={x}:y={y}{extra}[{next_label}]"
        )
        label = next_label

    text_dir = media.work_dir / f"_text_{output.stem}"
    text_dir.mkdir(parents=True, exist_ok=True)
    for old in text_dir.glob("*.txt"):
        old.unlink()
    tile_box = ":box=1:boxcolor=black@0.55:boxborderw=8"
    for name, x, y in (("Front", CAM_X0, CAM_Y), ("Side", SIDE_X, CAM_Y),
                       ("Overhead", OVERHEAD_X, CAM_Y), (face_caption, FACE_X, BOTTOM_Y)):
        add_text(text_dir / f"tile_{len(parts)}.txt", name, str(x + 14), str(y + 14), 20, bold, tile_box)

    if subtitles:
        max_text_width = BAR_W - 80
        for index, cue in enumerate(mvp.parse_srt(subtitles)):
            # Cue times are in narration-WAV time
            start, end = cue["start"] - audio_seek, cue["end"] - audio_seek
            text = cue["text"].replace("\n", " ").strip()
            if end <= 0 or start >= duration or not text:
                continue
            fitted, size = mvp._fit_subtitle_line(text, max_text_width, mvp.SUBTITLE_FONT_PATH)
            add_text(text_dir / f"cue_{index:04d}.txt", fitted, "(w-text_w)/2", f"{BAR_Y}+(({BAR_H}-text_h)/2)",
                     size, font, f":enable='between(t,{max(start, 0):.3f},{end:.3f})'")
    elif caption:
        add_text(text_dir / "caption.txt", caption, "(w-text_w)/2", f"{BAR_Y}+(({BAR_H}-text_h)/2)", 24, font,
                 ":fontcolor=0xb8b6b0")

    command = [
        media.ffmpeg, "-y", "-v", "error",
        "-i", str(front),
        "-i", str(side),
        "-i", str(overhead),
        "-ss", f"{max(face_seek, 0):.3f}", "-i", str(face),
        # -framerate 30 on looped PNGs: they default to 25fps, which beats
        # against the 30fps video (see mvp.compose_review_video)
        "-framerate", "30", "-loop", "1", "-i", str(chart),
        "-f", "concat", "-safe", "0", "-i", str(markers),
    ]
    if narration_audio:
        command += ["-ss", f"{max(audio_seek, 0):.3f}", "-i", narration_audio]
        parts += [
            f"[0:a]volume={PERFORMANCE_AUDIO_VOLUME}[performance_audio]",
            "[6:a]volume=1.0[narration_audio]",
            "[performance_audio][narration_audio]amix=inputs=2:duration=longest[audio]",
        ]
        audio_map = "[audio]"
    else:
        audio_map = "0:a?"

    # The graph goes in a file: one drawtext per subtitle cue overflows the
    # Windows command-line limit
    graph = text_dir / "filter_graph.txt"
    graph.write_text(";\n".join(parts), encoding="utf-8")
    command += [
        "-/filter_complex", str(graph),
        "-map", f"[{label}]", "-map", audio_map,
        "-t", f"{duration:.3f}",
        # Single keyframe: see the GOP-boundary note in mvp.compose_review_video
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-r", "30", "-g", "999999",
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(output),
    ]
    print(f"Rendering: {output}")
    result = mvp.run_media_tool(command, timeout=900 + 5 * duration)
    if result.returncode != 0:
        raise RuntimeError(f"composition failed: {result.stderr[-4000:]}")
    print(f"Created: {output}")


def _single_play_timeline(source_start: float, duration: float) -> dict:
    return {"review_duration_sec": duration, "segments": [{
        "type": "play", "review_start_sec": 0.0, "review_end_sec": duration,
        "source_start_sec": source_start, "source_end_sec": source_start + duration,
    }]}


def render_narration_view(trial: Path) -> Path:
    media = load_trial_media(trial)
    assets = media.assets
    if not assets.narration_audio:
        raise RuntimeError("narration audio is required")
    timing = mvp.build_timing_report(assets)
    timeline = mvp.build_pause_timeline(assets)
    duration = timeline["review_duration_sec"]

    clips = {}
    for name, source, offset in (("front", media.front, media.front_offset),
                                 ("side", media.side, media.side_offset),
                                 ("overhead", media.overhead, 0.0)):
        clips[name] = mvp.build_pause_aware_video(
            media.ffmpeg, source, offset, timeline, 0.0, duration, media.work_dir / f"narration_{name}.mp4")

    readings = read_hr(assets, timing["review_wall_start"], timing["review_wall_start"] + duration)
    pauses = [(s["review_start_sec"], s["review_end_sec"]) for s in timeline["segments"] if s["type"] == "hold"]
    chart = render_hr_chart(readings, duration, "Heart rate during narration",
                            media.work_dir / "narration_chart.png", pauses)
    markers = build_hr_marker_track(readings, duration, media.work_dir / "narration_markers")

    face_offset, audio_offset = mvp.review_media_offsets(assets)
    subtitles = find_subtitles(media.trial, assets)
    if not subtitles:
        print("  Note: no timed subtitles for this trial; the subtitle bar stays empty")
    output = media.trial / "MVP" / "narration_view.mp4"
    compose_view(
        media, output, duration, clips["front"], clips["side"], clips["overhead"],
        face=Path(assets.face_video), face_seek=face_offset, face_caption="Face (narrating)",
        chart=chart, markers=markers, narration_audio=assets.narration_audio,
        audio_seek=audio_offset, subtitles=subtitles,
    )
    return output


def performance_window(media: TrialMedia, timing: dict) -> tuple[float, float]:
    """(overhead position, duration) of the performance phase, limited to
    what the overhead camera recorded."""
    manifest = mvp.load_json(media.assets.sync_manifest) or {}
    bounds = phase_bounds(manifest, "performance")
    if not bounds:
        raise RuntimeError("manifest has no performance phase")
    overhead_start = timing["performance_wall_start"]  # last overhead_recorder_start
    if overhead_start is None:
        raise RuntimeError("manifest has no overhead recorder start")
    start = max(bounds[0], overhead_start)
    ends = [e for e in (bounds[1], timing["performance_wall_end"]) if e is not None]
    overhead_duration = (mvp.media_info(str(media.overhead)) or {}).get("duration_sec")
    if overhead_duration:
        ends.append(overhead_start + overhead_duration)
    end = min(ends)
    if end - start < 1.0:
        raise RuntimeError("performance phase and overhead recording don't overlap")
    return start - overhead_start, end - start


def performance_face_timeline(review_timeline: dict, overhead_start: float, duration: float) -> dict:
    """Map performance time onto the face video: for each moment of the
    performance, the review time at which the participant watched it.
    Talking pauses are skipped; moments the review never reached show a
    labelled frozen frame."""
    segments, cursor, last_review = [], 0.0, 0.0
    overhead_end = overhead_start + duration
    for seg in review_timeline["segments"]:
        a, b = seg["source_start_sec"], seg["source_end_sec"]
        if seg["type"] != "play" or b <= a:
            continue
        lo, hi = max(a, overhead_start), min(b, overhead_end)
        if hi - lo <= 1e-3:
            continue
        scale = (seg["review_end_sec"] - seg["review_start_sec"]) / (b - a)
        review_lo = seg["review_start_sec"] + (lo - a) * scale
        review_hi = seg["review_start_sec"] + (hi - a) * scale
        out_lo, out_hi = lo - overhead_start, hi - overhead_start
        if out_lo - cursor > 1e-3:
            segments.append({"type": "hold", "review_start_sec": cursor, "review_end_sec": out_lo,
                             "source_start_sec": review_lo, "source_end_sec": review_lo,
                             "label": "not watched in review"})
        segments.append({"type": "play", "review_start_sec": out_lo, "review_end_sec": out_hi,
                         "source_start_sec": review_lo, "source_end_sec": review_hi})
        cursor, last_review = out_hi, review_hi
    if duration - cursor > 1e-3:
        segments.append({"type": "hold", "review_start_sec": cursor, "review_end_sec": duration,
                         "source_start_sec": last_review, "source_end_sec": last_review,
                         "label": "not watched in review"})
    return {"review_duration_sec": duration, "segments": segments}


def render_performance_view(trial: Path) -> Path:
    media = load_trial_media(trial)
    assets = media.assets
    timing = mvp.build_timing_report(assets)
    overhead_pos, duration = performance_window(media, timing)
    wall_start = timing["performance_wall_start"] + overhead_pos

    timeline = _single_play_timeline(overhead_pos, duration)
    clips = {}
    for name, source, offset in (("front", media.front, media.front_offset),
                                 ("side", media.side, media.side_offset),
                                 ("overhead", media.overhead, 0.0)):
        clips[name] = mvp.build_pause_aware_video(
            media.ffmpeg, source, offset, timeline, 0.0, duration, media.work_dir / f"performance_{name}.mp4")

    face_offset, _ = mvp.review_media_offsets(assets)
    face_timeline = performance_face_timeline(mvp.build_pause_timeline(assets), overhead_pos, duration)
    face_clip = mvp.build_pause_aware_video(
        media.ffmpeg, Path(assets.face_video), face_offset, face_timeline, 0.0, duration,
        media.work_dir / "performance_face.mp4")

    readings = read_hr(assets, wall_start, wall_start + duration)
    chart = render_hr_chart(readings, duration, "Heart rate during performance", media.work_dir / "performance_chart.png")
    markers = build_hr_marker_track(readings, duration, media.work_dir / "performance_markers")

    output = media.trial / "MVP" / "performance_view.mp4"
    compose_view(
        media, output, duration, clips["front"], clips["side"], clips["overhead"],
        face=face_clip, face_seek=0.0, face_caption="Face (while watching this moment in review)",
        chart=chart, markers=markers,
        caption="Performance in real time  -  face shown from when this moment was watched during review",
    )
    return output


# --- CLI -----------------------------------------------------------------------

VIEWS = {"narration": render_narration_view, "performance": render_performance_view}
OUTPUT_NAMES = {"narration": "narration_view.mp4", "performance": "performance_view.mp4"}


def single_command(args: argparse.Namespace) -> int:
    output = VIEWS[args.command](args.trial)
    print(f"OK: {output}")
    return 0


def batch_command(args: argparse.Namespace) -> int:
    views = list(VIEWS) if args.view == "both" else [args.view]
    trials = sorted(p for p in args.data_root.iterdir() if p.is_dir() and p.name.startswith("P"))
    if args.start:
        trials = [t for t in trials if t.name >= args.start]
    jobs = [(t, v) for t in trials for v in views
            if not (args.skip_existing and (t / "MVP" / OUTPUT_NAMES[v]).is_file())]

    print(f"Rendering {len(jobs)} view(s) across {len(trials)} trial(s).")
    results = {"ok": [], "failed": []}
    for index, (trial, view) in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {trial.name} {view}")
        try:
            output = VIEWS[view](trial)
            results["ok"].append({"trial": trial.name, "view": view, "output": str(output)})
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            results["failed"].append({"trial": trial.name, "view": view, "error": str(exc)})

    print(f"\nDone. {len(results['ok'])}/{len(jobs)} succeeded.")
    for failure in results["failed"]:
        print(f"  {failure['trial']} {failure['view']}: {failure['error'][:300]}")
    report = args.data_root.parent / "logs" / "views_render_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report: {report}")
    return 0 if not results["failed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render narration-view and performance-view videos")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in VIEWS:
        sub = subparsers.add_parser(name, help=f"render MVP/{OUTPUT_NAMES[name]} for one trial")
        sub.add_argument("--trial", type=Path, required=True)
        sub.set_defaults(handler=single_command)
    batch = subparsers.add_parser("batch", help="render views for every trial, continuing past failures")
    batch.add_argument("--data-root", type=Path, required=True)
    batch.add_argument("--view", choices=["both", *VIEWS], default="both")
    batch.add_argument("--start", help="trial folder name to start from, e.g. P10")
    batch.add_argument("--skip-existing", action="store_true", help="skip views whose output already exists")
    batch.set_defaults(handler=batch_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
