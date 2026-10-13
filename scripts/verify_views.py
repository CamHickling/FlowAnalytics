#!/usr/bin/env python3
"""Check rendered narration/performance views (see views.py).

For each trial and view: the file exists, is 1920x1080 at 30 fps, its length
matches the timeline it was built from, and its audio isn't silent. Also
writes one contact sheet per trial (frames from both views) for a visual check.

Usage:
    python verify_views.py --data-root <dir> [--out <dir>] [--only P01 P26]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mvp
import views

DURATION_TOLERANCE_SEC = 1.0
SILENT_DB = -60.0          # mean volume below this counts as no audio
SHEET_TIMES = (0.1, 0.35, 0.6, 0.85)   # fractions of each video to sample
THUMB_W, THUMB_H = 480, 270


def probe(ffprobe: str, path: Path) -> dict:
    result = mvp.run_media_tool(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,r_frame_rate", "-of", "json", str(path)],
        timeout=120,
    )
    data = json.loads(result.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    num, _, den = video.get("r_frame_rate", "0/1").partition("/")
    return {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": float(num) / float(den or 1) if den != "0" else 0,
        "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    }


def mean_volume(ffmpeg: str, path: Path) -> float | None:
    result = mvp.run_media_tool(
        [ffmpeg, "-v", "info", "-nostats", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
        timeout=1800,
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", result.stderr)
    if not match:
        return None
    return float("-inf") if match.group(1) == "-inf" else float(match.group(1))


def expected_durations(trial: Path) -> dict[str, float]:
    """The length each view should have, from the same timelines views.py uses."""
    expected = {}
    try:
        media = views.load_trial_media(trial)
        timing = mvp.build_timing_report(media.assets)
    except Exception:
        return expected
    try:
        start, end = views.narration_window(media, timing, mvp.build_pause_timeline(media.assets))
        expected["narration"] = end - start
    except Exception:
        pass
    try:
        expected["performance"] = views.performance_window(media, timing)[1]
    except Exception:
        pass
    return expected


def contact_sheet(ffmpeg: str, trial: Path, videos: dict[str, tuple[Path, float]], out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = [(view, path, duration) for view, (path, duration) in videos.items()]
    label_h = 34
    sheet = Image.new("RGB", (THUMB_W * len(SHEET_TIMES), (THUMB_H + label_h) * len(rows)), (18, 19, 21))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(views.BOLD_FONT_PATH, 18)
    tmp = out.with_suffix(".frame.png")
    for row, (view, path, duration) in enumerate(rows):
        top = row * (THUMB_H + label_h)
        for col, fraction in enumerate(SHEET_TIMES):
            t = duration * fraction
            result = mvp.run_media_tool(
                [ffmpeg, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
                 "-frames:v", "1", "-vf", f"scale={THUMB_W}:{THUMB_H}", str(tmp)],
                timeout=120,
            )
            if result.returncode == 0 and tmp.is_file():
                sheet.paste(Image.open(tmp).convert("RGB"), (col * THUMB_W, top + label_h))
            draw.text((col * THUMB_W + 8, top + 7), f"{trial.name.split('_')[0]} {view} @ {int(t // 60)}:{int(t % 60):02d}",
                      font=font, fill=(230, 230, 225))
    tmp.unlink(missing_ok=True)
    sheet.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check rendered narration/performance views")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="report + contact sheets folder (default: <data-root>/../logs/views_check)")
    parser.add_argument("--only", nargs="+", help="trial folder prefixes to check, e.g. P01 P26")
    args = parser.parse_args()

    ffmpeg = mvp._resolve_ffmpeg_or_error()
    ffprobe = mvp._resolve_ffprobe()
    if not ffmpeg or not ffprobe:
        print("ERROR: ffmpeg and ffprobe are required", file=sys.stderr)
        return 2
    out_dir = args.out or args.data_root.parent / "logs" / "views_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = sorted(p for p in args.data_root.iterdir() if p.is_dir() and p.name.startswith("P"))
    if args.only:
        trials = [t for t in trials if any(t.name.startswith(prefix + "_") for prefix in args.only)]

    report, problems = [], 0
    for trial in trials:
        expected = expected_durations(trial)
        found = {}
        for view, name in views.OUTPUT_NAMES.items():
            path = trial / "MVP" / name
            entry = {"trial": trial.name, "view": view, "issues": []}
            if not path.is_file():
                entry["issues"].append("not rendered")
                report.append(entry)
                continue
            info = probe(ffprobe, path)
            entry.update(info)
            if (info["width"], info["height"]) != (views.CANVAS_W, views.CANVAS_H):
                entry["issues"].append(f"size {info['width']}x{info['height']}")
            if abs(info["fps"] - 30) > 0.01:
                entry["issues"].append(f"{info['fps']:.2f} fps")
            want = expected.get(view)
            entry["expected_duration"] = want
            if want is None:
                entry["issues"].append("couldn't work out expected length")
            elif abs(info["duration"] - want) > DURATION_TOLERANCE_SEC:
                entry["issues"].append(f"length {info['duration']:.1f}s, expected {want:.1f}s")
            if not info["has_audio"]:
                entry["issues"].append("no audio track")
            else:
                volume = mean_volume(ffmpeg, path)
                entry["mean_volume_db"] = volume
                if volume is None or volume < SILENT_DB:
                    entry["issues"].append(f"audio silent ({volume} dB)")
            found[view] = (path, info["duration"])
            report.append(entry)
        if found:
            contact_sheet(ffmpeg, trial, found, out_dir / f"{trial.name.split('_')[0]}_sheet.png")
        for entry in report[-len(views.OUTPUT_NAMES):]:
            status = "OK" if not entry["issues"] else "; ".join(entry["issues"])
            problems += bool(entry["issues"])
            print(f"{entry['trial'][:12]:12s} {entry['view']:11s} {status}")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{len(report) - problems}/{len(report)} views OK. Report and contact sheets: {out_dir}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
