"""Run OpenFace FeatureExtraction over a set of videos."""

import csv
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .config import Config


@dataclass
class Video:
    path: Path
    video_id: str  # unique, filesystem-safe name used for all outputs
    relative_path: str  # path relative to its input dir, '/'-separated (used for metadata)


def make_video_id(path: Path, input_dir: Path) -> str:
    """Build an ID from the path relative to its input dir.

    GoPro names (GH017313.MP4, ...) repeat across cameras and offloads, so the
    filename alone is not unique.
    """
    rel = path.relative_to(input_dir).with_suffix("")
    return "__".join(re.sub(r"[^A-Za-z0-9_-]+", "_", part).strip("_") for part in rel.parts)


def find_videos(cfg: Config) -> list[Video]:
    videos = []
    for input_dir in cfg.input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if cfg.patterns:
            candidates = {p for pattern in cfg.patterns for p in input_dir.glob(pattern)}
        else:
            candidates = {p for p in input_dir.rglob("*") if p.suffix.lower() in cfg.extensions}
        for path in sorted(candidates):
            if path.is_file() and path.stat().st_size >= cfg.min_size_mb * 1024**2:
                videos.append(Video(path, make_video_id(path, input_dir),
                                    path.relative_to(input_dir).as_posix()))

    ids = [v.video_id for v in videos]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate video IDs (same relative path in two input dirs?): {sorted(dupes)}")
    return videos


def done_marker(cfg: Config, video_id: str) -> Path:
    return cfg.openface_dir / f"{video_id}.done"


def run_openface(cfg: Config, video: Video) -> tuple[str, float]:
    """Process one video. Returns (status, seconds)."""
    cmd = [str(cfg.openface_exe), "-f", str(video.path),
           "-out_dir", str(cfg.openface_dir), "-of", video.video_id, *cfg.openface_flags]
    if cfg.save_tracked_video:
        cmd.append("-tracked")

    start = time.time()
    log_path = cfg.logs_dir / f"{video.video_id}.log"
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        # OpenFace looks for its model files relative to the working directory
        result = subprocess.run(cmd, cwd=cfg.openface_exe.parent, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    csv_path = cfg.openface_dir / f"{video.video_id}.csv"
    if result.returncode == 0 and csv_path.exists():
        # The marker records where the video came from, for metadata in `analyze`
        done_marker(cfg, video.video_id).write_text(json.dumps(
            {"source_path": str(video.path), "relative_path": video.relative_path}))
        return "ok", elapsed
    return f"failed (exit {result.returncode}, see {log_path.name})", elapsed


def extract_all(cfg: Config, limit: int | None = None, force: bool = False) -> None:
    if not cfg.openface_exe.exists():
        raise FileNotFoundError(
            f"OpenFace not found at {cfg.openface_exe}. Run scripts/setup_openface.ps1 first."
        )
    for d in (cfg.openface_dir, cfg.logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    videos = find_videos(cfg)
    todo = [v for v in videos if force or not done_marker(cfg, v.video_id).exists()]
    n_done = len(videos) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    print(f"Found {len(videos)} videos ({n_done} already done); "
          f"processing {len(todo)} with {cfg.workers} worker(s).")
    if not todo:
        return

    manifest_path = cfg.output_dir / "manifest.csv"
    new_file = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as mf, ThreadPoolExecutor(cfg.workers) as pool:
        writer = csv.writer(mf)
        if new_file:
            writer.writerow(["video_id", "source_path", "status", "seconds", "finished_at"])
        futures = {pool.submit(run_openface, cfg, v): v for v in todo}
        failures = 0
        for fut in tqdm(as_completed(futures), total=len(futures), unit="video"):
            video = futures[fut]
            try:
                status, secs = fut.result()
            except Exception as e:  # e.g. permission error writing the log
                status, secs = f"error: {e}", 0.0
            if status != "ok":
                failures += 1
                tqdm.write(f"[{status}] {video.path}")
            writer.writerow([video.video_id, video.path, status, round(secs, 1),
                             time.strftime("%Y-%m-%d %H:%M:%S")])
            mf.flush()

    print(f"Done. {len(todo) - failures} succeeded, {failures} failed. Manifest: {manifest_path}")
