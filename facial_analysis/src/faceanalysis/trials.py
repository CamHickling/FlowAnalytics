"""AU analysis of FlowAnalytics trial face videos, synced to the performance playback.

For each trial and condition (narration in review/, scoring in scoring/):
  1. work out the playback window from the trial logs (trial_sync)
  2. cut that window out of the face video and run OpenFace on it
  3. put every frame on the performance timeline, find AU events, count them, flag anything odd
"""

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from .analyze import clean_frames, load_openface_csv, summarize
from .config import Config, load_config
from .events import au_counts, au_events, cumulative_event_counts
from .extract import Video, run_openface
from .trial_sync import SyncWindow, Segment, annotate, build_window, probe_duration

CONDITION_FOLDERS = {"narration": "review", "scoring": "scoring"}
TRIAL_NAME = re.compile(r"^(?P<participant>P\d+)(?P<session>B?)_")


@dataclass
class TrialSettings:
    data_root: Path
    ffmpeg: Path
    ffprobe: Path
    keep_clips: bool
    min_event_s: float
    max_gap_s: float


@dataclass
class TrialVideo:
    trial: Path
    condition: str
    face_video: Path | None

    @property
    def video_id(self) -> str:
        return f"{self.trial.name}__{self.condition}"


def load_settings(config_path: str | Path) -> tuple[Config, TrialSettings]:
    cfg = load_config(config_path)
    raw = yaml.safe_load(Path(config_path).read_text())["trials"]
    resolve = lambda p: (cfg.root / p).resolve() if not Path(p).is_absolute() else Path(p)
    return cfg, TrialSettings(
        data_root=cfg.input_dirs[0],
        ffmpeg=resolve(raw["ffmpeg"]),
        ffprobe=resolve(raw["ffprobe"]),
        keep_clips=bool(raw.get("keep_clips", False)),
        min_event_s=float(raw["events"]["min_duration_s"]),
        max_gap_s=float(raw["events"]["max_gap_s"]),
    )


def find_trial_videos(settings: TrialSettings) -> list[TrialVideo]:
    found = []
    for trial in sorted(p for p in settings.data_root.iterdir() if p.is_dir() and TRIAL_NAME.match(p.name)):
        for condition, folder in CONDITION_FOLDERS.items():
            # Most trials name it *_face_<condition>.mp4; P06 kept the recorder's *_face_cam.mp4
            candidates = sorted((trial / folder).glob(f"*_face_{condition}.mp4")) or \
                         sorted((trial / folder).glob("*face_cam*.mp4"))
            found.append(TrialVideo(trial, condition, candidates[0] if candidates else None))
    return found


def _dirs(cfg: Config) -> dict[str, Path]:
    d = {name: cfg.output_dir / name for name in ("openface", "clips", "windows", "timeline", "logs", "plots")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def window_for(tv: TrialVideo, settings: TrialSettings) -> SyncWindow:
    window = build_window(tv.trial, tv.condition, tv.face_video, settings.ffprobe)
    if not window.usable:
        # Can't place it on the performance timeline: analyse the whole file, marked unsynced
        window.flags.append("unsynced_whole_file_analysed")
        window.face_start_s = 0.0
        window.duration_s = probe_duration(settings.ffprobe, tv.face_video)
        window.segments = []
    return window


def cut_clip(settings: TrialSettings, src: Path, start: float, duration: float, dst: Path, log: Path) -> bool:
    # Seek before -i with re-encoding is frame-accurate; audio isn't needed for OpenFace
    cmd = [str(settings.ffmpeg), "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{duration:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", str(dst)]
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode == 0 and dst.exists()


def process_one(cfg: Config, settings: TrialSettings, tv: TrialVideo, dirs: dict) -> str:
    window = window_for(tv, settings)
    (dirs["windows"] / f"{tv.video_id}.json").write_text(json.dumps(
        {"source_path": str(tv.face_video), **asdict(window)}, indent=2))
    clip = dirs["clips"] / f"{tv.video_id}.mp4"
    if not cut_clip(settings, tv.face_video, window.face_start_s, window.duration_s, clip,
                    dirs["logs"] / f"{tv.video_id}.cut.log"):
        return "failed: could not cut clip"
    status, _ = run_openface(cfg, Video(clip, tv.video_id, tv.video_id))
    if status == "ok" and not settings.keep_clips:
        clip.unlink()
    return status


def extract_trials(cfg: Config, settings: TrialSettings, limit: int | None = None,
                   only: list[str] | None = None, force: bool = False) -> None:
    dirs = _dirs(cfg)
    videos = [tv for tv in find_trial_videos(settings) if tv.face_video is not None]
    if only:
        videos = [tv for tv in videos if any(tv.trial.name.startswith(o) for o in only)]
    todo = [tv for tv in videos if force or not (dirs["openface"] / f"{tv.video_id}.done").exists()]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(videos)} trial face videos; processing {len(todo)} with {cfg.workers} worker(s).")
    failures = 0
    with ThreadPoolExecutor(cfg.workers) as pool:
        futures = {pool.submit(process_one, cfg, settings, tv, dirs): tv for tv in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="video"):
            tv = futures[fut]
            try:
                status = fut.result()
            except Exception as e:
                status = f"error: {e}"
            if status != "ok":
                failures += 1
                tqdm.write(f"[{status}] {tv.video_id}")
    print(f"Extraction done: {len(todo) - failures} ok, {failures} failed.")


def _load_window(path: Path) -> tuple[str, SyncWindow]:
    raw = json.loads(path.read_text())
    source = raw.pop("source_path")
    raw["segments"] = [Segment(**s) for s in raw["segments"]]
    return source, SyncWindow(**raw)


def analyze_trials(cfg: Config, settings: TrialSettings) -> None:
    dirs = _dirs(cfg)
    all_videos = find_trial_videos(settings)
    summaries, all_events, all_counts, flag_rows = [], [], [], []

    for tv in tqdm(all_videos, unit="video"):
        meta = {"video_id": tv.video_id, "trial": tv.trial.name, "condition": tv.condition,
                **_trial_meta(tv.trial.name)}
        if tv.face_video is None:
            flag_rows.append({**meta, "flag": f"missing_face_video_in_{CONDITION_FOLDERS[tv.condition]}"})
            continue
        if not (dirs["openface"] / f"{tv.video_id}.done").exists():
            flag_rows.append({**meta, "flag": "not_processed_yet_or_openface_failed"})
            continue

        source, window = _load_window(dirs["windows"] / f"{tv.video_id}.json")
        raw = load_openface_csv(dirs["openface"] / f"{tv.video_id}.csv")
        clean = clean_frames(raw, cfg.min_confidence)
        timeline = pd.concat([annotate(clean, window), clean.drop(columns=["timestamp"])], axis=1)
        if not window.segments:
            timeline["state"] = "unsynced"
        fps = 1 / timeline["window_time_s"].diff().median()
        events = au_events(timeline, fps, settings.min_event_s, settings.max_gap_s)
        counts = au_counts(timeline, events, fps)
        timeline = pd.concat([timeline, cumulative_event_counts(timeline, events)], axis=1)
        timeline.to_csv(dirs["timeline"] / f"{tv.video_id}.csv", index=False, float_format="%.4f")

        stats = summarize(clean)
        flags = list(window.flags)
        if stats["valid_fraction"] < 0.5:
            flags.append(f"low_tracking_{stats['valid_fraction']:.0%}_valid")
        if tv.condition == "narration":
            flags.append("mouth_aus_speech_confounded")
        playing = timeline["state"] == "playing"
        if playing.any() and timeline.loc[playing, "valid"].mean() < 0.5:
            flags.append(f"low_tracking_while_playing_{timeline.loc[playing, 'valid'].mean():.0%}")
        for flag in flags:
            flag_rows.append({**meta, "flag": flag})

        summaries.append({
            **meta,
            "window_start_in_face_video_s": round(window.face_start_s, 3),
            "window_duration_s": round(window.duration_s, 2),
            "playing_s": round(sum(s.t1 - s.t0 for s in window.segments if s.state == "playing"), 2),
            "paused_s": round(sum(s.t1 - s.t0 for s in window.segments if s.state == "paused"), 2),
            "n_pauses": sum(s.state == "paused" for s in window.segments),
            "n_au_events": len(events),
            "flags": "; ".join(flags),
            **stats,
            "source_path": source,
        })
        all_events.append(events.assign(**meta))
        all_counts.append(counts.assign(**meta))

    out = cfg.output_dir
    lead = ["video_id", "trial", "participant", "session", "condition"]
    def write(frames, name):
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df[lead + [c for c in df.columns if c not in lead]].to_csv(out / name, index=False, float_format="%.4f")
    write([pd.DataFrame(summaries)] if summaries else [], "summary.csv")
    write(all_events, "events.csv")
    write(all_counts, "au_counts.csv")
    pd.DataFrame(flag_rows, columns=lead + ["flag"]).to_csv(out / "flags.csv", index=False)

    n_flagged = len({r["video_id"] for r in flag_rows
                     if r["flag"] not in ("mouth_aus_speech_confounded", "not_processed_yet_or_openface_failed")})
    print(f"Wrote summary.csv ({len(summaries)} videos), events.csv, au_counts.csv, flags.csv to {out}")
    print(f"{n_flagged} video(s) have flags beyond the narration speech note - see flags.csv")


def _trial_meta(name: str) -> dict:
    m = TRIAL_NAME.match(name)
    return {"participant": m["participant"], "session": "B" if m["session"] else "main"}


def plot_trials(cfg: Config) -> None:
    from .plot import plot_video
    dirs = _dirs(cfg)
    files = sorted(dirs["timeline"].glob("*.csv"))
    for path in tqdm(files, unit="plot"):
        frames = pd.read_csv(path).rename(columns={"window_time_s": "timestamp"})
        plot_video(frames, path.stem, cfg.plot_smoothing_s, dirs["plots"] / f"{path.stem}.png")
    print(f"Wrote {len(files)} plot(s) to {dirs['plots']}")
