"""Place face-cam frames on the performance timeline for FlowAnalytics trials.

Every trial's logs share one wall clock:
  <trial>/*_sync_manifest.json        recorder start/stop, scoring playback start/stop
  <trial>/review/*_review_timestamps.json   each play (resume) / pause press during review,
                                            with the overhead video position at that press
  <trial>/MVP/sync_offsets.json       front/side GoPro seconds at performance position 0

The face cam starts recording when a phase begins, before the participant presses play,
so only the stretch from the first press until the performance has played out lines up
with what they were watching.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION_PHASE = {"narration": "review", "scoring": "scoring"}


@dataclass
class Segment:
    state: str        # "playing" or "paused"
    t0: float         # seconds since the first press of play
    t1: float
    perf0: float      # overhead (performance) video position over the segment
    perf1: float


@dataclass
class SyncWindow:
    condition: str
    face_start_s: float | None   # seconds into the face video where the window starts
    duration_s: float | None
    segments: list[Segment]
    performance_duration_s: float | None
    front_offset_s: float | None
    side_offset_s: float | None
    flags: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.face_start_s is not None and bool(self.duration_s) and self.duration_s > 0


def _load_json(path: Path | None):
    return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None


def _first(trial: Path, pattern: str) -> Path | None:
    return next(iter(sorted(trial.glob(pattern))), None)


def _event(events: list, name: str, phase: str | None = None) -> float | None:
    return next((e["wall_time"] for e in events
                 if e.get("event") == name and (phase is None or e.get("phase") == phase)), None)


def probe_duration(ffprobe: Path, video: Path) -> float | None:
    out = subprocess.run([str(ffprobe), "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


# The review player advances the overhead video at about 97% of real time (443.8 s of
# performance takes 457.5 s to play). Same model as scripts/mvp.py: measured per trial
# from its resume->pause segments, with the dataset median for trials that never paused.
DEFAULT_PLAYBACK_RATE = 0.970


def playback_rate(presses: list[dict]) -> float:
    rates = []
    for a, b in zip(presses, presses[1:]):
        wall = b["wall_time"] - a["wall_time"]
        if a["type"] == "resume" and b["type"] == "pause" and wall > 2:
            rates.append((b["video_position_sec"] - a["video_position_sec"]) / wall)
    rates = [r for r in rates if 0.8 < r < 1.2]
    return float(np.median(rates)) if rates else DEFAULT_PLAYBACK_RATE


def _play_out(t0: float, pos: float, perf_dur: float | None, t_stop: float | None,
              rate: float) -> tuple[float, float]:
    """End (t1, pos1) of a final segment playing from position `pos` at time `t0`:
    at the player's speed until the performance ends, or until the logged stop."""
    remaining = (perf_dur if perf_dur is not None else pos) - pos
    t1 = t0 + remaining / rate
    if t_stop is not None and t_stop < t1:
        return t_stop, pos + (t_stop - t0) * rate  # stopped early
    return t1, pos + remaining


def build_window(trial: Path, condition: str, face_video: Path, ffprobe: Path) -> SyncWindow:
    phase = CONDITION_PHASE[condition]
    manifest = _load_json(_first(trial, "*_sync_manifest.json")) or {}
    events = manifest.get("events", [])
    offsets = _load_json(trial / "MVP" / "sync_offsets.json") or {}
    flags: list[str] = []

    rec_start = _event(events, "face_recorder_start", phase)
    rec_stop = _event(events, "face_recorder_stop", phase)
    perf_dur = next((e.get("duration_sec") for e in events
                     if e.get("event") == f"{phase}_video_player_shown"), None)
    file_dur = probe_duration(ffprobe, face_video)

    if not manifest:
        flags.append("no_sync_manifest")
    if rec_start is None:
        flags.append("no_face_recorder_start")
    if perf_dur is None:
        flags.append("no_performance_duration")
    if not offsets:
        flags.append("no_gopro_sync_offsets")
    if file_dur is not None and rec_start is not None and rec_stop is not None:
        gap = file_dur - (rec_stop - rec_start)
        if abs(gap) > 1.0:
            flags.append(f"face_file_{'longer' if gap > 0 else 'shorter'}_than_log_by_{abs(gap):.0f}s")

    segments: list[Segment] = []
    if condition == "narration":
        presses = _load_json(_first(trial, "review/*review_timestamps.json")) or []
        presses = [p for p in presses if p.get("type") in ("resume", "pause")]
        play_stop = _event(events, "review_playback_stop")
        if not presses or presses[0]["type"] != "resume":
            flags.append("no_play_press_logged")
        else:
            t_play = presses[0]["wall_time"]
            for cur, nxt in zip(presses, presses[1:] + [None]):
                t0 = cur["wall_time"] - t_play
                pos = cur["video_position_sec"]
                if cur["type"] == "resume":
                    if nxt is not None:
                        t1, pos1 = nxt["wall_time"] - t_play, nxt["video_position_sec"]
                    else:  # plays out to the end of the performance
                        t1, pos1 = _play_out(t0, pos, perf_dur, play_stop - t_play if play_stop else None,
                                             playback_rate(presses))
                        if pos1 < (perf_dur or pos1) - 0.5:
                            flags.append("review_stopped_before_performance_ended")
                    segments.append(Segment("playing", t0, t1, pos, pos1))
                else:
                    t1 = (nxt["wall_time"] if nxt is not None else (play_stop or cur["wall_time"])) - t_play
                    segments.append(Segment("paused", t0, t1, pos, pos))
            face_start = t_play - rec_start if rec_start is not None else None
    else:
        play_start = _event(events, "scoring_playback_start")
        play_stop = _event(events, "scoring_playback_stop")
        if play_start is None:
            flags.append("no_play_press_logged")
        else:
            t1, pos1 = _play_out(0.0, 0.0, perf_dur, play_stop - play_start if play_stop else None,
                                 DEFAULT_PLAYBACK_RATE)
            if perf_dur is not None and pos1 < perf_dur - 1.0:
                flags.append(f"scoring_playback_stopped_early_{pos1:.0f}s_of_{perf_dur:.0f}s")
            segments.append(Segment("playing", 0.0, t1, 0.0, pos1))
            face_start = play_start - rec_start if rec_start is not None else None

    if not segments:
        face_start, duration = None, None
    else:
        duration = segments[-1].t1
        if file_dur is not None and face_start is not None and face_start + duration > file_dur + 0.5:
            flags.append("window_runs_past_end_of_face_file")
            duration = max(file_dur - face_start, 0.0)
        if duration is not None and duration < 30:
            flags.append(f"short_window_{duration:.0f}s")

    return SyncWindow(condition, face_start, duration, segments, perf_dur,
                      offsets.get("front_offset_sec"), offsets.get("side_offset_sec"), flags)


def annotate(frames: pd.DataFrame, window: SyncWindow) -> pd.DataFrame:
    """Add timeline columns to OpenFace frames whose `timestamp` is seconds into the window."""
    t = frames["timestamp"].to_numpy()
    state = np.full(len(t), "outside", dtype=object)
    perf = np.full(len(t), np.nan)
    for seg in window.segments:
        inside = (t >= seg.t0) & (t < seg.t1)
        state[inside] = seg.state
        span = seg.t1 - seg.t0
        perf[inside] = seg.perf0 + (t[inside] - seg.t0) / span * (seg.perf1 - seg.perf0) if span > 0 else seg.perf0
    out = pd.DataFrame({
        "window_time_s": t,
        "face_video_time_s": t + (window.face_start_s or 0.0),
        "state": state,
        "performance_time_s": perf,
    })
    for cam, offset in (("front", window.front_offset_s), ("side", window.side_offset_s)):
        out[f"{cam}_gopro_time_s"] = perf + offset if offset is not None else np.nan
    return out
