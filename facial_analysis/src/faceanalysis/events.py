"""AU events (continuous stretches where OpenFace marks an AU present) and counts."""

import numpy as np
import pandas as pd

from .analyze import AU_PRESENCE


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of True runs, end exclusive."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def au_events(frames: pd.DataFrame, fps: float, min_duration_s: float, max_gap_s: float) -> pd.DataFrame:
    """One row per AU event.

    `frames` is a cleaned frame table (invalid frames are NaN) with timeline columns.
    Presence gaps up to `max_gap_s` are bridged, but never across untracked frames,
    and events shorter than `min_duration_s` are dropped.
    """
    rows = []
    max_gap = int(round(max_gap_s * fps))
    min_len = max(1, int(round(min_duration_s * fps)))
    for col in frames.filter(regex=AU_PRESENCE).columns:
        au = col[:4]
        present = frames[col].to_numpy() == 1
        tracked = frames["valid"].to_numpy().astype(bool)
        bridged = present.copy()
        for s, e in _runs(~present & tracked):
            if e - s <= max_gap and s > 0 and e < len(present) and present[s - 1] and present[e]:
                bridged[s:e] = True
        intensity = frames.get(f"{au}_r")
        for s, e in _runs(bridged):
            if e - s < min_len:
                continue
            seg = frames.iloc[s:e]
            rows.append({
                "au": au,
                "start_s": seg["window_time_s"].iloc[0],
                "end_s": seg["window_time_s"].iloc[-1] + 1 / fps,
                "duration_s": (e - s) / fps,
                # An event belongs to the state it started in
                "state": seg["state"].iloc[0],
                "crosses_play_pause": seg["state"].nunique() > 1,
                "performance_start_s": seg["performance_time_s"].iloc[0],
                "performance_end_s": seg["performance_time_s"].iloc[-1],
                "front_gopro_start_s": seg["front_gopro_time_s"].iloc[0],
                "side_gopro_start_s": seg["side_gopro_time_s"].iloc[0],
                "peak_intensity": intensity.iloc[s:e].max() if intensity is not None else np.nan,
                "mean_intensity": intensity.iloc[s:e].mean() if intensity is not None else np.nan,
            })
    cols = ["au", "start_s", "end_s", "duration_s", "state", "crosses_play_pause", "performance_start_s", "performance_end_s",
            "front_gopro_start_s", "side_gopro_start_s", "peak_intensity", "mean_intensity"]
    return pd.DataFrame(rows, columns=cols).sort_values(["start_s", "au"], ignore_index=True)


def au_counts(frames: pd.DataFrame, events: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Cumulative totals per AU, overall and split by playing/paused."""
    rows = []
    aus = [c[:4] for c in frames.filter(regex=AU_PRESENCE).columns]
    for state in ["all"] + sorted(s for s in frames["state"].unique() if s in ("playing", "paused")):
        in_state = frames if state == "all" else frames[frames["state"] == state]
        tracked_s = in_state["valid"].sum() / fps
        ev = events if state == "all" else events[events["state"] == state]
        for au in aus:
            e = ev[ev["au"] == au]
            present = in_state[f"{au}_c"]
            intensity = in_state.get(f"{au}_r")
            rows.append({
                "au": au,
                "state": state,
                "tracked_s": round(tracked_s, 2),
                "n_events": len(e),
                "events_per_min": len(e) / (tracked_s / 60) if tracked_s else np.nan,
                "active_s": round((present == 1).sum() / fps, 2),
                "active_fraction": (present == 1).sum() / present.notna().sum() if present.notna().any() else np.nan,
                "mean_event_duration_s": e["duration_s"].mean() if len(e) else np.nan,
                "mean_event_peak": e["peak_intensity"].mean() if len(e) else np.nan,
                "mean_intensity": intensity.mean() if intensity is not None else np.nan,
            })
    return pd.DataFrame(rows)


def cumulative_event_counts(frames: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per-frame running count of events started so far, one column per AU."""
    out = {}
    t = frames["window_time_s"].to_numpy()
    for au in sorted(events["au"].unique()) if len(events) else []:
        starts = np.sort(events.loc[events["au"] == au, "start_s"].to_numpy())
        out[f"{au}_events_cum"] = np.searchsorted(starts, t, side="right")
    return pd.DataFrame(out, index=frames.index)
