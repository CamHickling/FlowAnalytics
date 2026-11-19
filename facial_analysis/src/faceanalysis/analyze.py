"""Quality-filter OpenFace output and summarise it per video."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config

AU_INTENSITY = r"^AU\d{2}_r$"  # 0-5 intensity (17 AUs)
AU_PRESENCE = r"^AU\d{2}_c$"   # 0/1 presence (18 AUs, incl. AU28)
POSE = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
GAZE = ["gaze_angle_x", "gaze_angle_y"]


def load_openface_csv(path: Path) -> pd.DataFrame:
    # OpenFace pads column names and values with spaces
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def valid_mask(df: pd.DataFrame, min_confidence: float) -> pd.Series:
    return (df["success"] == 1) & (df["confidence"] >= min_confidence)


def clean_frames(df: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Keep the analysis columns and blank out (NaN) frames that failed quality checks.

    Invalid frames are kept as NaN rows rather than dropped so the timeline stays
    intact for time-series work.
    """
    feature_cols = list(df.filter(regex=f"{AU_INTENSITY}|{AU_PRESENCE}").columns)
    feature_cols += [c for c in POSE + GAZE if c in df.columns]
    out = df[["frame", "timestamp", "confidence", "success"] + feature_cols].copy()
    out["valid"] = valid_mask(df, min_confidence)
    out.loc[~out["valid"], feature_cols] = np.nan
    return out


def summarize(clean: pd.DataFrame) -> dict:
    """One row of per-video statistics computed over valid frames only."""
    valid = clean[clean["valid"]]
    duration = clean["timestamp"].iloc[-1] - clean["timestamp"].iloc[0] if len(clean) > 1 else 0.0
    row = {
        "n_frames": len(clean),
        "duration_s": round(float(duration), 2),
        "valid_frames": len(valid),
        "valid_fraction": round(len(valid) / len(clean), 4) if len(clean) else 0.0,
        "mean_confidence": round(float(clean["confidence"].mean()), 4),
    }
    for col in clean.filter(regex=AU_INTENSITY).columns:
        row[f"{col}_mean"] = valid[col].mean()
        row[f"{col}_sd"] = valid[col].std()
    for col in clean.filter(regex=AU_PRESENCE).columns:
        row[f"{col}_rate"] = valid[col].mean()  # fraction of valid frames where the AU is present
    for col in POSE + GAZE:
        if col in clean.columns:
            row[f"{col}_mean"] = valid[col].mean()
            row[f"{col}_sd"] = valid[col].std()
    return row


def parse_metadata(relative_path: str, cfg: Config) -> dict:
    """Named groups of the configured regex, matched against the video's relative path."""
    if not cfg.metadata_pattern:
        return {}
    m = re.search(cfg.metadata_pattern, relative_path)
    if not m:
        print(f"Warning: metadata pattern did not match {relative_path}")
        return {}
    meta = {k: v or cfg.metadata_defaults.get(k, v) for k, v in m.groupdict().items()}
    for flag, conditions in cfg.metadata_flags.items():
        meta[flag] = all(meta.get(k) == v for k, v in conditions.items())
    return meta


def read_marker(marker: Path) -> dict:
    # Markers written before metadata support are empty
    text = marker.read_text().strip()
    return json.loads(text) if text else {}


def analyze_all(cfg: Config) -> pd.DataFrame:
    cfg.clean_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(cfg.openface_dir.glob("*.done"))
    if not done:
        raise FileNotFoundError(f"No finished OpenFace outputs in {cfg.openface_dir}. Run `extract` first.")

    rows = []
    for marker in tqdm(done, unit="video"):
        video_id = marker.stem
        source = read_marker(marker)
        clean = clean_frames(load_openface_csv(cfg.openface_dir / f"{video_id}.csv"), cfg.min_confidence)
        clean.to_csv(cfg.clean_dir / f"{video_id}.csv", index=False)
        stats = summarize(clean)
        rows.append({
            "video_id": video_id,
            **parse_metadata(source.get("relative_path", ""), cfg),
            "short_video": stats["duration_s"] < cfg.min_duration_s,
            "low_tracking": stats["valid_fraction"] < 0.5,
            **stats,
            "source_path": source.get("source_path", ""),
        })

    summary = pd.DataFrame(rows)
    out = cfg.output_dir / "summary.csv"
    summary.to_csv(out, index=False, float_format="%.4f")
    low = summary[summary["low_tracking"]]
    print(f"Wrote {out} ({len(summary)} videos).")
    if summary["short_video"].any():
        print(f"Note: {summary['short_video'].sum()} video(s) shorter than {cfg.min_duration_s:g}s are flagged short_video.")
    if len(low):
        print(f"Warning: {len(low)} video(s) have <50% valid frames - check face visibility/size:")
        for _, r in low.iterrows():
            print(f"  {r.video_id}: {r.valid_fraction:.0%} valid")
    return summary
