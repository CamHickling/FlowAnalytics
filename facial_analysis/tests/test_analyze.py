from pathlib import Path

import numpy as np
import pandas as pd

from faceanalysis.analyze import clean_frames, summarize
from faceanalysis.extract import make_video_id


def fake_openface(n=10):
    return pd.DataFrame({
        "frame": range(1, n + 1),
        "face_id": 0,
        "timestamp": np.arange(n) / 30,
        "confidence": [0.98] * (n - 3) + [0.5, 0.9, 0.9],
        "success": [1] * (n - 2) + [0, 0],
        "pose_Rx": 0.1,
        "AU12_r": np.linspace(0, 2, n),
        "AU12_c": [1] * n,
    })


def test_clean_frames_blanks_low_quality_rows_but_keeps_timeline():
    clean = clean_frames(fake_openface(), min_confidence=0.8)
    assert len(clean) == 10
    assert clean["valid"].sum() == 7  # 1 low-confidence + 2 tracking failures removed
    assert clean.loc[~clean["valid"], "AU12_r"].isna().all()
    assert clean["timestamp"].notna().all()


def test_summarize_uses_valid_frames_only():
    clean = clean_frames(fake_openface(), min_confidence=0.8)
    row = summarize(clean)
    assert row["valid_frames"] == 7
    assert row["valid_fraction"] == 0.7
    assert np.isclose(row["AU12_r_mean"], np.linspace(0, 2, 10)[:7].mean())
    assert row["AU12_c_rate"] == 1.0


def test_video_id_is_unique_across_cameras():
    root = Path("D:/footage")
    a = make_video_id(root / "front go pro offload 1/DCIM/100GOPRO/GH017313.MP4", root)
    b = make_video_id(root / "side go pro offload 1/100GOPRO/GH017313.MP4", root)
    assert a == "front_go_pro_offload_1__DCIM__100GOPRO__GH017313"
    assert a != b


def test_metadata_from_face_video_paths():
    from faceanalysis.analyze import parse_metadata
    from faceanalysis.config import load_config

    cfg = load_config(Path(__file__).parents[1] / "config_face_videos.yaml")
    cases = {
        "P01_20260223_112737/review/P01_face_narration.mp4": ("P01", "main", "narration", True),
        "P02B_20260223_134013/scoring/P02B_face_scoring.mp4": ("P02", "B", "scoring", False),
        "P04__20260224_130709/scoring/P04_face_scoring.mp4": ("P04", "main", "scoring", False),
        "P29_20260318_121454/review/29_face_narration.mp4": ("P29", "main", "narration", True),
    }
    for path, (participant, session, condition, confounded) in cases.items():
        meta = parse_metadata(path, cfg)
        assert (meta["participant"], meta["session"], meta["condition"],
                meta["mouth_aus_speech_confounded"]) == (participant, session, condition, confounded)
