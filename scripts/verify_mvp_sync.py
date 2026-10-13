"""Check a rendered MVP/review.mp4 against its timing data.

For a set of review times (during play, during pauses, near the end) it
compares the front and side panels with the undistorted GoPro frames the pause
timeline predicts, searching +-2 s to confirm the best match is at 0. It also
cross-correlates the narration track with the source WAV at the loudest
stretches, which should all land on the expected audio offset.

Usage: python verify_mvp_sync.py P20 P01 ...   (trial folder prefixes)
"""

import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

import mvp

DATA_ROOT = Path("D:/FlowAnalytics/Iris_Recorded_Taekwondo_Data")
FFMPEG = "C:/Users/BarlabPRIME/anaconda3/envs/video/Library/bin/ffmpeg.exe"


def _sig(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54)).astype(float)
    return ((g - g.mean()) / (g.std() + 1e-6)).ravel()


def _frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
    ok, f = cap.read()
    return f if ok else None


def _audio(path, start=None, duration=None, sr=8000):
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.int16).astype(float)


def check_panels(trial, assets, timeline, review_mp4):
    segs = timeline["segments"]
    offsets = mvp.load_sync_offsets(trial)
    prepared = json.loads((trial / "MVP" / "generated" / "prepared_assets.json").read_text())
    W, H, M, G = mvp.LAYOUT_VIDEO_W, mvp.LAYOUT_VIDEO_H, mvp.LAYOUT_MARGIN, mvp.LAYOUT_GAP
    panels = {"front": (M, M), "side": (M + W + G, M)}
    caps = {cam: cv2.VideoCapture(prepared[cam]["output"]) for cam in panels}
    lengths = {cam: caps[cam].get(7) / caps[cam].get(5) for cam in panels}
    rev = cv2.VideoCapture(str(review_mp4))

    R = timeline["review_duration_sec"]
    times = [min(20.0, R / 4), R / 2]
    holds = [s for s in segs if s["type"] == "hold" and s["review_end_sec"] - s["review_start_sec"] > 2]
    if holds:
        times.append((holds[len(holds) // 2]["review_start_sec"] + holds[len(holds) // 2]["review_end_sec"]) / 2)
    times.append(R - 2.0)
    ok_all = True
    for t in sorted(times):
        f = _frame(rev, t)
        if f is None:
            print(f"   t={t:7.1f}s  could not read review.mp4")
            ok_all = False
            continue
        pos = mvp.overhead_position_at(t, segs)
        state = next((s["type"] for s in segs if s["review_start_sec"] <= t <= s["review_end_sec"]), "?")
        parts = []
        for cam, (x, y) in panels.items():
            target = _sig(f[y + 10:y + H - 10, x + 10:x + W - 10])
            expected = pos + offsets[f"{cam}_offset_sec"]
            clamped = min(max(expected, 0.0), lengths[cam] - 2 / 30)
            scores = {}
            for d in np.round(np.arange(-2, 2.01, 0.2), 1):
                g = _frame(caps[cam], clamped + d)
                if g is not None:
                    scores[d] = float(np.dot(target, _sig(cv2.resize(g, (W, H))[10:H - 10, 10:W - 10])) / len(target))
            best = max(scores, key=scores.get)
            live = 0 <= expected <= lengths[cam]
            good = abs(best) <= 0.2 or scores.get(0.0, 0) > 0.97
            ok_all &= good
            parts.append(f"{cam} {'live' if live else 'frozen'} best {best:+.1f}s r={scores[best]:.3f} {'OK' if good else 'MISMATCH'}")
        print(f"   t={t:7.1f}s [{state:4}] perf {pos:6.2f}s  " + "  ".join(parts))
    return ok_all


def check_audio(assets, review_mp4, audio_seek, sr=8000):
    mix = _audio(review_mp4, sr=sr)
    src = _audio(assets.narration_audio, sr=sr)
    L = 3 * sr
    energy = sorted(((np.std(mix[i:i + L]), i) for i in range(0, len(mix) - L, L)), reverse=True)[:5]
    n = len(src) + L
    S = np.fft.rfft(src - src.mean(), n)
    offsets = []
    for _, i in energy:
        seg = mix[i:i + L] - mix[i:i + L].mean()
        cc = np.fft.irfft(S * np.conj(np.fft.rfft(seg, n)), n)[:len(src) - L]
        offsets.append(int(np.argmax(cc)) / sr - i / sr)
    good = all(abs(o - audio_seek) < 0.05 for o in offsets)
    print(f"   narration audio offsets {[round(o, 3) for o in offsets]} (expected {audio_seek:.3f}) {'OK' if good else 'MISMATCH'}")
    return good


def main():
    results = {}
    for prefix in sys.argv[1:]:
        trial = next(p for p in sorted(DATA_ROOT.iterdir()) if p.name.startswith(prefix))
        assets = mvp.discover_trial(trial, None)
        timeline = mvp.build_pause_timeline(assets)
        review_mp4 = trial / "MVP" / "review.mp4"
        cap = cv2.VideoCapture(str(review_mp4))
        length = cap.get(7) / cap.get(5)
        R = timeline["review_duration_sec"]
        print(f"{trial.name}: review.mp4 {length:.1f}s, review timeline {R:.1f}s {'OK' if abs(length - R) < 0.5 else 'LENGTH MISMATCH'}")
        _, audio_offset = mvp.review_media_offsets(assets)
        ok = abs(length - R) < 0.5
        ok &= check_panels(trial, assets, timeline, review_mp4)
        ok &= check_audio(assets, review_mp4, audio_offset)
        results[trial.name] = ok
    print("\nSummary:", {k: "OK" if v else "CHECK" for k, v in results.items()})


if __name__ == "__main__":
    main()
