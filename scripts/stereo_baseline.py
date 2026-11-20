#!/usr/bin/env python3
"""Measure the distance between the front and side GoPros from checkerboard frames both cameras saw.

Camera 1 is the front GoPro and camera 2 the side GoPro. For every moment where
the checkerboard is fully visible in both cameras, the corners are detected and
the relative pose of the two cameras is solved with cv2.stereoCalibrate
(intrinsics fixed). The baseline is the distance between the two optical
centres (not the tripod centres, which may differ by a few cm). The floor
distance removes the known height difference between the lenses, and the
result is checked against the room geometry (cameras along two walls of a 90°
corner).

Intrinsics: by default the repo's calibration/{front,side}_calibration.json
(fisheye, from osr_instr_calibrate.py) are reused. --solve-intrinsics solves
them from the frames instead. Corners are undistorted to an ideal pinhole image
before the stereo solve, so fisheye and pinhole calibrations are handled alike.

Inputs (pick one):
  --trial DIR                 raw front/side GoPro files in DIR/gopro_footage, paired with the
                              front/side lag from DIR/MVP/sync_offsets(_v2).json; scans the first
                              90 s (the calibration passes) at 3 fps
  --front VIDEO --side VIDEO  two videos; --lag is the front time at side t=0 (0 for the
                              *_synced.mp4 pairs from export_synced_pairs.py)
  --front DIR --side DIR      two folders of frames, paired by file name (or by sort order
                              when no names match)
Videos named *undistorted* or *synced* are taken to be undistort.py output
(--lens undistorted), so their intrinsics are the rectified matrix, not the raw lens.

The checkerboard size is probed on the first frames unless --pattern is given:
each candidate inner-corner pattern (default 6x9, then 7x10) is tried, and the
largest that is detected wins (a smaller pattern can match a sub-block of a
larger board).

Usage:
    python scripts/stereo_baseline.py --trial $env:FLOW_ANALYTICS_DATA_ROOT/P01_20260223_112737
    python scripts/stereo_baseline.py --front P01_front_synced.mp4 --side P01_side_synced.mp4 --out P01_stereo.json
    python scripts/stereo_baseline.py --front frames/cam1 --side frames/cam2 --lens raw --pattern 7 10
Writes stereo_baseline.json (into <trial>/MVP for --trial, else next to the
front input, or --out).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_synced_pairs import load_offsets  # noqa: E402
from osr_instr_calibrate import (  # noqa: E402
    BLUR_THRESHOLD,
    CALIBRATION_SAMPLE_FPS,
    CALIBRATION_SCAN_SECONDS,
    CameraCalibrator,
)
from undistort import (  # noqa: E402
    find_camera_video,
    get_ffmpeg,
    load_calibration,
    probe_video,
    rectified_camera_matrix,
)

REPO_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"
CAMERAS = ("front", "side")  # camera 1, camera 2
CANDIDATE_PATTERNS = [(6, 9), (7, 10)]  # inner corners (columns, rows); the board is portrait
SQUARE_MM = 25.0
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MIN_STEREO_PAIRS = 5
MAX_STEREO_RMS_PX = 0.5
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# --------------------------------------------------------------------------- board detection

def board_object_points(pattern: tuple[int, int], square_mm: float) -> np.ndarray:
    """Board corners in mm, x along the columns, y along the rows, z = 0 (same order as detection)."""
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return obj


def canonical_order(gray: np.ndarray, corners: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    """Reorder detected corners so the same physical corner comes first in every camera.

    OpenCV can start the grid at any of the four board corners. Two rules make it unique:
    the grid must be right-handed in the image (columns run left-to-right when rows run
    top-to-bottom, i.e. the front of the board is seen), and the first square must be the
    dark one. The colour rule needs columns + rows to be odd (true for 6x9 and 7x10),
    otherwise the board looks the same rotated by 180°.
    """
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    x_dir, y_dir = grid[0, 1] - grid[0, 0], grid[1, 0] - grid[0, 0]
    if x_dir[0] * y_dir[1] - x_dir[1] * y_dir[0] < 0:
        grid = grid[:, ::-1]  # mirrored: run the columns the other way

    def shade(r: int, c: int) -> float:
        x, y = grid[r:r + 2, c:c + 2].reshape(-1, 2).mean(axis=0)
        xi = int(np.clip(round(x), 1, gray.shape[1] - 2))
        yi = int(np.clip(round(y), 1, gray.shape[0] - 2))
        return float(gray[yi - 1:yi + 2, xi - 1:xi + 2].mean())

    if shade(0, 0) > shade(0, 1):
        grid = grid[::-1, ::-1]  # rotated by 180°
    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.float32)


def detect_board(
    gray: np.ndarray, pattern: tuple[int, int], blur_threshold: float = 0.0, fast: bool = False
) -> tuple[np.ndarray | None, str]:
    """Return (corners Nx2 in canonical order, "") or (None, reason)."""
    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
    )
    if not found and not fast and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
        )
    if not found:
        return None, "board not found"
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    if blur_threshold > 0:
        x0, y0 = np.maximum(corners.reshape(-1, 2).min(axis=0).astype(int) - 10, 0)
        x1, y1 = corners.reshape(-1, 2).max(axis=0).astype(int) + 10
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0 or cv2.Laplacian(roi, cv2.CV_64F).var() < blur_threshold:
            return None, "blurred"
    if not fast:
        # The search window must stay inside one square: far boards have squares of ~10 px,
        # where a fixed 11x11 half-window would pull corners onto their neighbours.
        cols = pattern[0]
        grid = corners.reshape(-1, 2)
        spacing = min(np.linalg.norm(np.diff(grid.reshape(-1, cols, 2), axis=1), axis=2).min(),
                      np.linalg.norm(grid[cols:] - grid[:-cols], axis=1).min())
        half = int(np.clip(spacing * 0.4, 2, 11))
        corners = cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), SUBPIX_CRITERIA)
    return canonical_order(gray, corners.reshape(-1, 2), pattern), ""


def probe_pattern(
    pairs, candidates: list[tuple[int, int]], max_frames: int, enough: int = 3
) -> tuple[tuple[int, int] | None, dict[str, int]]:
    """Try each candidate pattern on the first frames; return the largest that matched and the hit counts."""
    hits = {f"{c}x{r}": 0 for c, r in candidates}
    for i, (_, front, side) in enumerate(pairs):
        if i >= max_frames:
            break
        for c, r in candidates:
            for img in (front, side):
                if img is not None and detect_board(img, (c, r), fast=True)[0] is not None:
                    hits[f"{c}x{r}"] += 1
        if all(n >= enough for n in hits.values()):
            break
    matched = [p for p in candidates if hits[f"{p[0]}x{p[1]}"] > 0]
    return (max(matched, key=lambda p: p[0] * p[1]) if matched else None), hits


# --------------------------------------------------------------------------- frame sources

def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS


def iter_video_gray(path: Path, start_frame: int, step: int, count: int):
    """Yield (frame_index, gray) for frames start_frame, start_frame+step, ... decoded with real ffmpeg."""
    width, height, fps = probe_video(path)
    command = [
        get_ffmpeg(), "-v", "error", "-ss", f"{start_frame / fps:.6f}", "-i", str(path),
        "-map", "0:v:0", "-vf", f"select='not(mod(n\\,{step}))'", "-vsync", "0",
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for k in range(count):
            raw = process.stdout.read(width * height)
            if len(raw) < width * height:
                break
            yield start_frame + k * step, np.frombuffer(raw, np.uint8).reshape(height, width)
    finally:
        process.stdout.close()
        process.kill()
        process.wait()


def read_gray(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def folder_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


@dataclass
class PairSource:
    """Re-iterable source of (label, front_gray, side_gray) at matched instants."""
    front: Path
    side: Path
    lag_s: float = 0.0          # front time at side t=0
    start_s: float = 0.0        # side-timeline start of the scan
    scan_s: float = CALIBRATION_SCAN_SECONDS
    sample_fps: float = CALIBRATION_SAMPLE_FPS
    notes: list[str] = field(default_factory=list)

    def image_size(self, camera: str) -> tuple[int, int]:
        path = self.front if camera == "front" else self.side
        if path.is_dir():
            img = read_gray(folder_images(path)[0])
            return img.shape[1], img.shape[0]
        width, height, _ = probe_video(path)
        return width, height

    def __iter__(self):
        if self.front.is_dir() and self.side.is_dir():
            yield from self._folders()
        elif is_video(self.front) and is_video(self.side):
            yield from self._videos()
        else:
            raise ValueError("--front and --side must both be videos or both be folders of frames")

    def _folders(self):
        front, side = folder_images(self.front), folder_images(self.side)
        by_stem = {p.stem: p for p in side}
        common = [p for p in front if p.stem in by_stem]
        pairs = [(p, by_stem[p.stem]) for p in common] if common else list(zip(front, side))
        for f, s in pairs:
            yield f.name if common else f"#{f.name}|{s.name}", read_gray(f), read_gray(s)

    def _videos(self):
        _, _, fps = probe_video(self.side)
        _, _, front_fps = probe_video(self.front)
        if abs(front_fps - fps) > 0.01:
            raise ValueError(f"front is {front_fps:.3f} fps but side is {fps:.3f} fps")
        side_start = round(self.start_s * fps)
        front_start = side_start + round(self.lag_s * fps)
        if front_start < 0:  # front started later than the scan start: move both forward
            side_start, front_start = side_start - front_start, 0
        step = max(1, round(fps / self.sample_fps))
        count = int(self.scan_s * fps / step)
        frames = zip(
            iter_video_gray(self.front, front_start, step, count),
            iter_video_gray(self.side, side_start, step, count),
        )
        for (_, f), (si, s) in frames:
            yield f"t={si / fps:.2f}s", f, s


def single_camera_frames(path: Path, scan_s: float, sample_fps: float):
    """Frames from one camera only, for --*-intrinsics-frames."""
    if path.is_dir():
        for p in folder_images(path):
            yield p.name, read_gray(p)
        return
    _, _, fps = probe_video(path)
    step = max(1, round(fps / sample_fps))
    for i, gray in iter_video_gray(path, 0, step, int(scan_s * fps / step)):
        yield f"t={i / fps:.2f}s", gray


# --------------------------------------------------------------------------- intrinsics

@dataclass
class Intrinsics:
    model: str            # "fisheye" or "standard"
    K: np.ndarray
    D: np.ndarray
    source: str
    rms_px: float | None = None

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        """Map image points to the ideal pinhole image with the same camera matrix."""
        pts = pts.reshape(-1, 1, 2).astype(np.float64)
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts, self.K, self.D.reshape(4, 1), P=self.K)
        elif np.any(self.D):
            out = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        else:
            out = pts
        return out.reshape(-1, 2).astype(np.float32)


def load_intrinsics(path: Path, lens: str, size: tuple[int, int], alpha: float) -> Intrinsics:
    model, K, D = load_calibration(path)
    if lens == "undistorted":
        new_K = rectified_camera_matrix(model, K, D, size, alpha)
        return Intrinsics("standard", new_K, np.zeros(5), f"{path} (rectified, alpha={alpha})")
    return Intrinsics(model, K, D, str(path))


def solve_intrinsics(name: str, model: str, obj: np.ndarray, views: list[np.ndarray],
                     size: tuple[int, int]) -> Intrinsics:
    if len(views) < 3:
        raise RuntimeError(f"{name}: only {len(views)} board views, too few to solve intrinsics")
    if model == "fisheye":
        solver = CameraCalibrator(name)
        solver.objpoints = [obj.copy() for _ in views]
        solver.imgpoints = [v.reshape(-1, 1, 2).copy() for v in views]
        solver.image_size = size
        K, D = solver.auto_prune_and_calibrate()
        return Intrinsics("fisheye", K, D, f"solved from {len(solver.imgpoints)} frames (fisheye)")
    rms, K, D, _, _ = cv2.calibrateCamera([obj] * len(views), views, size, None, None)
    return Intrinsics("standard", K, D, f"solved from {len(views)} frames (pinhole)", float(rms))


def reprojection_rms(intr: Intrinsics, obj: np.ndarray, views: list[np.ndarray]) -> float:
    """RMS (px, in the ideal pinhole image) of the board fitted to each view with solvePnP."""
    sq, n = 0.0, 0
    for v in views:
        und = intr.undistort(v)
        ok, rvec, tvec = cv2.solvePnP(obj, und, intr.K, None)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, intr.K, None)
        sq += float(np.sum((proj.reshape(-1, 2) - und) ** 2))
        n += len(und)
    return math.sqrt(sq / n) if n else float("nan")


# --------------------------------------------------------------------------- stereo geometry

def stereo_extrinsics(obj: np.ndarray, pts1: list[np.ndarray], pts2: list[np.ndarray],
                      K1: np.ndarray, K2: np.ndarray, size: tuple[int, int]) -> dict:
    """cv2.stereoCalibrate on undistorted (ideal pinhole) points: X2 = R @ X1 + T."""
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        [obj] * len(pts1), pts1, pts2, K1, np.zeros(5), K2, np.zeros(5), size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    return {"rms_px": float(rms), "R": R, "T": T.reshape(3)}


def pnp_relative_pose(obj: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                      K1: np.ndarray, K2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative pose from one pair: compose board->camera1 and board->camera2 (X2 = R @ X1 + T)."""
    poses = []
    for pts, K in ((p1, K1), (p2, K2)):
        ok, rvec, tvec = cv2.solvePnP(obj, pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
        poses.append((cv2.Rodrigues(rvec)[0], tvec.reshape(3)))
    (R1, t1), (R2, t2) = poses
    R = R2 @ R1.T
    return R, t2 - R @ t1


def floor_distance(baseline_cm: float, height_delta_cm: float) -> float:
    if baseline_cm < abs(height_delta_cm):
        return float("nan")
    return math.sqrt(baseline_cm ** 2 - height_delta_cm ** 2)


def room_check(floor_cm: float, wall1_cm: float, wall2_cm: float, max_offset_cm: float) -> dict:
    """Compare the floor distance with the corner geometry.

    Camera 1 sits wall1_cm from the corner along wall A and d1 out from it; camera 2 sits
    wall2_cm along wall B and d2 out: floor^2 = (wall1 - d2)^2 + (wall2 - d1)^2, d1, d2 in
    [0, max_offset].
    """
    def dist(d1: float, d2: float) -> float:
        return math.hypot(wall1_cm - d2, wall2_cm - d1)

    lo, hi = dist(max_offset_cm, max_offset_cm), dist(0.0, 0.0)
    result = {"expected_floor_range_cm": [round(lo, 1), round(hi, 1)], "in_range": lo <= floor_cm <= hi}
    if not math.isfinite(floor_cm):
        result["in_range"] = False
        return result
    # d1 = d2 = d: 2d^2 - 2(w1 + w2)d + (w1^2 + w2^2 - F^2) = 0, smaller root
    s, c = wall1_cm + wall2_cm, wall1_cm ** 2 + wall2_cm ** 2 - floor_cm ** 2
    disc = s ** 2 - 2 * c
    result["equal_offsets_cm"] = round((s - math.sqrt(disc)) / 2, 1) if disc >= 0 else None
    # every (d1, d2) in the box consistent with the floor distance
    feasible = []
    for d1 in np.linspace(0, max_offset_cm, 3001):
        rest = floor_cm ** 2 - (wall2_cm - d1) ** 2
        if rest >= 0:
            d2 = wall1_cm - math.sqrt(rest)
            if 0 <= d2 <= max_offset_cm:
                feasible.append((float(d1), d2))
    if feasible:
        result["d1_range_cm"] = [round(feasible[0][0], 1), round(feasible[-1][0], 1)]
        result["d2_range_cm"] = sorted([round(feasible[0][1], 1), round(feasible[-1][1], 1)])
        result["extreme_solutions_cm"] = [
            {"d1": round(d1, 1), "d2": round(d2, 1)} for d1, d2 in (feasible[0], feasible[-1])
        ]
    else:
        result["d1_range_cm"] = result["d2_range_cm"] = None
    return result


def compute_distances(T: np.ndarray, height_delta_cm: float, unit_to_cm: float) -> dict:
    baseline = float(np.linalg.norm(T)) * unit_to_cm
    return {"baseline_3d_cm": round(baseline, 2),
            "floor_distance_cm": round(floor_distance(baseline, height_delta_cm), 2)}


# --------------------------------------------------------------------------- pipeline

def collect_detections(pairs, pattern, blur_threshold, verbose):
    """Detect the board in every pair. Returns per-camera views, matched pairs and a rejection log."""
    views = {cam: [] for cam in CAMERAS}
    paired: list[tuple[str, np.ndarray, np.ndarray]] = []
    rejected, n_pairs = [], 0
    for label, front, side in pairs:
        n_pairs += 1
        found = {}
        for cam, img in zip(CAMERAS, (front, side)):
            if img is None:
                found[cam], reason = None, "unreadable"
            else:
                found[cam], reason = detect_board(img, pattern, blur_threshold)
            if found[cam] is not None:
                views[cam].append(found[cam])
            elif reason != "board not found" or verbose:
                print(f"    skip {cam} {label}: {reason}", flush=True)
        if found["front"] is not None and found["side"] is not None:
            paired.append((label, found["front"], found["side"]))
        elif any(v is not None for v in found.values()):
            seen = "front" if found["front"] is not None else "side"
            rejected.append({"frame": label, "reason": f"board only in {seen}"})
            print(f"    skip pair {label}: board only in {seen}", flush=True)
        else:
            rejected.append({"frame": label, "reason": "board in neither camera"})
    return views, paired, rejected, n_pairs


def estimate(obj, intr: dict[str, Intrinsics], paired, size, height_delta_cm, unit_to_cm) -> dict:
    """Stereo solve on matched pairs plus the per-pair solvePnP cross-check."""
    p1 = [intr["front"].undistort(f) for _, f, _ in paired]
    p2 = [intr["side"].undistort(s) for _, _, s in paired]
    K1, K2 = intr["front"].K, intr["side"].K
    out: dict = {"pairs_used": len(paired)}
    stereo = stereo_extrinsics(obj, p1, p2, K1, K2, size)
    out["stereo"] = {
        "rms_px": round(stereo["rms_px"], 4),
        "R": stereo["R"].tolist(),
        "T_cm": (stereo["T"] * unit_to_cm).round(3).tolist(),
        **compute_distances(stereo["T"], height_delta_cm, unit_to_cm),
    }
    per_pair = []
    for (label, _, _), a, b in zip(paired, p1, p2):
        R, T = pnp_relative_pose(obj, a, b, K1, K2)
        per_pair.append({"frame": label, **compute_distances(T, height_delta_cm, unit_to_cm),
                         "R": R.tolist(), "T_cm": (T * unit_to_cm).round(3).tolist()})
    baselines = np.array([p["baseline_3d_cm"] for p in per_pair])
    best = per_pair[int(np.argmin(np.abs(baselines - np.median(baselines))))]
    out["pnp"] = {
        "baseline_3d_cm_median": round(float(np.median(baselines)), 2),
        "baseline_3d_cm_spread": [round(float(baselines.min()), 2), round(float(baselines.max()), 2)],
        "floor_distance_cm_median": round(floor_distance(float(np.median(baselines)), height_delta_cm), 2),
        "median_pair": best,
        "per_pair": per_pair,
    }
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    src = p.add_argument_group("input")
    src.add_argument("--trial", type=Path, help="trial folder (raw GoPro footage + MVP sync offsets)")
    src.add_argument("--front", type=Path, help="camera 1 video or folder of frames")
    src.add_argument("--side", type=Path, help="camera 2 video or folder of frames")
    src.add_argument("--lag", type=float, default=0.0, help="front time (s) at side t=0, for --front/--side videos")
    src.add_argument("--start", type=float, default=0.0, help="side-timeline second where the scan starts")
    src.add_argument("--scan-seconds", type=float, default=CALIBRATION_SCAN_SECONDS)
    src.add_argument("--sample-fps", type=float, default=CALIBRATION_SAMPLE_FPS)
    src.add_argument("--lens", choices=("auto", "raw", "undistorted"), default="auto",
                     help="raw lens footage, or undistort.py output (auto: by file name)")
    src.add_argument("--alpha", type=float, default=1.0, help="alpha/balance used by undistort.py (mvp prepare: 1.0)")

    board = p.add_argument_group("checkerboard")
    board.add_argument("--pattern", type=int, nargs=2, metavar=("COLS", "ROWS"),
                       help="inner corners (columns, rows); skips the probe")
    board.add_argument("--square-mm", type=float, default=SQUARE_MM)
    board.add_argument("--probe-frames", type=int, default=60, help="frames (pairs) used to probe the pattern")
    board.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD,
                       help="reject boards whose Laplacian variance is lower (0 disables)")

    cal = p.add_argument_group("intrinsics")
    cal.add_argument("--calibration-dir", type=Path, default=REPO_CALIBRATION_DIR,
                     help="folder with front_calibration.json / side_calibration.json to reuse")
    cal.add_argument("--solve-intrinsics", action="store_true", help="solve intrinsics from the frames instead")
    cal.add_argument("--model", choices=("fisheye", "standard"), default=None,
                     help="lens model for --solve-intrinsics (default: fisheye for raw, standard for undistorted)")
    cal.add_argument("--front-intrinsics-frames", type=Path, help="extra front-only video/folder for intrinsics")
    cal.add_argument("--side-intrinsics-frames", type=Path, help="extra side-only video/folder for intrinsics")

    geo = p.add_argument_group("room geometry")
    geo.add_argument("--front-height-cm", type=float, default=104.0)
    geo.add_argument("--side-height-cm", type=float, default=117.0)
    geo.add_argument("--front-wall-cm", type=float, default=380.0, help="camera 1 distance from the corner along wall A")
    geo.add_argument("--side-wall-cm", type=float, default=336.0, help="camera 2 distance from the corner along wall B")
    geo.add_argument("--max-wall-offset-cm", type=float, default=30.0)

    p.add_argument("--out", type=Path, help="JSON report path")
    p.add_argument("-v", "--verbose", action="store_true", help="log every frame without a board")
    args = p.parse_args(argv)
    if bool(args.trial) == bool(args.front or args.side) or (not args.trial and not (args.front and args.side)):
        p.error("give either --trial, or both --front and --side")
    return args


def build_source(args) -> tuple[PairSource, Path]:
    if args.trial:
        gopro = args.trial / "gopro_footage"
        front, side = find_camera_video(gopro, "front"), find_camera_video(gopro, "side")
        offsets, name = load_offsets(args.trial)
        lag = offsets["front_offset_sec"] - offsets["side_offset_sec"]
        source = PairSource(front, side, lag, args.start, args.scan_seconds, args.sample_fps,
                            [f"lag {lag:+.3f} s from MVP/{name}"])
        return source, args.out or args.trial / "MVP" / "stereo_baseline.json"
    source = PairSource(args.front, args.side, args.lag, args.start, args.scan_seconds, args.sample_fps,
                        [f"lag {args.lag:+.3f} s (--lag)"] if is_video(args.front) else [])
    default_out = (args.front if args.front.is_dir() else args.front.parent) / "stereo_baseline.json"
    return source, args.out or default_out


def main(argv=None) -> int:
    args = parse_args(argv)
    source, out_path = build_source(args)
    lens = args.lens
    if lens == "auto":
        names = f"{source.front.name} {source.side.name}".lower()
        lens = "undistorted" if ("undistorted" in names or "synced" in names) else "raw"
    print(f"Camera 1 (front): {source.front}\nCamera 2 (side):  {source.side}\nLens: {lens}")
    for note in source.notes:
        print(f"Pairing: {note}")
    size = source.image_size("front")
    if source.image_size("side") != size:
        print(f"WARNING: image sizes differ (front {size}, side {source.image_size('side')})")

    report: dict = {"front": str(source.front), "side": str(source.side), "lens": lens,
                    "pairing": source.notes, "warnings": []}
    warn = report["warnings"].append

    # 1. checkerboard pattern
    if args.pattern:
        pattern = tuple(args.pattern)
        report["pattern_probe"] = "skipped (--pattern)"
    else:
        print(f"Probing checkerboard pattern on up to {args.probe_frames} frames: "
              + ", ".join(f"{c}x{r}" for c, r in CANDIDATE_PATTERNS))
        pattern, hits = probe_pattern(source, CANDIDATE_PATTERNS, args.probe_frames)
        print(f"    detections: {hits}")
        report["pattern_probe"] = hits
        if pattern is None:
            print("ERROR: no candidate pattern was detected; check the board or pass --pattern COLS ROWS "
                  "(osr_instr_calibrate.py uses 10x7 inner corners)", file=sys.stderr)
            return 2
    print(f"Pattern: {pattern[0]}x{pattern[1]} inner corners "
          f"({pattern[0] + 1}x{pattern[1] + 1} squares of {args.square_mm} mm)")
    if (pattern[0] + pattern[1]) % 2 == 0:
        warn("columns + rows is even: the board looks the same rotated 180°, so its orientation "
             "may not match between cameras")
    report["pattern_inner_corners"] = list(pattern)
    report["square_mm"] = args.square_mm
    obj = board_object_points(pattern, args.square_mm)

    # 2. corners in every pair
    print("Detecting corners...")
    views, paired, rejected, n_pairs = collect_detections(source, pattern, args.blur_threshold, args.verbose)
    report["frames"] = {"pairs_scanned": n_pairs, "pairs_used": len(paired), "pairs_rejected": len(rejected),
                        "front_views": len(views["front"]), "side_views": len(views["side"]),
                        "rejected": [r for r in rejected if r["reason"] != "board in neither camera"],
                        "board_in_neither": sum(r["reason"] == "board in neither camera" for r in rejected)}
    print(f"    {n_pairs} pairs scanned: {len(paired)} with the board in both cameras; "
          f"front saw it in {len(views['front'])}, side in {len(views['side'])}")
    if not paired:
        print("ERROR: no frame has the board fully visible in both cameras", file=sys.stderr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    # 3. intrinsics
    intr: dict[str, Intrinsics] = {}
    for cam, extra in (("front", args.front_intrinsics_frames), ("side", args.side_intrinsics_frames)):
        if args.solve_intrinsics:
            cam_views = list(views[cam])
            if extra:
                for _, gray in single_camera_frames(extra, args.scan_seconds, args.sample_fps):
                    corners, _ = detect_board(gray, pattern, args.blur_threshold)
                    if corners is not None:
                        cam_views.append(corners)
            model = args.model or ("fisheye" if lens == "raw" else "standard")
            intr[cam] = solve_intrinsics(cam.upper(), model, obj, cam_views, size)
        else:
            path = args.calibration_dir / f"{cam}_calibration.json"
            intr[cam] = load_intrinsics(path, lens, size, args.alpha)
        intr[cam].rms_px = reprojection_rms(intr[cam], obj, views[cam])
        print(f"Intrinsics {cam}: {intr[cam].source}; RMS reprojection {intr[cam].rms_px:.3f} px "
              f"over {len(views[cam])} views")
    report["intrinsics"] = {cam: {"model": i.model, "source": i.source, "rms_px": round(i.rms_px, 4),
                                  "camera_matrix": i.K.tolist(), "dist_coeffs": np.ravel(i.D).tolist()}
                            for cam, i in intr.items()}

    # 4. stereo + distances
    height_delta = args.side_height_cm - args.front_height_cm
    result = estimate(obj, intr, paired, size, height_delta, unit_to_cm=0.1)
    report["height_delta_cm"] = height_delta
    report.update(result)
    stereo, pnp = result["stereo"], result["pnp"]
    if stereo["rms_px"] > MAX_STEREO_RMS_PX:
        warn(f"stereo RMS {stereo['rms_px']:.3f} px is above {MAX_STEREO_RMS_PX} px")
    if len(paired) < MIN_STEREO_PAIRS:
        warn(f"only {len(paired)} matched pair(s) (< {MIN_STEREO_PAIRS}): both estimates are less accurate; "
             "the solvePnP result from a single pair is reported alongside")

    # 5. room sanity check
    check = room_check(stereo["floor_distance_cm"], args.front_wall_cm, args.side_wall_cm, args.max_wall_offset_cm)
    report["sanity_check"] = check
    if not check["in_range"]:
        warn(f"floor distance {stereo['floor_distance_cm']} cm is outside the expected "
             f"{check['expected_floor_range_cm'][0]}-{check['expected_floor_range_cm'][1]} cm")
    report["note"] = "Distances are between the optical centres, which may be a few cm from the tripod centres."

    np.set_printoptions(precision=5, suppress=True)
    print("\n=== Stereo baseline (camera 1 = front, camera 2 = side; X_side = R X_front + T) ===")
    print(f"Pairs used: {len(paired)}   rejected: {len(rejected)} "
          f"({report['frames']['board_in_neither']} with the board in neither camera)")
    print(f"RMS reprojection: front {intr['front'].rms_px:.3f} px, side {intr['side'].rms_px:.3f} px, "
          f"stereo {stereo['rms_px']:.3f} px")
    print(f"R =\n{np.array(stereo['R'])}")
    print(f"T = {np.array(stereo['T_cm'])} cm")
    print(f"baseline_3d_cm    = {stereo['baseline_3d_cm']:.1f}   (solvePnP median {pnp['baseline_3d_cm_median']:.1f}, "
          f"range {pnp['baseline_3d_cm_spread'][0]:.1f}-{pnp['baseline_3d_cm_spread'][1]:.1f})")
    print(f"floor_distance_cm = {stereo['floor_distance_cm']:.1f}   (height delta {height_delta:.1f} cm)")
    rng = check["expected_floor_range_cm"]
    line = f"Sanity: expected {rng[0]:.1f}-{rng[1]:.1f} cm -> {'OK' if check['in_range'] else 'OUT OF RANGE'}"
    if check.get("equal_offsets_cm") is not None:
        d = check["equal_offsets_cm"]
        line += f"; consistent with d1 = d2 = {d:.1f} cm"
        if not 0 <= d <= args.max_wall_offset_cm:
            line += f" (outside 0-{args.max_wall_offset_cm:g} cm)"
    if check.get("d1_range_cm"):
        line += f"; d1 {check['d1_range_cm'][0]}-{check['d1_range_cm'][1]} cm, d2 {check['d2_range_cm'][0]}-{check['d2_range_cm'][1]} cm"
    print(line)
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"Note: {report['note']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
