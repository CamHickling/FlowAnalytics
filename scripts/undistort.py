#!/usr/bin/env python3
"""Undistort GoPro videos using calibration JSON from osr_instr_calibrate.py."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


CAMERA_NAMES = ("front", "side")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


def load_calibration(calibration_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)

    try:
        camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(calibration["dist_coeffs"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Missing calibration field {exc} in {calibration_path}") from exc

    model = calibration.get("model", "standard")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Camera matrix must be 3x3: {calibration_path}")
    return model, camera_matrix, dist_coeffs


def get_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("FFmpeg is required to preserve source audio") from exc


def mux_source_audio(video_only: Path, source: Path, output: Path) -> None:
    command = [
        get_ffmpeg(),
        "-y",
        "-i", str(video_only),
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio mux failed:\n{result.stderr.strip()}")


def probe_video(input_path: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video dimensions: {input_path}")
    return width, height, fps


def iter_frames_ffmpeg(input_path: Path, width: int, height: int):
    """Decode every video frame with real ffmpeg rather than cv2.VideoCapture.

    GoPro files interleave video/audio with tmcd/gpmd/fdsc metadata tracks that
    OpenCV's bundled ffmpeg backend frequently stumbles on mid-grab ("packet read
    max attempts exceeded"), silently truncating playback after only a handful of
    frames. A direct ffmpeg subprocess decoding only the video stream (-map 0:v:0)
    does not have this problem.
    """
    frame_size = width * height * 3
    command = [
        get_ffmpeg(),
        "-v", "error",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = process.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    finally:
        process.stdout.close()
        process.wait()


def undistort_video(
    input_path: Path,
    output_path: Path,
    calibration_path: Path,
    alpha: float = 1.0,
) -> int:
    """Undistort one video and return the number of processed frames."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    model, camera_matrix, dist_coeffs = load_calibration(calibration_path)
    width, height, fps = probe_video(input_path)

    map1 = map2 = None
    new_matrix = None
    if model == "fisheye":
        dist_coeffs = dist_coeffs.reshape(4, 1)
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (width, height), np.eye(3), balance=alpha
        )
        # estimateNewCameraMatrixForUndistortRectify can silently return a
        # degenerate matrix (near-zero focal length) for calibrations with
        # poorly-conditioned distortion coefficients, which maps the whole frame
        # out of bounds and produces an all-black video. Fall back to the
        # calibrated camera matrix itself, which is always well-scaled.
        min_focal = min(new_matrix[0, 0], new_matrix[1, 1])
        if not np.isfinite(min_focal) or min_focal < 1.0:
            new_matrix = camera_matrix.copy()
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_matrix, (width, height), cv2.CV_16SC2
        )
    else:
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (width, height), alpha, (width, height)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(suffix=".mp4", dir=output_path.parent)
    os.close(fd)
    temporary_video = Path(temporary_name)
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError(f"Could not create temporary video: {temporary_video}")

    frame_count = 0
    try:
        for frame in iter_frames_ffmpeg(input_path, width, height):
            if map1 is not None:
                corrected = cv2.remap(
                    frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )
            else:
                corrected = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_matrix)
            writer.write(corrected)
            frame_count += 1
    finally:
        writer.release()

    try:
        if frame_count == 0:
            raise RuntimeError(f"No frames decoded from: {input_path}")
        mux_source_audio(temporary_video, input_path, output_path)
    finally:
        temporary_video.unlink(missing_ok=True)

    return frame_count


def find_camera_video(gopro_dir: Path, camera: str) -> Path:
    # Prefer a '_COMBINED' file when present: the GoPro splits recordings
    # into ~4GB chapters, and some trials only have the first chapter as the
    # plain '<camera>.MP4' name, with the full merged recording saved
    # separately as '<camera>_COMBINED.MP4'. Excluding "combined" here used
    # to silently pick the truncated first chapter for every such trial.
    candidates = sorted(
        path
        for path in gopro_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and camera in path.stem.lower()
        and "sync" not in path.stem.lower()
        and "undistorted" not in path.stem.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find a {camera} video in {gopro_dir}")
    combined = [path for path in candidates if "combined" in path.stem.lower()]
    return combined[0] if combined else candidates[0]


def process_trial(
    trial_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
    alpha: float,
) -> None:
    gopro_dir = trial_dir / "gopro_footage"
    if not gopro_dir.is_dir():
        raise FileNotFoundError(f"GoPro folder not found: {gopro_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for camera in CAMERA_NAMES:
        source = find_camera_video(gopro_dir, camera)
        calibration = calibration_dir / f"{camera}_calibration.json"
        output = output_dir / f"{source.stem}_undistorted.mp4"
        print(f"{camera}: {source.name} -> {output}")
        frames = undistort_video(source, output, calibration, alpha=alpha)
        print(f"  processed {frames} frames")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Undistort GoPro footage using standard OpenCV calibration"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="One GoPro video to process")
    source.add_argument("--trial", type=Path, help="Trial folder containing gopro_footage")
    parser.add_argument("--output", type=Path, help="Output path for --input")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path.cwd(),
        help="Folder containing front_calibration.json and side_calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder for --trial; defaults to trial/MVP/generated",
    )
    parser.add_argument("--camera", choices=CAMERA_NAMES, help="Required with --input")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="OpenCV border setting: 0 crops more, 1 preserves more of the frame",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")

    if args.input:
        if not args.output or not args.camera:
            parser.error("--input requires --output and --camera")
        calibration = args.calibration_dir / f"{args.camera}_calibration.json"
        frames = undistort_video(args.input, args.output, calibration, alpha=args.alpha)
        print(f"Created {args.output} from {frames} frames")
        return

    output_dir = args.output_dir or args.trial / "MVP" / "generated"
    process_trial(args.trial, args.calibration_dir, output_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
