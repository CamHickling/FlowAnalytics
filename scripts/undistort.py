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


def load_calibration(calibration_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)

    try:
        camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(calibration["dist_coeffs"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Missing calibration field {exc} in {calibration_path}") from exc

    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Camera matrix must be 3x3: {calibration_path}")
    return camera_matrix, dist_coeffs


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

    camera_matrix, dist_coeffs = load_calibration(calibration_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions: {input_path}")

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
        capture.release()
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError(f"Could not create temporary video: {temporary_video}")

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            corrected = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_matrix)
            writer.write(corrected)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    try:
        if frame_count == 0:
            raise RuntimeError(f"No frames decoded from: {input_path}")
        mux_source_audio(temporary_video, input_path, output_path)
    finally:
        temporary_video.unlink(missing_ok=True)

    return frame_count


def find_camera_video(gopro_dir: Path, camera: str) -> Path:
    candidates = sorted(
        path
        for path in gopro_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and camera in path.stem.lower()
        and "sync" not in path.stem.lower()
        and "combined" not in path.stem.lower()
        and "undistorted" not in path.stem.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find a {camera} video in {gopro_dir}")
    return candidates[0]


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
