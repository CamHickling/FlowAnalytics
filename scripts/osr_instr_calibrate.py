#!/usr/bin/env python3
"""
calibrate.py

Unified IRIS Calibration Suite.
Handles live hardware pre-flight connectivity checks and robust global 
intrinsic matrix generation for GoPro fisheye correction.
"""

import sys
import threading
import time
import json
import math
import argparse
from typing import Optional
from pathlib import Path

import numpy as np
import cv2

# Ensure relative imports work for the hardware check modules
try:
    from .camera import Camera, CameraConfig
    from .heart_rate import PolarH10
except ImportError:
    pass # Bypassed if running solely for post-hoc intrinsics

# --- Intrinsic Calibration Configuration ---
CHESSBOARD_SIZE = (9, 6)       # Inner corners (width, height)
SQUARE_SIZE_METERS = 0.025     # Measured size of a single square in meters
BLUR_THRESHOLD = 80.0          # Minimum Variance of Laplacian to accept frame
GRID_COLS, GRID_ROWS = 4, 4    # Spatial bins to ensure edge/corner coverage
MAX_PER_BIN = 15               # Prevent over-fitting to the lens center
TARGET_RMS = 0.8               # Auto-prune worst frames until global RMS drops below this
MAX_CALIB_ITERATIONS = 10      # Failsafe for the pruning loop


class CalibrationTool:
    """Checks connectivity and status of all configured devices."""

    def __init__(self, settings: dict):
        self.settings = settings
        self._results: list[dict] = []

    def run(self) -> bool:
        """Run all device checks. Returns True if everything passes[cite: 1]."""
        print("=" * 60)
        print("  CALIBRATION TOOL - Device Connectivity Check")
        print("=" * 60)

        self._check_cameras()
        self._check_gopros()
        self._check_heart_rate()
        self._check_microphone()
        self._check_intrinsics()
        self._print_summary()

        all_passed = all(r["status"] in ("PASS", "SKIP", "WARN") for r in self._results)
        return all_passed

    def _check_cameras(self):
        """Check each USB camera can connect and capture a frame[cite: 1]."""
        cameras_cfg = self.settings.get("cameras", [])
        if not cameras_cfg:
            print("\nNo USB cameras configured.")
            return

        print(f"\n--- USB Cameras ({len(cameras_cfg)}) ---")
        for cfg in cameras_cfg:
            if not cfg.get("enabled", True):
                self._results.append({"device": cfg["name"], "type": "USB Camera", "status": "SKIP", "detail": "Disabled"})
                continue

            config = CameraConfig(
                id=cfg["id"], name=cfg["name"], device_index=cfg["device_index"],
                resolution=tuple(cfg["resolution"]), fps=cfg["fps"], enabled=cfg["enabled"]
            )
            camera = Camera(config)
            connected = camera.open()
            frame = None
            frame_shape = None

            if connected:
                frame = camera.read_frame()
                if frame is not None:
                    frame_shape = f"{frame.shape[1]}x{frame.shape[0]}"
            camera.close()

            passed = connected and frame is not None
            detail = ["connected" if connected else "connection failed"]
            if frame is not None: detail.append(f"frame captured ({frame_shape})")
            elif connected: detail.append("frame capture failed")

            status = "PASS" if passed else "FAIL"
            print(f"  {cfg['name']}: {status} - {', '.join(detail)}")
            self._results.append({"device": cfg["name"], "type": "USB Camera", "status": status, "detail": ", ".join(detail)})

    def _check_gopros(self):
        """GoPros always run in manual mode - mark them accordingly[cite: 1]."""
        gopro_cfgs = self.settings.get("gopros", [])
        enabled_cfgs = [c for c in gopro_cfgs if c.get("enabled", True)]
        if not enabled_cfgs: return

        print(f"\n--- GoPro Cameras ({len(enabled_cfgs)}) - Manual Mode ---")
        for cfg in gopro_cfgs:
            if not cfg.get("enabled", True):
                self._results.append({"device": cfg["name"], "type": "GoPro", "status": "SKIP", "detail": "Disabled"})
                continue

            print(f"  {cfg['name']}: Manual mode - start/stop GoPros yourself")
            self._results.append({"device": cfg["name"], "type": "GoPro", "status": "SKIP", "detail": "manual mode"})

    def _check_heart_rate(self):
        """Check Polar H10 connectivity and signal[cite: 1]."""
        hr_settings = self.settings.get("heart_rate", {})
        if not hr_settings.get("enabled", False):
            print("\nPolar H10: Disabled in config.")
            return

        print("\n--- Polar H10 Heart Rate Monitor ---")
        monitor = PolarH10(device_address=hr_settings.get("device_address"), ecg_enabled=False)
        connected = monitor.connect()
        battery = monitor.battery_level if connected else None
        hr_signal = False

        if connected:
            monitor.start_recording(phase="calibration_check")
            time.sleep(3.0)
            monitor.stop_recording()
            hr_signal = len(monitor.get_samples()) > 0

        detail = ["connected" if connected else "connection failed"]
        if battery is not None: detail.append(f"battery={battery}%")
        if connected: detail.append("HR signal detected" if hr_signal else "no HR signal")

        status = "PASS" if connected and hr_signal else "FAIL"
        print(f"  Polar H10: {status} - {', '.join(detail)}")
        self._results.append({"device": "Polar H10", "type": "Heart Rate", "status": status, "detail": ", ".join(detail)})
        monitor.disconnect()

    def _check_microphone(self):
        """Check that the configured USB microphone can record audio[cite: 1]."""
        mic_settings = self.settings.get("microphone", {})
        if not mic_settings.get("enabled", False):
            print("\nMicrophone: Disabled in config.")
            return

        print("\n--- Microphone ---")
        try:
            from .audio import AudioConfig, AudioRecorder, find_audio_device
            import tempfile, os

            device_name = mic_settings.get("device_name", "Tonor")
            device_index = mic_settings.get("device_index") or find_audio_device(device_name)

            if device_index is None:
                print(f"  Microphone '{device_name}': FAIL - device not found")
                self._results.append({"device": f"Microphone ({device_name})", "type": "Audio", "status": "FAIL", "detail": "device not found"})
                return

            config = AudioConfig(device_name=device_name, device_index=device_index, sample_rate=mic_settings.get("sample_rate", 44100), channels=mic_settings.get("channels", 1))
            recorder = AudioRecorder(config)
            tmp_path = tempfile.mktemp(suffix=".wav")
            opened = recorder.open(tmp_path)
            has_signal = False

            if opened:
                recorder.start_recording()
                time.sleep(1.0)
                recorder.stop_recording()
                recorder.close()
                has_signal = os.path.getsize(tmp_path) > 1000
                os.unlink(tmp_path)

            detail = [f"device {device_index}"] if opened else ["failed to open"]
            if opened: detail.append("signal detected" if has_signal else "no signal")

            status = "PASS" if opened and has_signal else "FAIL"
            print(f"  Microphone ({device_name}): {status} - {', '.join(detail)}")
            self._results.append({"device": f"Microphone ({device_name})", "type": "Audio", "status": status, "detail": ", ".join(detail)})

        except Exception as e:
            print(f"  Microphone: FAIL - {e}")
            self._results.append({"device": "Microphone", "type": "Audio", "status": "FAIL", "detail": str(e)})

    def _check_intrinsics(self):
        """Validates that GoPro intrinsic matrices have been computed and exist."""
        print("\n--- Intrinsic Calibration Matrices ---")
        for cam_type in ["front", "side"]:
            json_path = Path(f"{cam_type}_calibration.json")
            if json_path.exists():
                status, detail = "PASS", f"Loaded {json_path.name}"
            else:
                status, detail = "WARN", f"Missing {json_path.name} (Run --intrinsics)"
                
            print(f"  {cam_type.capitalize()} Matrix: {status} - {detail}")
            self._results.append({"device": f"{cam_type.capitalize()} Matrix", "type": "Intrinsics", "status": status, "detail": detail})

    def _print_summary(self):
        """Print a formatted summary table[cite: 1]."""
        print("\n" + "=" * 60 + "\n  CALIBRATION SUMMARY\n" + "=" * 60)
        if not self._results: return

        name_w = max(len(r["device"]) for r in self._results) + 2
        type_w = max(len(r["type"]) for r in self._results) + 2
        header = f"  {'Device':<{name_w}} {'Type':<{type_w}} Status  Detail"
        
        print(header + "\n  " + "-" * (len(header) - 2))
        for r in self._results:
            print(f"  {r['device']:<{name_w}} {r['type']:<{type_w}} {r['status']:<8}{r['detail']}")

        passed = sum(1 for r in self._results if r["status"] == "PASS")
        failed = sum(1 for r in self._results if r["status"] == "FAIL")
        total = len(self._results)

        print(f"\n  Results: {passed}/{total} passed")
        print("  Overall: PASS - All devices ready" if failed == 0 else "  Overall: FAIL - Some devices not available")
        print("=" * 60)


class CameraCalibrator:
    """Extracts checkerboards, enforces spatial binning, and solves global intrinsics."""
    def __init__(self, name: str):
        self.name = name
        self.grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        self.objpoints, self.imgpoints = [], []
        self.image_size = None
        
        self.objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE_METERS

    def process_frame(self, frame) -> bool:
        if self.image_size is None:
            self.image_size = (frame.shape[1], frame.shape[0])
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
        
        if not ret: return False
            
        x_min, x_max = int(max(0, np.min(corners[:, 0, 0]) - 10)), int(min(gray.shape[1], np.max(corners[:, 0, 0]) + 10))
        y_min, y_max = int(max(0, np.min(corners[:, 0, 1]) - 10)), int(min(gray.shape[0], np.max(corners[:, 0, 1]) + 10))
        roi = gray[y_min:y_max, x_min:x_max]
        
        if roi.size == 0 or cv2.Laplacian(roi, cv2.CV_64F).var() < BLUR_THRESHOLD:
            return False 
            
        col = min(int((np.mean(corners[:, 0, 0]) / self.image_size[0]) * GRID_COLS), GRID_COLS - 1)
        row = min(int((np.mean(corners[:, 0, 1]) / self.image_size[1]) * GRID_ROWS), GRID_ROWS - 1)
        
        if self.grid[row, col] >= MAX_PER_BIN:
            return False 
            
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        self.grid[row, col] += 1
        self.imgpoints.append(corners_refined)
        self.objpoints.append(self.objp)
        return True

    def auto_prune_and_calibrate(self):
        print(f"\n[{self.name}] Initiating solve with {len(self.imgpoints)} frames...")
        for iteration in range(MAX_CALIB_ITERATIONS):
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_size, None, None)
            
            errors = []
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                errors.append((i, error))
                
            print(f"  Iteration {iteration + 1}: Global RMS = {ret:.4f} px")
            if ret <= TARGET_RMS or len(self.imgpoints) < 20:
                print(f"  [{self.name}] Calibration locked. Final RMS: {ret:.4f}")
                return mtx, dist
                
            errors.sort(key=lambda x: x[1], reverse=True)
            prune_count = max(1, int(len(errors) * 0.05))
            indices_to_drop = {e[0] for e in errors[:prune_count]}
            
            print(f"  Dropping {prune_count} worst outlier frames...")
            self.objpoints = [p for i, p in enumerate(self.objpoints) if i not in indices_to_drop]
            self.imgpoints = [p for i, p in enumerate(self.imgpoints) if i not in indices_to_drop]
            
        return mtx, dist


def sample_video(video_path: Path, calibrator: CameraCalibrator):
    cap = cv2.VideoCapture(video_path.as_posix())
    if not cap.isOpened(): return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_to_check, max_frames, current_frame = int(fps * 0.5), int(fps * 120), 0
    
    while current_frame < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret: break
            
        if calibrator.process_frame(frame):
            current_frame += int(fps * 2) 
        else:
            current_frame += frames_to_check
    cap.release()


def run_intrinsic_generation(dataset_path: str):
    root_dir = Path(dataset_path).resolve()
    front_calibrator = CameraCalibrator("FRONT_CAMERA")
    side_calibrator = CameraCalibrator("SIDE_CAMERA")
    
    print("Extracting sharp, spatially-diverse checkerboard frames...")
    for item in sorted(root_dir.iterdir()):
        if item.is_dir() and item.name.startswith("P"):
            gopro_dir = item / "gopro_footage"
            if not gopro_dir.is_dir(): continue
                
            for video_file in gopro_dir.iterdir():
                if not video_file.is_file() or video_file.suffix.lower() not in {'.mp4', '.mov'}: continue
                stem = video_file.stem.lower()
                if "scoring" in stem or "sync_full" in stem: continue #
                    
                if "front" in stem: sample_video(video_file, front_calibrator)
                elif "side" in stem: sample_video(video_file, side_calibrator)

    front_mtx, front_dist = front_calibrator.auto_prune_and_calibrate()
    side_mtx, side_dist = side_calibrator.auto_prune_and_calibrate()
    
    for name, mtx, dist in [("front", front_mtx, front_dist), ("side", side_mtx, side_dist)]:
        with open(f"{name}_calibration.json", "w") as f:
            json.dump({"camera_matrix": mtx.tolist(), "dist_coeffs": dist.tolist()}, f, indent=4)
        print(f"Saved {name}_calibration.json")


def main():
    parser = argparse.ArgumentParser(description="IRIS Pre-flight & Intrinsic Calibration Suite")
    parser.add_argument("--hardware", action="store_true", help="Run device connectivity checks")
    parser.add_argument("--config", type=str, default="settings.json", help="Path to hardware settings JSON")
    parser.add_argument("--intrinsics", type=str, metavar="DATA_DIR", help="Generate permanent camera matrices from dataset")
    args = parser.parse_args()

    if args.intrinsics:
        run_intrinsic_generation(args.intrinsics)

    if args.hardware:
        try:
            with open(args.config, 'r') as f:
                settings = json.load(f)
        except Exception:
            settings = {} # Fallback to empty to still run GoPro/Intrinsic checks
            print(f"Warning: Could not load {args.config}")
            
        tool = CalibrationTool(settings)
        tool.run()
        
    if not args.hardware and not args.intrinsics:
        parser.print_help()

if __name__ == "__main__":
    main()