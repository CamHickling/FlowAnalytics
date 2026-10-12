# Camera Calibration

`scripts/osr_instr_calibrate.py` computes one intrinsic fisheye calibration per
GoPro (front and side). `scripts/undistort.py` and `scripts/mvp.py prepare` then
use those calibrations to undistort the footage.

## Running it

```powershell
Push-Location calibration
python ..\scripts\osr_instr_calibrate.py --intrinsics $env:FLOW_ANALYTICS_DATA_ROOT
Pop-Location
```

This writes `front_calibration.json` and `side_calibration.json` to the current
directory. Each file stores `model: "fisheye"`, the camera matrix, and the
distortion coefficients.

The script can take a while. OpenCV may warn about multiple streams in GoPro
MP4s. The warning is harmless unless the process stalls or no output files are
written. If reads fail, try setting `OPENCV_FFMPEG_READ_ATTEMPTS=16384`.

`--hardware [--config settings.json]` runs device connectivity pre-flight checks
instead of calibrating.

## How the solve works

A calibration computed from a single video is easily thrown off by noise, motion
blur, and poor checkerboard placement. The script treats each GoPro lens as fixed
hardware and builds one master solve from frames sampled across the whole
dataset:

| Safeguard | Detail | Constant |
| --- | --- | --- |
| **Cross-trial aggregation** | Samples checkerboard passes from the first `CALIBRATION_SCAN_SECONDS` of every trial's `gopro_footage/`. This averages out lighting and thermal variation. | 90 s at 3 fps |
| **Motion-blur rejection** | Variance of Laplacian, computed only inside the detected board's bounding box. Frames below the threshold are dropped. | `BLUR_THRESHOLD = 80` |
| **Spatial grid binning** | Splits the frame into a grid and caps the frames kept per cell, so the fisheye edges are covered as well as the centre. | 4 × 4 grid, `MAX_PER_BIN = 15` |
| **Sub-pixel refinement** | Refines corners with `cv2.cornerSubPix`. | 11 × 11 window |
| **Fisheye model** | Solves with `cv2.fisheye.calibrate`, because GoPro wide-FOV lenses are true fisheye. Ill-conditioned frames are dropped automatically. | `CHESSBOARD_SIZE = (10, 7)` inner corners, 25 mm squares |
| **Iterative reprojection pruning** | While global RMS error is above target, drops the worst 5 % of frames by reprojection error and re-solves. | `TARGET_RMS = 0.8` px, at most 10 iterations |

## Files in `calibration/`

| Path | Contents |
| --- | --- |
| `front_calibration.json`, `side_calibration.json` | Active calibrations used by the pipeline |
| `archive/` | Earlier solves kept for comparison (standard pinhole model, wrong-board-size fisheye) |
| `test/` | Calibrations from a test run |
