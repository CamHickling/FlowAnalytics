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

## Stereo baseline

`scripts/stereo_baseline.py` measures the distance between the front GoPro
(camera 1) and the side GoPro (camera 2) from moments where the checkerboard is
fully visible in both cameras.

```powershell
# raw footage of one trial, paired with the lag from MVP/sync_offsets(_v2).json
python scripts\stereo_baseline.py --trial $env:FLOW_ANALYTICS_DATA_ROOT\P01_20260223_112737

# a synced pair from export_synced_pairs.py (already in sync, already undistorted)
python scripts\stereo_baseline.py --front P01_front_synced.mp4 --side P01_side_synced.mp4 --out P01_stereo.json

# two folders of frames, paired by file name
python scripts\stereo_baseline.py --front frames\cam1 --side frames\cam2 --lens raw --pattern 7 10
```

How it works:

1. **Pattern probe.** Unless `--pattern` is given, it tries 6 × 9, then 7 × 10 inner corners
   on the first 60 frames, and keeps the largest pattern that is detected. A smaller pattern
   can match a sub-block of a larger board.
2. **Corners.** `findChessboardCorners`, falling back to `findChessboardCornersSB`, then
   `cornerSubPix`. Blurred boards are dropped (`BLUR_THRESHOLD`). Corners are put in one
   order for both cameras: the grid is right-handed in the image, and the first square is dark.
3. **Intrinsics.** By default it reuses `calibration/{front,side}_calibration.json`. For
   `*_undistorted` / `*_synced` videos it uses the rectified matrix that `undistort.py`
   produced (`--alpha`, default 1.0). `--solve-intrinsics` solves them from the frames
   instead (fisheye for raw footage). The per-camera RMS is reported either way.
4. **Stereo.** Corners are undistorted to an ideal pinhole image, then `cv2.stereoCalibrate`
   runs with `CALIB_FIX_INTRINSIC` on matched pairs. Each pair is also solved on its own
   with `solvePnP`, as a cross-check and as the fallback when there are fewer than 5 pairs.
5. **Distances.** `baseline_3d_cm = ||T||`, and
   `floor_distance_cm = sqrt(baseline² − (side_height − front_height)²)`.
   The default lens heights are 104 cm (front) and 117 cm (side).
6. **Sanity check.** The cameras sit 380 cm (front, wall A) and 336 cm (side, wall B) from
   a 90° corner, each 0–30 cm out from its wall, so the expected floor distance is
   464.9–507.2 cm. The script also reports which wall offsets (d1, d2) are consistent with
   the measurement. It flags results outside that range, and stereo RMS above 0.5 px.

Distances are between the optical centres, which can be a few cm from the tripod centres.
