import os
from pathlib import Path

manual_content = """# IRIS Calibration Suite: Technical Manual

The `calibrate.py` suite is a unified command-line tool designed to manage experimental hardware diagnostics and compute permanent, highly accurate intrinsic camera matrices for fisheye lens correction.

## Core Computer Vision Techniques

Calculating camera intrinsics per-video is vulnerable to environmental noise, motion blur, and poor checkerboard placement. This pipeline treats GoPro lenses as fixed physical hardware, computing one master matrix per camera using the following automated safeguards:

* **Cross-Trial Aggregation:** The script dynamically samples frames across the entire dataset rather than relying on a single dedicated calibration video. This averages out subtle lens thermal expansion and varying lighting conditions.
* **Targeted Motion Blur Rejection:** It applies a Variance of Laplacian filter (`cv2.Laplacian().var()`) strictly within the bounding box of the detected checkerboard. Frames falling below the sharpness threshold are instantly discarded.
* **Spatial Grid Binning:** To prevent the solver from over-fitting to the center of the lens (leaving fisheye edges unmapped), the camera frame is mathematically divided into a 4x4 grid. The script caps each grid sector at 15 frames, forcing the pool to wait for extreme edge and corner shots before proceeding.
* **Subpixel Refinement:** OpenCV's `cv2.cornerSubPix` refines the detected grid intersections to sub-pixel accuracy, reducing foundational triangulation errors.
* **Iterative Reprojection Pruning:** After the initial `cv2.calibrateCamera` solve, the script calculates the Root Mean Square (RMS) reprojection error. If the global error exceeds 0.8 pixels, it sorts all frames by their individual L2 norm error, drops the worst-performing 5%, and re-solves. This loop repeats until the mathematical projection tightly matches the physical pixels.

## Pipeline Operation

The script operates in two primary modes via command-line arguments. 

### Step 1: Compute Intrinsic Matrices
Run this process once to generate your baseline lens profiles. Ensure your dataset contains standard checkerboard passes within the `gopro_footage` subfolders[cite: 2].

```bash
python -m scripts.calibrate --intrinsics Iris_Recorded_Taekwondo_Data