# FlowAnalytics Current Context

## Current state

We are in the validation phase of the MVP video pipeline, not the full production batch phase.

The project is now organized around the following flow:

1. Resolve transcript mismatches and keep only reviewed narration pairs
2. Generate calibration matrices for the GoPro videos
3. Validate one trial end-to-end with inspect + timing + chart
4. Run undistortion on the front and side GoPro videos
5. Render a 15-second preview to validate layout, audio, and subtitles
6. Render the full-duration review video if the preview passes
7. Expand to more trials only after the first trial is validated

## Current step

We are currently at Step 2: generating the GoPro calibration files.

This is the required prerequisite before any real undistortion or preview render can be produced.

## Important decisions already made

- The cleaned narration text is the source of truth.
- WhisperX timing is used only to recover timing and segment timing.
- The first render target is a 15-second clip, not the full-length review.
- Performance audio should remain quieter than narration by default.
- The corrected/undistorted intermediate videos should be retained.
- The subtitle files are considered good to go after review of the flagged cases.

## Verified transcript status

A batch comparison of the narration TXT/SRT pairs found:
- 51 narration pairs processed
- 48 fully matched
- 3 requiring manual review: P02, P28, P42

The P01 pair was verified as fully matched:
- 160 cleaned lines
- 160 timed segments
- 160 exact matches
- 0 unmatched

## Files currently relevant

- Project root: D:\FlowAnalytics
- Main project docs: README.md, MVP_SPEC.md
- MVP pipeline: D:\FlowAnalytics\scripts\mvp.py
- Undistortion script: D:\FlowAnalytics\scripts\undistort.py
- Comparison utility: D:\FlowAnalytics\scripts\compare_narration_transcripts.py
- Transcript set: D:\FlowAnalytics\flow transcripts
- Synthesis outputs: D:\FlowAnalytics\flow transcripts\synthesis
- Calibration output directory: D:\FlowAnalytics\calibration
- Trial to validate first: D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737

## Next concrete command sequence

### Step 2: generate calibration files

```powershell
cd D:\FlowAnalytics

$calibration = 'D:\FlowAnalytics\calibration'
$dataRoot = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data'

New-Item -ItemType Directory -Force $calibration | Out-Null

$env:OPENCV_FFMPEG_READ_ATTEMPTS = '16384'

Push-Location $calibration
python D:\FlowAnalytics\scripts\osr_instr_calibrate.py --intrinsics $dataRoot
Pop-Location
```

Check afterwards:

```powershell
Get-ChildItem D:\FlowAnalytics\calibration -File
```

Success means the calibration JSON files exist.

## Next phase after calibration

Once calibration succeeds, the next phase is:

1. inspect the selected trial
2. validate timing and pause-map
3. generate chart output
4. undistort both GoPro videos
5. render a 15-second preview
6. only then render the full video

## Important caution

The calibration script can take time. Do not treat the OpenCV FFmpeg warning as a fatal error unless the process stalls or the output files never appear. The warning usually means the GoPro MP4s have multiple streams and OpenCV is being conservative.
