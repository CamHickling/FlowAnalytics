# FlowAnalytics

![FlowAnalytics logo](FlowAnalyticsLogo.png)

Tools for synchronizing and reviewing dual-camera athlete performance recordings.
The working project and dataset are located on `D:` to keep large video files off
the system drive.

## Current Status

Implemented and testable:

- Constellation-style audio landmark synchronization in `scripts/cfsync.py`.
- Side-by-side GoPro synchronization tools in `scripts/batch_sync.py` and
   `scripts/visual_sync.py`.
- Standard OpenCV calibration in `scripts/osr_instr_calibrate.py`.
- Per-camera video undistortion in `scripts/undistort.py`.
- Staged MVP inspection, timing, pause mapping, chart generation, preparation,
   and short preview rendering in `scripts/mvp.py`.
- WhisperX launcher scripts for transcription workflows.

The full-length pause-aware MVP composition is still under development. The
`render-preview` command is for short validation renders; do not treat it as the
finished batch compositor yet.

## Repository Layout

```text
FlowAnalytics/
├── scripts/                         Python processing tools
├── tests/                           Test and synchronization utilities
├── Iris_Recorded_Taekwondo_Data/    Trial data on D:
├── MVP_SPEC.md                      MVP composition specification
├── FlowAnalyticsLogo.png            Repository logo
├── requirements.txt                 Core Python dependencies
├── run_whisperx_batch.ps1           PowerShell WhisperX launcher
└── run_whisperx_capture.bat        Windows batch WhisperX launcher
```

Each trial normally contains `gopro_footage`, `review`, `heart_rate`, and a
`*_sync_manifest.json`. MVP generated files are written under the trial's
`MVP/` directory.

## Setup

Open the D: checkout and use an environment with the dependencies in
`requirements.txt`:

```powershell
cd D:\FlowAnalytics
python -m pip install -r requirements.txt
```

The repository also contains Conda environment definitions for WhisperX:
`environment-whisperx-gpu.yml` and `environment-whisperx-gpu-min.yml`.

## Staged MVP Workflow

Run each stage separately against one trial before processing a batch:

```powershell
cd D:\FlowAnalytics
$trial = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737'
$calibration = 'D:\FlowAnalytics\calibration'

# Discover inputs and report missing prerequisites.
python scripts\mvp.py inspect --trial $trial --calibration-dir $calibration

# Validate manifest timing and review pause/resume pairing.
python scripts\mvp.py timing --trial $trial
python scripts\mvp.py pause-map --trial $trial

# Generate a heart-rate JSON data file and PNG preview.
python scripts\mvp.py chart --trial $trial --at-sec 60

# Show the undistortion plan without processing video.
python scripts\mvp.py prepare --trial $trial --calibration-dir $calibration --dry-run

# Undistort front and side GoPro videos into trial\MVP\generated.
python scripts\mvp.py prepare --trial $trial --calibration-dir $calibration

# Render a short composition preview.
python scripts\mvp.py render-preview --trial $trial --duration 15
```

Add cleaned timestamped subtitles explicitly when available:

```powershell
python scripts\mvp.py render-preview --trial $trial --duration 15 `
   --subtitles "$trial\review\narration.srt"
```

The preview supports `--front-start-sec`, `--side-start-sec`, and
`--review-start-sec` for testing alignment. Calibration JSON files must exist in
the directory supplied to `--calibration-dir` as `front_calibration.json` and
`side_calibration.json`.

## Calibration and Undistortion

Generate calibration matrices from checkerboard footage:

```powershell
cd D:\FlowAnalytics
$calibration = 'D:\FlowAnalytics\calibration'
$dataRoot = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data'
New-Item -ItemType Directory -Force $calibration | Out-Null
Push-Location $calibration
python D:\FlowAnalytics\scripts\osr_instr_calibrate.py --intrinsics $dataRoot
Pop-Location
```

This creates the standard OpenCV calibration files used by `undistort.py`.
Undistortion processes front and side independently before the MVP composition.

## Dataset Location

Use this environment variable for scripts that support a configurable data root:

```powershell
$env:FLOW_ANALYTICS_DATA_ROOT = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data'
```

The large dataset and generated videos should remain on `D:`. Video, audio, CSV,
and archive formats are excluded from Git by `.gitignore`; Git tracks the scripts,
configuration, documentation, and metadata needed to reproduce the workflow.

## WhisperX Launchers

The PowerShell launcher accepts `-DataRoot`, `-Model`, `-Device`, `-Language`,
`-Resume`, and `-NoDiarize`. It currently references a local Anaconda executable,
so update `$pythonExe` in `run_whisperx_batch.ps1` if that environment is installed
elsewhere.

```powershell
.\run_whisperx_batch.ps1 `
   -DataRoot D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data `
   -Device cpu `
   -Language en `
   -Resume
```

The batch launcher uses the `capture` Conda environment and defaults to CUDA.
Use it only when that environment and GPU runtime are available.

## Storage Cleanup

Package caches can be reviewed and cleared with:

```powershell
conda clean -a
pip cache purge
```

Do not remove the C: dataset until the D: copy has been verified. The current
working checkout is `D:\FlowAnalytics`.

## Related Documentation

- [MVP specification](MVP_SPEC.md)
- [Technical calibration notes](scripts/markdown_gen.py)

## Contributing

Keep changes focused, validate staged commands against one trial, and avoid
committing large media files. Use pull requests for changes intended for the
shared repository.
