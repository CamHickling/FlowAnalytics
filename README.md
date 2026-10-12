<p align="center">
  <img src="docs/assets/FlowAnalyticsLogo.png" alt="FlowAnalytics logo" width="320">
</p>

<h1 align="center">FlowAnalytics</h1>

<p align="center">
  Synchronize, undistort, transcribe, and compose multi-camera athlete performance
  recordings into a single pause-aware review video with narration, subtitles,
  and live heart-rate.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</p>

---

## Overview

FlowAnalytics is the processing toolkit behind a flow-state study of taekwondo
poomsae performance. Each trial is recorded by several cameras (front GoPro,
side GoPro, overhead, face cam) alongside a heart-rate monitor, and the athlete
later narrates a review of their own performance.

The toolkit turns those raw recordings into one composed **review video** per trial:

```text
+-----------------------------+------------------------------+
| Front GoPro (undistorted)   | Side GoPro (undistorted)     |
|                             |              +-------------+ |
|                             |              | Face camera | |
|                             |              +-------------+ |
+-----------------------------+------------------------------+
| Subtitles from the athlete's cleaned narration             |
+------------------------------------------------------------+
| Heart-rate graph with moving current-value marker          |
+------------------------------------------------------------+
```

When the athlete pauses playback to talk, the performance footage freezes while
narration, the face cam, subtitles, and heart-rate keep running. See the
[MVP specification](docs/MVP_SPEC.md) for the full design.

### Features

| Stage | What it does | Tool |
| --- | --- | --- |
| **Sync** | Audio-landmark ("constellation") and cross-correlation sync between cameras, plus interactive manual sync | `scripts/cfsync.py`, `scripts/batch_sync.py`, `scripts/sync_review.py` |
| **Calibrate** | Robust OpenCV intrinsic calibration from checkerboard passes across the whole dataset | `scripts/osr_instr_calibrate.py` |
| **Undistort** | Per-camera lens correction of GoPro footage | `scripts/undistort.py` |
| **Transcribe** | Batch WhisperX transcription with resume support | `scripts/process_videos_whisperx.py` |
| **Align subtitles** | Merge human-cleaned narration onto WhisperX timing, flagging mismatches | `scripts/compare_narration_transcripts.py` |
| **Compose** | Staged pipeline: inspect → timing → pause map → chart → prepare → render | `scripts/mvp.py` |
| **Analyze** | Heart-rate analysis, narration segmentation, LLM-assisted coding, inter-rater reliability | `analysis/` |

## Quick start

```bash
git clone https://github.com/CamHickling/FlowAnalytics.git
cd FlowAnalytics
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

FFmpeg is bundled through `imageio-ffmpeg`, so no separate install is needed for
the core pipeline. Optional extras:

```bash
pip install -r envs/requirements-analysis.txt          # analysis/ scripts
conda env create -f envs/environment-whisperx-gpu.yml  # WhisperX transcription (CUDA)
```

Point the tools at your dataset:

```powershell
$env:FLOW_ANALYTICS_DATA_ROOT = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data'
```

## Usage

The examples below use PowerShell; the scripts themselves are cross-platform.

```powershell
$trial       = "$env:FLOW_ANALYTICS_DATA_ROOT\P01_20260223_112737"
$calibration = '.\calibration'
```

### 1. Calibrate the cameras (once per camera set)

```powershell
Push-Location $calibration
python ..\scripts\osr_instr_calibrate.py --intrinsics $env:FLOW_ANALYTICS_DATA_ROOT
Pop-Location
```

Writes `front_calibration.json` and `side_calibration.json`. See
[docs/calibration.md](docs/calibration.md) for how the solver rejects blurry
frames and prunes outliers.

### 2. Run the staged MVP pipeline on one trial

Validate each stage on a single trial before running a batch:

```powershell
python scripts\mvp.py inspect   --trial $trial --calibration-dir $calibration   # find inputs, report gaps
python scripts\mvp.py timing    --trial $trial                                  # validate manifest timing
python scripts\mvp.py pause-map --trial $trial                                  # map review pauses
python scripts\mvp.py chart     --trial $trial --at-sec 60                      # heart-rate chart + preview
python scripts\mvp.py prepare   --trial $trial --calibration-dir $calibration --dry-run
python scripts\mvp.py prepare   --trial $trial --calibration-dir $calibration   # undistort front + side
python scripts\mvp.py render-preview --trial $trial --duration 15 `
    --subtitles "$trial\review\narration.srt"                                    # 15 s check clip
```

When the preview looks right, render the full video, or every trial at once:

```powershell
python scripts\mvp.py render --trial $trial
python scripts\mvp.py batch-render --data-root $env:FLOW_ANALYTICS_DATA_ROOT --skip-existing
```

Camera offsets can be suggested automatically (`suggest-offsets`), checked
visually (`verify-offsets`), or recorded from a manually matched moment
(`record-match`, or interactively with `scripts\sync_review.py`). Run
`python scripts\mvp.py <command> --help` for all options.

All intermediates (undistorted video, timing, chart data, subtitle cues) are
kept under `<trial>\MVP\generated\`.

### 3. Build timed subtitles from cleaned narration

The human-cleaned `<trial>_audio_narration.txt` is the **source of truth for
wording**; WhisperX is used only to recover timing.

```powershell
# Recover timing with WhisperX
python scripts\process_videos_whisperx.py "$trial\review" --audio-only `
    --output-dir "$trial\MVP\generated\whisperx" --model large-v3 `
    --device cuda --language en --no-diarize

# Merge cleaned wording onto the timed segments
python scripts\compare_narration_transcripts.py `
    --clean  "$trial\review\P01_audio_narration.txt" `
    --timed  "$trial\MVP\generated\whisperx\P01_audio_narration.srt" `
    --output-dir "$trial\MVP\generated\synthesis"
```

Unmatched lines are marked `needs_review` rather than given guessed timings, and
original files are never overwritten. For whole-dataset transcription, use
`tools\run_whisperx_batch.ps1` (pass `-PythonExe` or set
`FLOW_ANALYTICS_PYTHON`) or `tools\run_whisperx_capture.bat`.

### 4. Research analysis

Scripts in [`analysis/`](analysis) support the study itself:

| Script | Purpose |
| --- | --- |
| `heart_rate_analysis.py` | Full-session HR analysis: resting vs. warm-up baselines, performance ROI |
| `narration_make_units.py` | Segment narration into thought-units for coding |
| `narration_freeze.py` / `narration_promote.py` | Freeze hand-edited segmentation and build coder templates |
| `narration_llm_label.py` | Apply the coding rubric with Claude as an additional coder (needs `ANTHROPIC_API_KEY`) |
| `narration_kappa.py` | Cohen's, weighted, and Fleiss' kappa plus PABAK across coders |
| `count_playback_events.py` | Summarize playback start/stop events per session |

They read from `FLOW_ANALYTICS_DATA_ROOT`, falling back to
`./Iris_Recorded_Taekwondo_Data`.

## Dataset layout

Each trial folder follows this structure:

```
<data root>/
└── P01_20260223_112737/
    ├── gopro_footage/          front + side GoPro recordings, sync info
    ├── review/                 face cam, narration audio + cleaned transcript, review timestamps
    ├── heart_rate/             HR session CSVs and plots
    ├── P01_sync_manifest.json  event timeline for the session
    └── MVP/                    everything this toolkit generates
```

Participant data is **not** distributed with this repository. Keep the dataset
outside the repo and point `FLOW_ANALYTICS_DATA_ROOT` at it; the data folder
names are also listed in `.gitignore` as a safeguard.

## Project structure

```
FlowAnalytics/
├── scripts/        Core pipeline: sync, calibration, undistortion, transcription, MVP composer
├── analysis/       Research analysis: heart rate, narration coding, inter-rater reliability
├── tools/          Launchers and utilities (WhisperX batch runners, session lister, codepack)
├── experiments/    Early sync prototypes kept for reference
├── tests/          Unit tests
├── calibration/    Camera calibration JSON (archive/ and test/ hold older or trial solves)
├── envs/           Conda environments and optional requirement sets
└── docs/           Specification, calibration notes, project status, task plan
```

## Project status

FlowAnalytics is a **research prototype** in active development. The staged
pipeline is implemented end to end; validation is proceeding one trial at a time
before batch rendering. See [docs/project-status.md](docs/project-status.md) and
[docs/tasks.md](docs/tasks.md) for the current state and next steps.

## Testing

```bash
python -m unittest discover -s tests
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request.

## Author

**Cam Hickling**

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE)
for attribution.

## Acknowledgements

Thanks to Angela Czarina Mejia for support with the flow study.
