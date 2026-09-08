# FlowAnalytics

FlowAnalytics is a repository focused on analyzing workflow, process, and data flow patterns. It provides tools and examples to help teams understand, visualize, and optimize end-to-end flows for more efficient operations.

## Key Features

- Analyze flow data and metrics
- Visualize system or process pathways
- Support for customizable analytics workflows
- Easy-to-extend structure for additional flow analysis use cases

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/CamHickling/FlowAnalytics.git
   cd FlowAnalytics
   ```
2. Review the project files and configuration.
3. Install any required dependencies as appropriate for your environment.

## Usage

- Explore the repository structure to identify analytics scripts and flow definitions.
- Run example workflows or analysis tools included in the repository.
- Adapt existing components to support your own flow data sources and visualization needs.

## Keep Large Data on D:

The video dataset and generated outputs can be stored outside the repository. The
sync scripts honor `FLOW_ANALYTICS_DATA_ROOT`, and the WhisperX launchers accept a
dataset root on `D:`.

To migrate the existing dataset safely from PowerShell:

```powershell
$source = 'C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data'
$destination = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
robocopy $source $destination /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /XJ
```

After checking that the files are present on `D:`, use the new location for the
rest of the session:

```powershell
$env:FLOW_ANALYTICS_DATA_ROOT = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data'
python scripts\batch_sync.py
python scripts\visual_batch.py
python scripts\cfsync.py $env:FLOW_ANALYTICS_DATA_ROOT
.\run_whisperx_batch.ps1 -DataRoot $env:FLOW_ANALYTICS_DATA_ROOT
```

Do not delete the old dataset until the copy has been inspected. Once it is
verified, remove it manually or with `Remove-Item -Recurse` to reclaim the space.

For quick cleanup of non-project caches, review first, then use `conda clean -a`
and `pip cache purge`. These commands can remove downloaded package caches but do
not move the Conda environment itself.

## MVP Pipeline Stages

The MVP compositor is intentionally staged so each part can be checked before a
long render:

```powershell
$trial = 'D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737'
$calibration = 'D:\FlowAnalytics\calibration'

# 1. Discover files and report missing prerequisites.
python scripts\mvp.py inspect --trial $trial --calibration-dir $calibration

# 2. Verify manifest bounds and pause/resume pairing.
python scripts\mvp.py timing --trial $trial
python scripts\mvp.py pause-map --trial $trial

# 3. Generate and inspect the heart-rate chart preview.
python scripts\mvp.py chart --trial $trial --at-sec 60

# 4. Preview the planned undistortion outputs without processing video.
python scripts\mvp.py prepare --trial $trial --calibration-dir $calibration --dry-run

# 5. Undistort front and side into $trial\MVP\generated\.
python scripts\mvp.py prepare --trial $trial --calibration-dir $calibration

# 6. Render a short composition preview after preparation.
python scripts\mvp.py render-preview --trial $trial --duration 15
```

The preview command accepts `--front-start-sec`, `--side-start-sec`, and
`--review-start-sec` so source alignment can be tested explicitly. Add
`--subtitles path\to\cleaned.srt` when the cleaned timestamped subtitles are
available. Full-length pause-aware composition will use the same validated timing
artifacts after this preview stage is confirmed.

## Contributing

Contributions are welcome. If you want to improve FlowAnalytics:

- Open an issue to discuss new features or fixes
- Submit a pull request with clear descriptions of your changes
- Keep code and documentation consistent with the repository style

## License

This repository uses the license defined in the project. If no license is present, please add one to clarify usage rights.
