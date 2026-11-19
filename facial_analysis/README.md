# Facial Analysis (OpenFace)

Batch facial analysis of videos with [OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace):
facial Action Units (AUs), head pose, gaze and landmarks for every frame. Frames are then filtered by
tracking quality and summarized per video.

## Setup (one time)

```powershell
cd D:\FlowAnalytics\facial_analysis
powershell -ExecutionPolicy Bypass -File scripts\setup_openface.ps1   # OpenFace binary + models -> tools/
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The project uses its own `.venv`. The Anaconda base environment on this machine has a NumPy 1.x/2.x
conflict that stops pandas from importing.

## Usage

Set the input folders and options in `config.yaml`, then run:

```powershell
.\.venv\Scripts\python.exe -m faceanalysis list                # show the videos found and done/todo status
.\.venv\Scripts\python.exe -m faceanalysis extract --limit 2   # try a couple of videos first
.\.venv\Scripts\python.exe -m faceanalysis run                 # extract + analyze + plot everything
```

| Command   | What it does |
|-----------|--------------|
| `extract` | Runs `FeatureExtraction.exe` on each video, several in parallel (`workers`). Videos that already finished are skipped, so you can stop it and run it again. `--force` re-runs them. |
| `analyze` | Marks frames valid when `success == 1` and `confidence >= min_confidence`. It writes cleaned per-frame CSVs (invalid frames become NaN, so the timeline stays intact) and `summary.csv`. |
| `plot`    | Saves one AU-intensity heatmap per video. Gray means the face wasn't tracked. |
| `run`     | Runs all three in order. |

Run all commands from `D:\FlowAnalytics\facial_analysis`. Use `-c other_config.yaml` to point any
command at a different config.

### Trial face videos (narration and scoring)

`config_face_videos.yaml` processes the close-up face videos in each trial of
`Iris_Recorded_Taekwondo_Data`: `review/*_face_narration.mp4` and `scoring/*_face_scoring.mp4`
(106 videos, 53 per condition; P06 has neither).

```powershell
.\.venv\Scripts\python.exe -m faceanalysis -c config_face_videos.yaml run
```

Outputs go to `outputs/face_videos/`. `summary.csv` has these extra columns, read from each video's path:

| Column | Meaning |
|--------|---------|
| `participant` | e.g. `P02` (P29's narration file lacks the "P"; the trial folder is used) |
| `session` | `main` or `B` (P02B, P10B, P18B, P27B are separate rows) |
| `condition` | `narration` or `scoring` |
| `mouth_aus_speech_confounded` | `True` for narration. Speech moves AU10/12/14/15/17/20/23/25/26, so interpret those with care. All AUs are kept |
| `short_video` | Shorter than 30 s (the P02B clips are 5-10 s) |
| `low_tracking` | Fewer than 50% of frames valid |

## Outputs (`outputs/`)

```
openface/<video_id>.csv    raw OpenFace per-frame output (+ .done marker, _of_details.txt)
clean/<video_id>.csv       AUs/pose/gaze with a `valid` column; invalid frames set to NaN
summary.csv                one row per video: valid_fraction, AU means/SDs, AU presence rates, pose, gaze
plots/<video_id>.png       AU heatmap over time
logs/<video_id>.log        OpenFace console output (check here when a video fails)
manifest.csv               run history: status and processing time per video
```

`video_id` is built from the video's path relative to the input folder (e.g.
`front_go_pro_offload_1__DCIM__100GOPRO__GH017313`). GoPro filenames repeat across cameras, so the
filename alone isn't unique.

**OpenFace columns:** `AUxx_r` is intensity on a 0-5 scale (17 AUs). `AUxx_c` is presence, 0 or 1
(18 AUs). `pose_T*` is head position in mm and `pose_R*` is head rotation in radians. `gaze_angle_*`
is gaze direction in radians. See the
[OpenFace output format docs](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format).

## Notes and limitations

- **Speed:** about 15 fps per process on 1080p footage, so an 18-minute GoPro clip takes about
  35 minutes per worker. The full `D:\footage` set (about 187 videos, 366 GB) will take days. Use
  `--limit` and run it in batches. Re-running continues where it stopped.
- **Face size matters:** OpenFace expects a reasonably large, mostly frontal face. With wide-angle room
  footage, tracking fails often (the test clip tracked only 34% of frames). Check `valid_fraction` in
  `summary.csv`. The `analyze` step warns about videos with less than 50% valid frames. Side-camera
  (profile) views will be worse. Cropping or zooming to the person before running OpenFace would help.
- **One face per video:** `FeatureExtraction` follows a single face. For several people, use
  `FaceLandmarkVidMulti.exe`, which is in the same `tools/` folder.
- OpenFace 2.2 runs on the CPU only. The GPU is not used.
