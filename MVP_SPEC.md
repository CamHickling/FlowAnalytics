# MVP Performance Review Video Specification

## 1. Objective

Create one composed review video for each trial folder. The output combines the athlete's front and side GoPro performance footage, the athlete's narrated review, a persistent face-camera overlay, synchronized English subtitles, and a live heart-rate chart.

The output must preserve review pauses: while the athlete pauses playback and continues speaking, the performance image holds still while narration, face video, subtitles, and heart-rate time continue.

## 2. Scope

The first implementation is a command-line Python tool in `scripts/mvp.py`. It will process one trial or a dataset root and write output to a new `MVP` folder inside each trial. Source and output files must support a dataset root on `D:` via `FLOW_ANALYTICS_DATA_ROOT`.

FFmpeg will perform the final composition. Python will prepare timing metadata and the heart-rate chart assets.

## 3. Input Assets

### 3.1 GoPro performance footage

Each trial has two GoPro recordings under `gopro_footage/`:

- Front camera: displayed on the left.
- Side camera: displayed on the right.

Both GoPro sources must be undistorted independently before they are stitched
together. The MVP compositor consumes the corrected front and side videos, with
front on the left and side on the right.

### 3.2 Calibration and undistortion

The current calibration tool is `scripts/osr_instr_calibrate.py`. Its existing `--intrinsics DATA_DIR` mode generates `front_calibration.json` and `side_calibration.json`; it does not currently expose a video-undistortion mode.

The separate `scripts/undistort.py` tool uses those calibration files through an
isolated adapter:

```text
undistort_video(input_path, output_path, camera_name, calibration_path)
```

The tool processes the front and side videos separately, preserves their source
frame rate and dimensions, preserves source audio, and writes corrected files
under `MVP/generated/`. The MVP compositor then stitches those corrected files
side by side. The standard OpenCV model is used because the calibration script
calls `cv2.calibrateCamera()`, not the fisheye calibration API.

### 3.3 Review narration and face video

The logical review asset is `review/face_cam.mp4`. Existing trials commonly use generated names such as:

```text
review/<trial_id>_face_narration.mp4
review/<trial_id>_audio_narration.wav
```

The tool should use manifest file entries first, then documented filename fallbacks. The dedicated narration WAV is the primary speech source. The review face video remains visible in the top-right corner for the entire final video, including silent sections.

### 3.4 Sync manifest

Read the trial-level `<trial_id>_sync_manifest.json` on every run. It supplies wall-clock alignment events including:

- `overhead_recorder_start` and `overhead_recorder_stop`
- `face_recorder_start` and `face_recorder_stop`
- `audio_recorder_start` and `audio_recorder_stop`
- `review_video_player_shown`
- `review_playback_stop`

Media must not be assumed to start at time zero. Manifest wall times must be converted into offsets relative to a documented composition origin and recorded in the output timing report.

### 3.5 Review pause/resume timestamps

The review folder contains a file such as `<trial_id>_review_timestamps.json`. Events have this form:

```json
{
  "type": "pause",
  "wall_time": 1771875822.8576872,
  "video_position_sec": 409.1
}
```

Pair each `pause` with the next `resume`. During each interval:

- Hold the GoPro performance frame at `video_position_sec`.
- Continue narration audio and the face-camera timeline.
- Continue subtitles and the heart-rate marker.
- Preserve any frozen frames already written by the review recorder.

Unpaired or invalid pause events must fail that trial with a clear diagnostic.

### 3.6 Subtitles

Subtitles must be English, cleaned/edited, legible, and placed immediately above the heart-rate graph. Preferred inputs are:

1. An existing timestamped cleaned subtitle file (`.srt`, `.vtt`, or timestamped JSON).
2. A cleaned timestamped transcript supplied by the reviewer.
3. A timestamped transcription generated from the narration WAV only when enabled.

The current plain-text `audio_narration.txt` files are useful transcript references but have no timing. They must not be assigned arbitrary equal durations without an explicit approximation warning. Subtitle timing may use the same review playback timeline and pause events, but the implementation must confirm whether the edited subtitle timestamps are relative to audio time, face-video time, or review wall time.

Style: high-contrast white text with a dark outline or semi-opaque background, at most two lines, wrapped without covering the chart or face overlay.

### 3.7 Heart rate

Use `heart_rate/<trial_id>_hr_full_session.csv`, whose expected columns include `timestamp`, `bpm`, `sensor_contact`, and `phase`.

The chart must show BPM for the relevant interval and a moving vertical marker or dot at the current reading. It must handle missing or disconnected readings by holding the last valid value and recording the gap rather than crashing.

## 4. Timing Model

The composition timeline is the review wall-clock timeline.

1. Determine the interval where the two corrected GoPro videos overlap.
2. Determine the review start and end from manifest events.
3. Use the intersection of those intervals. The output therefore starts no earlier than the later required start and ends no later than the earliest required end. This is what “limit to the review session duration” means: no performance-only footage before review begins or after review ends.
4. Align face video, narration, subtitles, and heart rate using manifest offsets.
5. Apply pause intervals by holding only the performance frame while all review timeline elements continue.
6. Record the selected start, end, offsets, and pause intervals in JSON metadata.

## 5. Composition Layout

Default output: 1920x1080 at constant 30 FPS.

```text
+----------------------------------------------------------+
| Front GoPro (left)       | Side GoPro (right)             |
|                          +-------------------------------+|
|                          | Face narration overlay       ||
+----------------------------------------------------------+
| English subtitles                                        |
+----------------------------------------------------------+
| Heart-rate graph with moving current-value marker        |
+----------------------------------------------------------+
```

The GoPro panels remain dominant. The face overlay is configurable in size and margin and must not cover subtitles or the heart-rate chart.

## 6. Audio

The final AAC track contains:

- The athlete's narration as the primary audio.
- A quiet mix of the performance audio underneath it.

The performance level must be configurable, with an initial default around -18 dB relative to narration. The output timing must follow the review-limited interval.

## 7. Output

```text
<trial>/MVP/
    <trial_id>_MVP.mp4
    <trial_id>_MVP_timing.json
    generated/
```

The final MP4 must use H.264 video, AAC audio, constant frame rate, and no
unintended last-frame repetition after an input ends. Chart and subtitle assets
may be stored in `generated`; corrected videos are stored there only when the
optional correction mode is enabled.

## 8. Heart-Rate Axis Options

### Trial-specific scaling

Set the y-axis from that trial's observed BPM range, with padding.

- Uses screen space efficiently and makes small changes easy to see.
- Two videos cannot be compared visually by chart height because their axes may differ.
- A short range can make ordinary variation look more dramatic.

### Fixed shared scaling

Use the same y-axis limits for every trial, for example a project-defined BPM range.

- Chart height has the same meaning across trials and supports comparison.
- Small changes may look flatter, and the chosen range must cover all useful data.
- An outlier can force a range that makes normal readings hard to see.

Recommended MVP default: fixed shared scaling for cross-trial analysis, with configurable limits and a visible numeric current-BPM label. Trial-specific scaling remains an option for exploratory review videos.

## 9. Undistortion Cache Options

### Per-trial cache

Store corrected videos in `<trial>/MVP/generated/`. This is simple to audit and clean up, but can duplicate large corrected files if sources are reused.

### Shared cache

Store corrected videos in a configurable folder on `D:` keyed by source path, calibration file, and processing version. This avoids duplicate intermediates but requires cache invalidation when calibration or undistortion code changes.

Recommended future default if correction is enabled: per-trial
`MVP/generated/` for traceability. The initial MVP should not create an
undistortion cache at all.

## 10. Configuration and Errors

Command-line options should cover trial/dataset root, output folder, undistortion adapter and calibration paths, subtitle source, output dimensions/FPS, face scale, performance-audio level, cache mode, intermediate retention, dry-run, and single-trial testing.

Fail an individual trial with a useful message when required footage, manifest, narration, face video, heart-rate data, subtitle timing, or valid pause intervals are missing. Continue processing other trials in batch mode.

## 11. Acceptance Tests

Validate one trial with both GoPros, at least one pause/resume interval, narration, face video, heart rate, and cleaned timestamped subtitles. Confirm that:

- Front is left and side is right.
- Performance holds during review pauses.
- Narration and face video continue during pauses.
- Subtitles follow narration.
- The heart-rate marker advances continuously.
- The face overlay remains in the top-right throughout.
- Audio contains narration plus quiet performance sound.
- Output is review-limited, readable, and constant 30 FPS.

## 12. Remaining Questions

1. **Undistortion integration:** Should the first implementation stop unless a video-undistortion adapter is configured, or should it permit an explicitly labeled passthrough test mode? The current calibration script only writes the intrinsic JSON files; it does not yet undistort video itself.
2. **Subtitle time base:** Are the cleaned subtitle timestamps relative to the narration WAV, face video, or review wall clock? This determines the conversion needed when pauses occur.
3. **Heart-rate mode:** Confirm fixed shared scaling as the default, with trial-specific scaling as an option.
4. **Cache mode:** Confirm per-trial `MVP/generated` as the initial default, with a shared `D:` cache as a later optimization.
