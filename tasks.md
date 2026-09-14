# FlowAnalytics Task Plan

## 1) Review the three flagged narration files and resolve them manually

Goal: Resolve the only narration files that do not match their timed transcript exactly.

Files to review:
- D:\FlowAnalytics\flow transcripts\P02_audio_narration.txt
- D:\FlowAnalytics\flow transcripts\P28_audio_narration.txt
- D:\FlowAnalytics\flow transcripts\P42_audio_narration.txt

Why this is first:
- It is the smallest remaining cleanup item.
- It removes the known review cases before any render work.
- It keeps the subtitle pipeline honest: only exact or reviewed matches should be used.

What to do:
- Open each cleaned TXT file and compare it to the corresponding SRT/JSON output.
- Remove obviously erroneous WhisperX artifacts such as repeated metadata strings like "Hubsan X4 H502E Desire 2-3-18".
- Keep the cleaned human wording as the source of truth.
- If extra timed spoken lines are present in the SRT but absent from the TXT, decide whether they should be removed or whether the TXT should be expanded.
- Save the final cleaned version in the same folder or in a clearly labeled review folder if you want to preserve the original untouched.

Success check:
- Each file should end up with either:
  - exact alignment between TXT and SRT, or
  - an explicit, reviewed set of extra timed lines that are intentionally kept.

---

## 2) Generate calibration files for the GoPro setup

Goal: Create the files required for undistortion.

Target files:
- D:\FlowAnalytics\calibration\front_calibration.json
- D:\FlowAnalytics\calibration\side_calibration.json

Why this is next:
- Calibration is required before any real undistorted video output can be produced.
- The MVP pipeline and undistortion tool require these before rendering.

What to do:
- Run the calibration script against the data folder.
- Save output into D:\FlowAnalytics\calibration.

Success check:
- Both JSON files exist and load without error.

---

## 3) Validate one trial end-to-end with inspect + timing + chart

Goal: Confirm the metadata pipeline for one known-good trial.

Trial example:
- D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737

What to do:
- Run the inspection command.
- Run timing and pause-map checks.
- Generate chart output.
- Confirm the review session timing is sane.

Success check:
- No missing required files.
- Timing report is valid.
- Pause map looks coherent.
- Chart preview is created.

---

## 4) Run the undistortion step for both GoPro videos

Goal: Produce corrected front and side videos.

What to do:
- Use the prepare command with the calibration directory.
- Keep the corrected videos in the trial's MVP/generated folder.

Success check:
- The trial contains corrected front and side video outputs in the generated folder.

---

## 5) Render a 15-second preview

Goal: Confirm layout, audio, chart, and subtitle behavior before a full render.

What to do:
- Render a short preview clip.
- Feed in the corrected GoPro videos, face video, narration, chart, and subtitle file.
- Review the clip before doing anything longer.

Success check:
- Front on left, side on right.
- Audio plays and is quieter under narration.
- Chart and subtitles are legible.
- No obvious frame or sync glitches.

---

## 6) Render the full review video once the preview looks good

Goal: Produce the final performance review output.

What to do:
- Use the same validated settings as the preview, but for the full review duration.
- Retain all generated intermediates under the trial's MVP/generated folder.

Success check:
- Final review video renders without crashes.
- Review timing holds during pauses.
- Subtitle and chart stay aligned.

---

## 7) Expand the workflow to the remaining trials

Goal: Apply the proven process to the rest of the dataset.

What to do:
- Repeat the same sequence for each trial.
- Batch only after the first one is confirmed.

Success check:
- At least one trial is fully validated before batch processing begins.

---

## 8) Only after the workflow is stable: automate or batch the remaining trials

Goal: Reduce manual work once the process is proven.

What to do:
- Batch the remaining trials with the same settings.
- Keep per-trial outputs and logs.

Success check:
- New batches follow the same verified settings and produce traceable outputs.
