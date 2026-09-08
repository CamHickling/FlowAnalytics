import os
import io
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter
import librosa
import imageio_ffmpeg

# ==============================================================================
# CONFIGURATION
# ==============================================================================
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".MP4", ".MOV"}
SAMPLE_RATE = 22050
HOP_LENGTH = 256            # ~11.6 ms temporal resolution
MAX_SEARCH_SEC = 90         # Maximum expected start difference (seconds)
ANALYSIS_DURATION = 180     # Analyze first 3 minutes of overlap
CONFIDENCE_THRESHOLD = 4.0  # Peak-to-background SNR ratio for acceptance
SNIPPET_DURATION = 5.0      # Duration of validation video in seconds
OUTPUT_FPS = 30             # Camera recordings are nominally 30 FPS
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
# ==============================================================================


def extract_spectral_landmarks(file_path, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, duration=ANALYSIS_DURATION):
    """
    Loads audio, filters dojang room boom, and extracts 2D spectral peak coordinates.
    """
    cmd = [
        FFMPEG_EXE, "-v", "error", "-i", str(file_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-t", str(duration),
        "-f", "wav", "pipe:1"
    ]
    try:
        decoded_audio = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"FFmpeg could not decode audio from '{file_path}': {error}") from exc

    y, _ = librosa.load(io.BytesIO(decoded_audio), sr=sr, mono=True)
    
    # High-pass filter above 400Hz to remove room rumble/echo
    sos = signal.butter(4, 400, btype='highpass', fs=sr, output='sos')
    y = signal.sosfilt(sos, y)

    # STFT to dB scale
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 2D local maximum filter for landmark detection
    neighborhood_size = (25, 15)  # (frequency bins, time frames)
    local_max = maximum_filter(S_db, size=neighborhood_size) == S_db
    
    # Filter out low-energy background floor
    peaks = local_max & (S_db > -35.0)
    freq_bins, time_frames = np.where(peaks)
    
    return freq_bins, time_frames


def compute_audio_offset(video1_path, video2_path):
    """
    Calculates time offset using combinatorial landmark voting.
    Returns: (offset_in_seconds, confidence_score, status_str)
    """
    f1, t1 = extract_spectral_landmarks(video1_path)
    f2, t2 = extract_spectral_landmarks(video2_path)

    # Map frequency bins to their occurrence timestamps for Cam 1
    lookup = {}
    for f, t in zip(f1, t1):
        lookup.setdefault(f, []).append(t)

    # Accumulate relative time offsets (Δt = t1 - t2) for matching frequencies
    max_lag_frames = int((MAX_SEARCH_SEC * SAMPLE_RATE) / HOP_LENGTH)
    time_diff_votes = []

    for f, t_cam2 in zip(f2, t2):
        if f in lookup:
            for t_cam1 in lookup[f]:
                diff = t_cam1 - t_cam2
                if abs(diff) <= max_lag_frames:
                    time_diff_votes.append(diff)

    if not time_diff_votes:
        return 0.0, 0.0, "FAILED_NO_MATCHES"

    # Build histogram of time differences
    bins = np.arange(-max_lag_frames, max_lag_frames + 1)
    counts, bin_edges = np.histogram(time_diff_votes, bins=bins)

    best_idx = np.argmax(counts)
    best_frame_diff = bin_edges[best_idx]
    peak_votes = counts[best_idx]
    
    # Metric: Peak-to-background noise ratio
    baseline_noise = np.median(counts) + 1e-5
    confidence = float(peak_votes / baseline_noise)
    
    offset_seconds = float(librosa.frames_to_time(best_frame_diff, sr=SAMPLE_RATE, hop_length=HOP_LENGTH))
    
    status = "SUCCESS" if (confidence >= CONFIDENCE_THRESHOLD and peak_votes >= 10) else "LOW_CONFIDENCE"
    return offset_seconds, confidence, status


def create_synced_video(vid1, vid2, offset_sec, out_path, duration=None):
    """
    Generates a synchronized side-by-side video using FFmpeg.
    When duration is None, the output runs for the full remaining overlap.
    """
    # Start 3 seconds after shared overlap begins to skip static setup
    padding = 3.0
    
    if offset_sec >= 0:
        # Vid1 started before Vid2
        ss1 = offset_sec + padding
        ss2 = padding
    else:
        # Vid2 started before Vid1
        ss1 = padding
        ss2 = abs(offset_sec) + padding

    # FFmpeg command: scales both to 720p height, places side-by-side, mixes audio
    filter_complex = (
        f"[0:v]scale=-1:720,fps={OUTPUT_FPS},setpts=PTS-STARTPTS[left];"
        f"[1:v]scale=-1:720,fps={OUTPUT_FPS},setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2:shortest=1[v];"
        "[0:a][1:a]amix=inputs=2:duration=shortest[a]"
    )

    cmd = [FFMPEG_EXE, "-y", "-ss", f"{ss1:.4f}"]
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend([
        "-i", str(vid1), "-ss", f"{ss2:.4f}",
    ])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend([
        "-i", str(vid2),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-shortest",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-r", str(OUTPUT_FPS),
        "-c:a", "aac",
        str(out_path)
    ])

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def get_trial_name(current_dir, base_path):
    relative_parts = current_dir.relative_to(base_path).parts
    return relative_parts[0] if relative_parts else current_dir.name


def folder_outputs_missing(current_dir, base_path):
    """Return whether the expected outputs for a previously processed folder are missing."""
    sync_info_path = current_dir / "sync_info.json"
    folder_log_path = next(
        (path for path in (current_dir / "sync_log.txt", current_dir / "sync_log_folder.txt") if path.is_file()),
        None,
    )

    if not sync_info_path.is_file() or folder_log_path is None:
        return True

    try:
        with open(sync_info_path, "r", encoding="utf-8") as f:
            status = json.load(f).get("status")
    except (OSError, json.JSONDecodeError):
        return True

    required_outputs = [sync_info_path, folder_log_path]
    if status in {"SUCCESS", "LOW_CONFIDENCE"}:
        trial_name = get_trial_name(current_dir, base_path)
        full_video_path = current_dir / f"{trial_name}_sync_full.mp4"
        required_outputs.extend([
            full_video_path,
        ])

    return any(not path.is_file() for path in required_outputs)


def process_folder_tree(base_dir):
    """
    Recursively scans base_dir for target folders containing 2 video files.
    """
    base_path = Path(base_dir).resolve()
    print(f"Scanning base directory: {base_path}\n{'='*70}")

    processed_count = 0
    main_log_path = base_path / "sync_log.txt"
    with open(main_log_path, "w", encoding="utf-8") as main_log:
        main_log.write(f"FlowAnalytics synchronization log\nBase directory: {base_path}\n\n")

    for root, _, files in os.walk(base_path):
        current_dir = Path(root)
        
        # Identify video files in current folder
        video_files = sorted([
            current_dir / f for f in files 
            if Path(f).suffix in VIDEO_EXTENSIONS
            and not f.startswith("._")
            and not f.lower().startswith(("sync_", "preview_sync"))
            and "synced" not in f.lower()
        ])

        # Target folders containing exactly two GoPro/camera recordings
        if len(video_files) == 2:
            if not folder_outputs_missing(current_dir, base_path):
                print(f"Skipping complete folder: {current_dir.relative_to(base_path)}")
                continue

            processed_count += 1
            vid1, vid2 = video_files[0], video_files[1]
            
            print(f"[{processed_count}] Processing folder: {current_dir.relative_to(base_path)}")
            print(f"    - File 1 (Ref): {vid1.name}")
            print(f"    - File 2:       {vid2.name}")

            try:
                # 1. Compute alignment
                offset, conf, status = compute_audio_offset(vid1, vid2)
                print(f"    - Calculated Offset: {offset:+.4f} s (Confidence: {conf:.2f} | Status: {status})")

                # 2. Prepare metadata
                trim_instructions = {}
                if offset >= 0:
                    trim_instructions["video_1_trim_start"] = round(offset, 4)
                    trim_instructions["video_2_trim_start"] = 0.0
                    explanation = f"'{vid1.name}' started {offset:.4f}s before '{vid2.name}'."
                else:
                    trim_instructions["video_1_trim_start"] = 0.0
                    trim_instructions["video_2_trim_start"] = round(abs(offset), 4)
                    explanation = f"'{vid2.name}' started {abs(offset):.4f}s before '{vid1.name}'."

                sync_data = {
                    "status": status,
                    "offset_seconds": round(offset, 4),
                    "confidence_score": round(conf, 2),
                    "explanation": explanation,
                    "reference_video": vid1.name,
                    "target_video": vid2.name,
                    "trim_parameters": trim_instructions,
                    "ffmpeg_sync_example": (
                        f"ffmpeg -ss {trim_instructions['video_1_trim_start']} -i '{vid1.name}' ... AND "
                        f"ffmpeg -ss {trim_instructions['video_2_trim_start']} -i '{vid2.name}' ..."
                    )
                }

                # 3. Create a synchronized video when alignment produced a usable result
                full_video_created = False
                trial_name = get_trial_name(current_dir, base_path)
                full_video_path = current_dir / f"{trial_name}_sync_full.mp4"
                if status in {"SUCCESS", "LOW_CONFIDENCE"}:
                    full_video_created = create_synced_video(vid1, vid2, offset, full_video_path)
                    if full_video_created:
                        print(f"    Full synchronized video generated: {full_video_path.name}")
                
                sync_data["full_video_generated"] = full_video_created
                sync_data["full_video_name"] = full_video_path.name if status in {"SUCCESS", "LOW_CONFIDENCE"} else None

                # 4. Save sync_info.json document in target folder
                json_path = current_dir / "sync_info.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(sync_data, f, indent=4)
                print(f"    Recorded sync document: {json_path.name}\n")

                folder_log_name = "sync_log_folder.txt" if current_dir == base_path else "sync_log.txt"
                log_path = current_dir / folder_log_name
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"Reference video: {vid1.name}\n")
                    f.write(f"Target video: {vid2.name}\n")
                    f.write(f"Status: {status}\n")
                    f.write(f"Offset seconds: {offset:+.4f}\n")
                    f.write(f"Confidence score: {conf:.2f}\n")
                    f.write(f"Full synchronized video: {'generated' if full_video_created else 'not generated'}\n")
                    f.write(f"Full video path: {full_video_path.name if status in {'SUCCESS', 'LOW_CONFIDENCE'} else 'N/A'}\n")
                print(f"    Recorded sync log: {log_path.name}\n")

                with open(main_log_path, "a", encoding="utf-8") as main_log:
                    main_log.write(f"Folder: {current_dir.relative_to(base_path)}\n")
                    main_log.write(f"Reference video: {vid1.name}\n")
                    main_log.write(f"Target video: {vid2.name}\n")
                    main_log.write(f"Status: {status}\n")
                    main_log.write(f"Offset seconds: {offset:+.4f}\n")
                    main_log.write(f"Confidence score: {conf:.2f}\n")
                    main_log.write(f"Full synchronized video: {'generated' if full_video_created else 'not generated'}\n")
                    main_log.write("\n")

            except Exception as e:
                print(f"    ❌ Error processing folder: {type(e).__name__}: {e}\n")
                with open(main_log_path, "a", encoding="utf-8") as main_log:
                    main_log.write(f"Folder: {current_dir.relative_to(base_path)}\n")
                    main_log.write(f"Reference video: {vid1.name}\n")
                    main_log.write(f"Target video: {vid2.name}\n")
                    main_log.write(f"Status: ERROR ({type(e).__name__}: {e})\n\n")

    print(f"{'='*70}\nFinished! Processed {processed_count} paired trial folders.")


if __name__ == "__main__":
    target_directory = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get(
            "FLOW_ANALYTICS_DATA_ROOT",
            r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data",
        )
    )

    if not os.path.isdir(target_directory):
        print(f"Error: Directory '{target_directory}' does not exist.")
        sys.exit(1)

    process_folder_tree(target_directory)