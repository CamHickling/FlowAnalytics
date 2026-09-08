import os
import subprocess
import argparse
import time
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from scipy import signal
import imageio_ffmpeg
import logging
import sys


def extract_audio(video_file, audio_file):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, '-y', '-i', video_file,
        '-vn', '-ac', '1', '-ar', '44100', '-f', 'wav', audio_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_top_audio_offsets(audio_file1, audio_file2, num_peaks=5, max_offset_sec=60, min_dist_sec=10):
    rate1, data1 = wavfile.read(audio_file1)
    rate2, data2 = wavfile.read(audio_file2)

    data1 = data1.astype(np.float32) / (np.max(np.abs(data1)) + 1e-7)
    data2 = data2.astype(np.float32) / (np.max(np.abs(data2)) + 1e-7)

    correlation = signal.correlate(data1, data2, mode='full', method='fft')
    zero_offset_index = len(data2) - 1

    max_samples = int(max_offset_sec * rate1)
    lower_bound = max(0, zero_offset_index - max_samples)
    upper_bound = min(len(correlation), zero_offset_index + max_samples)

    correlation[:lower_bound] = 0
    correlation[upper_bound:] = 0

    min_distance_samples = int(rate1 * min_dist_sec)
    peaks, _ = signal.find_peaks(correlation, distance=min_distance_samples)

    peak_heights = correlation[peaks]
    if len(peak_heights) == 0:
        return []

    top_peak_indices = peaks[np.argsort(peak_heights)[-min(num_peaks, len(peaks)):]][::-1]

    offsets = []
    for lag_index in top_peak_indices:
        offset_samples = lag_index - zero_offset_index
        offset_seconds = offset_samples / rate1
        offsets.append(offset_seconds)

    return offsets


def create_preview_video(gopro1, gopro2, offset, output_file, preview_start_time=30, duration=5):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    filter_complex = "[0:v]scale=640:-1[v0];[1:v]scale=640:-1[v1];[v0][v1]hstack=inputs=2[vout]"

    if offset > 0:
        g1_start = preview_start_time
        g2_start = preview_start_time + offset
    else:
        g1_start = preview_start_time + abs(offset)
        g2_start = preview_start_time

    command = [
        ffmpeg_exe, '-y',
        '-ss', str(g1_start), '-i', gopro1,
        '-ss', str(g2_start), '-i', gopro2,
        '-t', str(duration),
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def create_full_synced_video(gopro1, gopro2, offset, output_file):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    filter_complex = "[0:v]scale=960:-1[v0];[1:v]scale=960:-1[v1];[v0][v1]hstack=inputs=2[vout]"

    if offset > 0:
        command = [
            ffmpeg_exe, '-y',
            '-i', gopro1,
            '-ss', str(offset), '-i', gopro2,
            '-filter_complex', filter_complex,
            '-map', '[vout]', '-map', '0:a',
            '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-shortest',
            output_file
        ]
    else:
        command = [
            ffmpeg_exe, '-y',
            '-ss', str(abs(offset)), '-i', gopro1,
            '-i', gopro2,
            '-filter_complex', filter_complex,
            '-map', '[vout]', '-map', '0:a',
            '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-shortest',
            output_file
        ]

    subprocess.run(command)


def find_gopro_pair(session_dir):
    gopro_dir = os.path.join(session_dir, 'gopro_footage')
    if not os.path.isdir(gopro_dir):
        return None, None

    mp4s = [f for f in os.listdir(gopro_dir) if f.lower().endswith('.mp4')]
    if len(mp4s) < 2:
        return None, None

    # Prefer front and side naming if present
    front = None
    side = None
    for f in mp4s:
        lf = f.lower()
        if 'front' in lf and front is None:
            front = f
        if 'side' in lf and side is None:
            side = f

    if front and side:
        return os.path.join(gopro_dir, front), os.path.join(gopro_dir, side)

    # fallback: first two
    return os.path.join(gopro_dir, mp4s[0]), os.path.join(gopro_dir, mp4s[1])


def get_latest_selection(session_path):
    result_file = os.path.join(session_path, 'sync_results.txt')
    if not os.path.exists(result_file):
        return None
    last_sel = None
    with open(result_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split('\t')
            info = {}
            for part in parts[1:]:
                if '=' in part:
                    k, v = part.split('=', 1)
                    info[k] = v
            if 'option' in info and info['option'] != '0' and 'offset' in info:
                try:
                    offset = float(info['offset'])
                except Exception:
                    continue
                last_sel = offset
    return last_sel


def process_session(session_path, preview_timestamp=60):
    g1, g2 = find_gopro_pair(session_path)
    if not g1 or not g2:
        print(f"Skipping {session_path}: couldn't find two GoPro MP4 files")
        return

    # configure per-session logger
    log_path = os.path.join(session_path, 'sync_progress.log')
    logger = logging.getLogger(session_path)
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s	%(levelname)s	%(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    logger.info(f"Processing session: {session_path}")
    audio1 = os.path.join(session_path, 'temp_audio1.wav')
    audio2 = os.path.join(session_path, 'temp_audio2.wav')

    try:
        logger.info("Extracting audio from videos...")
        extract_audio(g1, audio1)
        extract_audio(g2, audio2)

        logger.info("Computing candidate offsets...")
        offsets = find_top_audio_offsets(audio1, audio2, num_peaks=5, max_offset_sec=60, min_dist_sec=10)
        if not offsets:
            logger.info("No candidate offsets found. Skipping.")
            return

        # create previews in a dedicated subfolder
        previews_dir = os.path.join(session_path, 'previews')
        os.makedirs(previews_dir, exist_ok=True)
        preview_files = []
        for i, offset in enumerate(offsets):
            out_file = os.path.join(previews_dir, f"preview_sync_option_{i+1}.mp4")
            logger.info(f"Creating preview {i+1} with offset {offset:.4f}s -> {out_file}")
            create_preview_video(g1, g2, offset, out_file, preview_timestamp)
            preview_files.append(out_file)

        logger.info("Opening previews folder for preview videos...")
        try:
            os.startfile(previews_dir)
        except Exception:
            logger.info(f"Open folder manually: {previews_dir}")

        # ask user for choice and optional note (interactive only)
        while True:
            choice = input(f"Choose sync option for session '{os.path.basename(session_path)}' (1-{len(preview_files)}), 0 to skip: ")
            if choice.isdigit() and 0 <= int(choice) <= len(preview_files):
                choice = int(choice)
                break
            print("Invalid choice")

        note = ''
        if choice != 0:
            note = input("Optional note for this sync (press Enter to skip): ")
            selected_offset = offsets[choice - 1]
            # do not render final now; record selection for later batch rendering
            result_file = os.path.join(session_path, 'sync_results.txt')
            with open(result_file, 'a', encoding='utf-8') as fh:
                fh.write(f"{datetime.now().isoformat()}\toption={choice}\toffset={selected_offset:.4f}\tvideo={os.path.basename(session_path)}_synced.mp4\tnote={note}\n")
            logger.info(f"Recorded selection (deferred render) to {result_file}")
        else:
            logger.info("Marked session as skipped (no selection recorded).")

    finally:
        # cleanup temporary audio files but keep previews for review
        logger.info("Cleaning up temporary audio files...")
        for f in [audio1, audio2]:
            if os.path.exists(f):
                os.remove(f)
        logger.info("Done with session (previews retained in 'previews/' folder).")


def main(root_dir, preview_timestamp=60):
    if not os.path.isdir(root_dir):
        print(f"Root directory does not exist: {root_dir}")
        return

    # configure aggregated top-level log for all sessions
    batch_log = os.path.join(root_dir, 'batch_sync.log')
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        fh = logging.FileHandler(batch_log, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s'))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(fh)
        root_logger.addHandler(sh)

    sessions = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    sessions.sort()

    for session in sessions:
        process_session(session, preview_timestamp=preview_timestamp)

    # after reviewing all sessions, offer to render all deferred final videos
    print('\nAll sessions processed for preview and selection.')
    while True:
        resp = input('Render final synced videos for all sessions now? (y/n): ').strip().lower()
        if resp in ('y', 'n'):
            break
    if resp == 'y':
        root_logger.info('Starting batch rendering of final synced videos for all sessions...')
        for session in sessions:
            offset = get_latest_selection(session)
            if offset is None:
                root_logger.info(f"Skipping session (no selection recorded): {session}")
                continue
            g1, g2 = find_gopro_pair(session)
            if not g1 or not g2:
                root_logger.info(f"Skipping session (missing GoPro pair): {session}")
                continue
            final_name = os.path.join(session, f"{os.path.basename(session)}_synced.mp4")
            root_logger.info(f"Rendering: {final_name} with offset {offset:.4f}s")
            create_full_synced_video(g1, g2, offset, final_name)
            root_logger.info(f"Rendered: {final_name}")
        root_logger.info('Batch rendering complete.')
    else:
        root_logger.info('Batch rendering deferred by user.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch sync GoPro pairs across sessions')
    parser.add_argument('--root', required=False, help='Root folder with session subfolders',
                        default=os.environ.get(
                            'FLOW_ANALYTICS_DATA_ROOT',
                            r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data",
                        ))
    parser.add_argument('--preview-timestamp', type=int, default=60, help='Preview start time in seconds')
    args = parser.parse_args()

    # enforce interactive-only mode
    if not sys.stdin.isatty():
        print('This script requires an interactive terminal. Run from PowerShell or cmd interactively.')
        sys.exit(1)

    main(args.root, preview_timestamp=args.preview_timestamp)
