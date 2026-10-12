import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import imageio_ffmpeg

def extract_audio(video_file, audio_file):
    print(f"Extracting audio from {os.path.basename(video_file)}...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, '-y', '-i', video_file,
        '-vn', '-ac', '1', '-ar', '44100', '-f', 'wav', audio_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def find_top_audio_offsets(audio_file1, audio_file2, num_peaks=5, max_offset_sec=60, min_dist_sec=10):
    print(f"Analyzing audio tracks (limiting search to a +/- {max_offset_sec} second window)...")
    rate1, data1 = wavfile.read(audio_file1)
    rate2, data2 = wavfile.read(audio_file2)

    data1 = data1.astype(np.float32) / (np.max(np.abs(data1)) + 1e-7)
    data2 = data2.astype(np.float32) / (np.max(np.abs(data2)) + 1e-7)

    correlation = signal.correlate(data1, data2, mode='full', method='fft')
    
    # Calculate the array index that represents perfect zero offset
    zero_offset_index = len(data2) - 1
    
    # Zero out any correlation data outside our realistic 60-second window
    max_samples = int(max_offset_sec * rate1)
    lower_bound = max(0, zero_offset_index - max_samples)
    upper_bound = min(len(correlation), zero_offset_index + max_samples)
    
    correlation[:lower_bound] = 0
    correlation[upper_bound:] = 0

    # Force the algorithm to space its guesses at least 10 seconds apart
    min_distance_samples = int(rate1 * min_dist_sec)
    peaks, _ = signal.find_peaks(correlation, distance=min_distance_samples)
    
    peak_heights = correlation[peaks]
    top_peak_indices = peaks[np.argsort(peak_heights)[-num_peaks:]][::-1]

    offsets = []
    print("\nCalculated Offsets:")
    for i, lag_index in enumerate(top_peak_indices):
        offset_samples = lag_index - zero_offset_index
        offset_seconds = offset_samples / rate1
        offsets.append(offset_seconds)
        print(f"  Option {i+1}: {offset_seconds:.4f} seconds")
        
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
    print(f"\nStarting full render with offset: {offset:.4f} seconds")
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
    print(f"\nSuccess! Final video saved to: {output_file}")

if __name__ == "__main__":
    gopro1_path = r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737\gopro_footage\P01_front.MP4"
    gopro2_path = r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737\gopro_footage\P01_side.MP4"
    
    output_dir = os.path.dirname(gopro1_path)
    final_output_path = os.path.join(output_dir, "P01_synced_full_performance.mp4")
    
    temp_audio1 = "temp_audio1.wav"
    temp_audio2 = "temp_audio2.wav"
    
    # Adjust this to a timestamp where rapid motion is happening
    preview_timestamp = 60 

    try:
        extract_audio(gopro1_path, temp_audio1)
        extract_audio(gopro2_path, temp_audio2)
        
        # Enforcing a max offset of 60s and forcing guesses to be 10s apart
        top_offsets = find_top_audio_offsets(temp_audio1, temp_audio2, num_peaks=5, max_offset_sec=60, min_dist_sec=10)

        print("\nGenerating 5 distinct preview clips...")
        preview_files = []
        for i, offset in enumerate(top_offsets):
            out_file = f"preview_sync_option_{i+1}.mp4"
            create_preview_video(gopro1_path, gopro2_path, offset, out_file, preview_timestamp)
            preview_files.append(out_file)

        print("\nOpening the folder containing the previews...")
        os.startfile(os.getcwd())

        print("-" * 50)
        print("Please watch the 5 preview videos to see which one perfectly aligns the action.")
        while True:
            choice = input("Which option is perfectly synced? Enter a number (1-5), or 0 to cancel: ")
            
            if choice.isdigit() and 0 <= int(choice) <= 5:
                choice = int(choice)
                break
            print("Invalid input. Please enter a number between 0 and 5.")

        if choice == 0:
            print("Operation cancelled. Exiting...")
        else:
            selected_offset = top_offsets[choice - 1]
            create_full_synced_video(gopro1_path, gopro2_path, selected_offset, final_output_path)
            
    finally:
        print("\nCleaning up temporary files...")
        if os.path.exists(temp_audio1): os.remove(temp_audio1)
        if os.path.exists(temp_audio2): os.remove(temp_audio2)
        for i in range(1, 6):
            preview_file = f"preview_sync_option_{i}.mp4"
            if os.path.exists(preview_file): os.remove(preview_file)