import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import imageio_ffmpeg

def extract_audio(video_file, audio_file):
    """
    Extracts a mono audio track from the video at a standard 44.1kHz sample rate
    using a guaranteed standalone FFmpeg binary.
    """
    print(f"Extracting audio from {video_file}...")
    
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Cannot find the input video: {video_file}")

    # Get the path to the working, standalone FFmpeg executable
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,         # Use the bulletproof binary
        '-y',               # Overwrite output files without asking
        '-i', video_file,   # Input file
        '-vn',              # Disable video stream
        '-ac', '1',         # Downmix to mono
        '-ar', '44100',     # Standardize sample rate to 44.1 kHz
        '-f', 'wav',        # WAV container
        audio_file
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    if result.returncode != 0:
        print("\n" + "=" * 20 + " FFmpeg Full Output " + "=" * 20)
        print(result.stdout)
        print("=" * 60 + "\n")
        raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}")

def find_audio_offset(audio_file1, audio_file2):
    """
    Finds the time offset between two audio files using cross-correlation.
    """
    print("Reading audio arrays...")
    rate1, data1 = wavfile.read(audio_file1)
    rate2, data2 = wavfile.read(audio_file2)

    if rate1 != rate2:
        raise ValueError("Sample rates do not match!")

    # Normalize audio signals
    data1 = data1.astype(np.float32) / (np.max(np.abs(data1)) + 1e-7)
    data2 = data2.astype(np.float32) / (np.max(np.abs(data2)) + 1e-7)

    print("Running FFT cross-correlation...")
    correlation = signal.correlate(data1, data2, mode='full', method='fft')
    
    lag_index = np.argmax(correlation)
    offset_samples = lag_index - (len(data2) - 1)
    offset_seconds = offset_samples / rate1
    
    return offset_seconds

if __name__ == "__main__":
    # Specify the full paths to your GoPro videos
    gopro1_path = r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737\gopro_footage\P01_front.MP4"
    gopro2_path = r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737\gopro_footage\P01_side.MP4" 
    
    temp_audio1 = "temp_audio1.wav"
    temp_audio2 = "temp_audio2.wav"

    try:
        # Step 1: Extract audio tracks
        extract_audio(gopro1_path, temp_audio1)
        extract_audio(gopro2_path, temp_audio2)

        # Step 2: Calculate cross-correlation offset
        offset = find_audio_offset(temp_audio1, temp_audio2)

        print("\n" + "-" * 40)
        if offset > 0:
            print(f"Result: GoPro 2 started {abs(offset):.4f} seconds BEFORE GoPro 1.")
            print(f"Action: Trim {abs(offset):.4f} seconds from the start of GoPro 2.")
        elif offset < 0:
            print(f"Result: GoPro 1 started {abs(offset):.4f} seconds BEFORE GoPro 2.")
            print(f"Action: Trim {abs(offset):.4f} seconds from the start of GoPro 1.")
        else:
            print("Result: Both videos are perfectly synced.")
        print("-" * 40 + "\n")
            
    finally:
        # Step 3: Cleanup temporary audio files
        if os.path.exists(temp_audio1): 
            os.remove(temp_audio1)
        if os.path.exists(temp_audio2): 
            os.remove(temp_audio2)