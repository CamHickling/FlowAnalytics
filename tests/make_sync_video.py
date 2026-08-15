import subprocess
import os
import imageio_ffmpeg

def create_synced_video(gopro1, gopro2, offset, output_file):
    """
    Trims the first video by the offset, scales both to 960p wide, 
    and stacks them side-by-side horizontally.
    """
    print(f"Initializing standalone FFmpeg...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # The FFmpeg complex filter:
    # 1. Scales GoPro 1 to 960px wide (maintaining aspect ratio)
    # 2. Scales GoPro 2 to 960px wide (maintaining aspect ratio)
    # 3. Stacks them horizontally (hstack)
    filter_complex = "[0:v]scale=960:-1[v0];[1:v]scale=960:-1[v1];[v0][v1]hstack=inputs=2[vout]"

    command = [
        ffmpeg_exe,
        '-y',                 # Overwrite output without asking
        '-ss', str(offset),   # Fast and frame-accurate seek to trim GoPro 1
        '-i', gopro1,         # Input 0 (GoPro 1 - Front)
        '-i', gopro2,         # Input 1 (GoPro 2 - Side, starts at 0)
        '-filter_complex', filter_complex,
        '-map', '[vout]',     # Map the visual output from the filter
        '-map', '0:a',        # Keep the audio from GoPro 1 (which is now perfectly synced)
        '-c:v', 'libx264',    # Video codec
        '-crf', '23',         # Quality (lower is better, 23 is default balanced)
        '-c:a', 'aac',        # Audio codec
        '-shortest',          # Stop encoding when the shortest video ends
        output_file
    ]

    print(f"Generating side-by-side video: {output_file}")
    print("This may take a few minutes depending on the video length...")

    # Run FFmpeg and stream the output to the console so you can see the progress
    result = subprocess.run(command)

    if result.returncode == 0:
        print("\n" + "=" * 40)
        print(f"Success! Synced video saved to: {output_file}")
        print("=" * 40 + "\n")
    else:
        print("\nFFmpeg encountered an error. Please check the console output above.")

if __name__ == "__main__":
    gopro1_path = r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737\gopro_footage\P01_front.MP4"
    gopro2_path = r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737\gopro_footage\P01_side.MP4"
    
    # Set the calculated offset
    sync_offset = 14.4305
    
    # Define output file in the same directory
    output_path = r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data\P01_20260223_112737\gopro_footage\P01_synced_review.mp4"

    create_synced_video(gopro1_path, gopro2_path, sync_offset, output_path)