import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import process_videos_whisperx as pvw


class ProcessVideosWhisperxTests(unittest.TestCase):
    def test_extract_audio_uses_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            video_path.write_bytes(b"fake-video")
            audio_path = Path(tmpdir) / "sample.wav"

            # The audio-stream check (ffprobe) has its own test below; here only the ffmpeg call is checked
            with patch("subprocess.run") as mock_run, patch("shutil.which", return_value="ffmpeg"),                     patch.object(pvw, "has_audio_stream", return_value=True):
                result = pvw.extract_audio(video_path, audio_path, ffmpeg_bin="ffmpeg")

            self.assertEqual(result, audio_path)
            self.assertEqual(mock_run.call_count, 1)
            args = mock_run.call_args.args[0]
            self.assertEqual(args[:2], ["ffmpeg", "-y"])
            self.assertIn(str(audio_path), args)

    def test_extract_audio_falls_back_to_narration_wav_without_audio_stream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "P01_face_narration.mp4"
            video_path.write_bytes(b"fake-video")
            fallback = Path(tmpdir) / "P01_audio_narration.wav"
            fallback.write_bytes(b"fake-audio")
            audio_path = Path(tmpdir) / "out" / "P01_face_narration.wav"

            with patch("subprocess.run") as mock_run, patch("shutil.which", return_value="ffmpeg"),                     patch.object(pvw, "has_audio_stream", return_value=False):
                result = pvw.extract_audio(video_path, audio_path, ffmpeg_bin="ffmpeg")

            self.assertEqual(result, fallback)
            mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
