@echo off
REM One-click transcription using the capture conda environment.
REM Modify INPUT_DIR and OUTPUT_DIR as needed.
setlocal
rem Edit CONDA_ROOT below if your Anaconda is installed elsewhere
set "CONDA_ROOT=C:\Users\BarlabPRIME\anaconda3"
rem Default to the dataset root so the run covers all participant folders recursively
set "INPUT_DIR=%~dp0\Iris_Recorded_Taekwondo_Data"
set "OUTPUT_DIR=%~dp0\Iris_Recorded_Taekwondo_Data\test_output"
set "SCRIPT=%~dp0\Iris_Recorded_Taekwondo_Data\scripts\process_videos_whisperx.py"

echo Running WhisperX audio-only transcription in capture env...
if exist "%CONDA_ROOT%\condabin\conda.bat" (
    call "%CONDA_ROOT%\condabin\conda.bat" activate capture
) else if exist "%CONDA_ROOT%\Scripts\activate.bat" (
    call "%CONDA_ROOT%\Scripts\activate.bat" capture
) else (
    echo WARNING: could not find conda activation scripts at "%CONDA_ROOT%".
    echo If conda is not on PATH the batch will fail. Edit this file and set CONDA_ROOT to your Anaconda installation.
)

set KMP_DUPLICATE_LIB_OK=TRUE
python "%SCRIPT%" "%INPUT_DIR%" --audio-only --recursive --device cuda --output-dir "%OUTPUT_DIR%" --resume --no-diarize
if errorlevel 1 (
    echo.
    echo ERROR: transcription failed.
    pause
    exit /b 1
)

echo.
echo Completed. Output files are in:
echo %OUTPUT_DIR%
pause
endlocal