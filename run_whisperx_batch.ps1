param(
    [string]$InputDir = "Iris_Recorded_Taekwondo_Data",
    [string]$OutputDir = "Iris_Recorded_Taekwondo_Data/transcripts",
    [string]$AudioDir = "Iris_Recorded_Taekwondo_Data/audio",
    [string]$Model = "large-v3",
    [string]$Device = "cpu",
    [string]$Language = "en",
    [string]$HfToken = $env:HF_TOKEN,
    [switch]$Resume,
    [switch]$NoDiarize
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = "C:/Users/BarlabPRIME/anaconda3/envs/flowhr/python.exe"
$scriptPath = Join-Path $repoRoot "Iris_Recorded_Taekwondo_Data/scripts/process_videos_whisperx.py"

$arguments = @(
    $scriptPath,
    $InputDir,
    "--output-dir", $OutputDir,
    "--audio-dir", $AudioDir,
    "--model", $Model,
    "--device", $Device,
    "--language", $Language,
    "--batch-size", "8"
)

if ($Resume) { $arguments += "--resume" }
if ($NoDiarize) { $arguments += "--no-diarize" }
if ($HfToken) { $arguments += "--hf-token"; $arguments += $HfToken }

Write-Host "Running WhisperX batch transcription..."
& $pythonExe @arguments
