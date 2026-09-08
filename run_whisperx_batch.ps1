param(
    [string]$DataRoot = "D:\FlowAnalytics\Iris_Recorded_Taekwondo_Data",
    [string]$InputDir,
    [string]$OutputDir,
    [string]$AudioDir,
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

if (-not $InputDir) { $InputDir = $DataRoot }
if (-not $OutputDir) { $OutputDir = Join-Path $DataRoot "transcripts" }
if (-not $AudioDir) { $AudioDir = Join-Path $DataRoot "audio" }

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
