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
    [switch]$NoDiarize,
    [string]$PythonExe = $(if ($env:FLOW_ANALYTICS_PYTHON) { $env:FLOW_ANALYTICS_PYTHON } else { "C:/Users/BarlabPRIME/anaconda3/envs/flowhr/python.exe" })
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if (-not $InputDir) { $InputDir = $DataRoot }
if (-not $OutputDir) { $OutputDir = Join-Path $DataRoot "transcripts" }
if (-not $AudioDir) { $AudioDir = Join-Path $DataRoot "audio" }

$scriptPath = Join-Path $repoRoot "scripts/process_videos_whisperx.py"

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
& $PythonExe @arguments
