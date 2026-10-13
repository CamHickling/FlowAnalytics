# Overnight queue: re-render every MVP review video, verify each one's sync,
# then run the action-unit analysis on the trial face videos.
# Logs go to D:\FlowAnalytics\logs\. Each step starts only after the previous one finishes.

# Optional: -StartFrom P30 resumes the render at that trial (earlier ones are left as they are)
param([string]$StartFrom = "")

$ErrorActionPreference = "Continue"
$root = "D:\FlowAnalytics"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmm"

# mvp.py needs the `capture` env's Python and the `video` env's working ffmpeg
$env:PATH = "C:\Users\BarlabPRIME\anaconda3\envs\video\Library\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = "1"
$capturePy = "C:\Users\BarlabPRIME\anaconda3\envs\capture\python.exe"
$data = Join-Path $root "Iris_Recorded_Taekwondo_Data"

"$(Get-Date) step 1/3: batch re-render of all MVP review videos" | Tee-Object -Append "$logs\queue_$stamp.log"
$startArgs = @(); if ($StartFrom) { $startArgs = @("--start", $StartFrom) }
& $capturePy "$root\scripts\mvp.py" batch-render --data-root $data @startArgs *> "$logs\mvp_render_$stamp.log"
"$(Get-Date) render exit code $LASTEXITCODE" | Tee-Object -Append "$logs\queue_$stamp.log"

"$(Get-Date) step 2/3: verify sync of every render" | Tee-Object -Append "$logs\queue_$stamp.log"
$trials = Get-ChildItem $data -Directory | Where-Object { $_.Name -match '^P\d' } | ForEach-Object { $_.Name }
Push-Location "$root\scripts"
& $capturePy verify_mvp_sync.py @trials *> "$logs\mvp_verify_$stamp.log"
Pop-Location
"$(Get-Date) verify exit code $LASTEXITCODE" | Tee-Object -Append "$logs\queue_$stamp.log"

"$(Get-Date) step 3/3: action-unit analysis" | Tee-Object -Append "$logs\queue_$stamp.log"
Push-Location "$root\facial_analysis"
& ".\.venv\Scripts\python.exe" -m faceanalysis -c config_trials.yaml trials run *> "$logs\au_analysis_$stamp.log"
Pop-Location
"$(Get-Date) AU analysis exit code $LASTEXITCODE; all done" | Tee-Object -Append "$logs\queue_$stamp.log"
