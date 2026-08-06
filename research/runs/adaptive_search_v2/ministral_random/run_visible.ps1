$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "CODEX | Adaptive Search v2 | Ministral-3B | Random-to-TPE | physical GPU0 | 120 trials"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$runDirectory = $PSScriptRoot
Set-Location -LiteralPath $runDirectory

Write-Host "Task: Adaptive Search v2 A/B"
Write-Host "Model: Ministral-3B"
Write-Host "Sampler: Random startup -> multivariate TPE"
Write-Host "Physical GPU: 0 (remapped to cuda:0 inside this process)"
Write-Host "Budget: 120 trials (60 startup + 60 TPE)"

Start-Transcript -LiteralPath "$runDirectory\console.log" -Append
try {
    & "F:\AI\heretic_env\Scripts\hereticMOE.exe"
    $runExitCode = $LASTEXITCODE
} finally {
    Stop-Transcript
}
Write-Host "Heretic exit code: $runExitCode"
