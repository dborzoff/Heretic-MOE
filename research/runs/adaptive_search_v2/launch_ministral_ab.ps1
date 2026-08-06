$randomScript = Join-Path $PSScriptRoot "ministral_random\run_visible.ps1"
$sobolScript = Join-Path $PSScriptRoot "ministral_sobol\run_visible.ps1"

Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$randomScript`""
)
Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$sobolScript`""
)
