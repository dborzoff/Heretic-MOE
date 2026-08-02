$randomScript = "F:\AI\heretic-moe\research\runs\adaptive_search_v2\ministral_random\run_visible.ps1"
$sobolScript = "F:\AI\heretic-moe\research\runs\adaptive_search_v2\ministral_sobol\run_visible.ps1"

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
