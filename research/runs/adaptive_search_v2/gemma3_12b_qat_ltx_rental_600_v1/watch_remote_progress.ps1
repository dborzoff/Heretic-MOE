$Host.UI.RawUI.WindowTitle = "Codex | Gemma3-12B QAT | Heretic-MOE 600"
$ErrorActionPreference = "Continue"

$sshKey = "C:\Users\borzo\.ssh\id_ed25519_vast"
$remote = "root@154.7.90.1"
$port = 46804
$log = "/workspace/heretic-gemma3/runs/gemma3_12b_qat_ltx_rental_600_v1/adaptive-600-supervisor.log"

while ($true) {
    Clear-Host
    Write-Host "Heretic-MOE | Gemma3-12B QAT | live progress" -ForegroundColor Cyan
    Write-Host "Close this window to stop monitoring. The server run will continue." -ForegroundColor DarkGray
    Write-Host ""
    & ssh -i $sshKey -p $port -o BatchMode=yes -o ConnectTimeout=15 $remote "tail -n 100 -F '$log'"
    Write-Host ""
    Write-Host "SSH disconnected; reconnecting in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
