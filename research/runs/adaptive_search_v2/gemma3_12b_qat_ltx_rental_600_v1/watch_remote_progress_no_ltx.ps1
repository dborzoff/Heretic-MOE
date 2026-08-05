$Host.UI.RawUI.WindowTitle = "CODEX | GEMMA3 12B | 600-1000 | NO LTX OBJECTIVE"
$key = "C:\Users\borzo\.ssh\id_ed25519_vast"
$hostName = "154.7.90.1"
$port = 46804
$log = "/workspace/heretic-gemma3/runs/gemma3_12b_qat_no_ltx_1000_v2/adaptive-1000-supervisor.log"

while ($true) {
    Clear-Host
    Write-Host "CODEX | GEMMA3 12B | trials 600 -> 1000 | geometry + absolute PPL" -ForegroundColor Cyan
    Write-Host "LTX is not evaluated during search; historical measurements remain archived." -ForegroundColor DarkGray
    ssh -i $key -p $port -o StrictHostKeyChecking=no "root@$hostName" "supervisorctl status heretic-gemma3-adaptive-1000-no-ltx-v2; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader; tail -n 90 '$log'"
    Start-Sleep -Seconds 10
}
