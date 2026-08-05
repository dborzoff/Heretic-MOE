$ErrorActionPreference = "Stop"
$key = "C:\Users\borzo\.ssh\id_ed25519_vast"
$hostName = "154.7.90.1"
$port = 46804
$log = "/workspace/heretic-gemma3/runs/gemma3_12b_qat_qwen_night_abs_600_v1/adaptive-600-supervisor.log"
$Host.UI.RawUI.WindowTitle = "CODEX | Gemma3 Qwen-night abs PPL | 120 -> 600"

Write-Host "CODEX | GEMMA3 12B | Qwen-night + abs(PPL) | trials 457 -> 657" -ForegroundColor Cyan
Write-Host "Two objectives: refusal geometry + absolute PPL drift; calibrated finalist cost." -ForegroundColor Gray
Write-Host "LTX is not loaded. Following one persistent stream without redraw." -ForegroundColor DarkGray
ssh -i $key -p $port -o StrictHostKeyChecking=no "root@$hostName" "supervisorctl status heretic-gemma3-qwen-night-abs-600; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader; stdbuf -oL tail -n 120 -F '$log' | sed -u -E 's/, LTX[[:space:]]*$//; /conditioning drift:/d'"
