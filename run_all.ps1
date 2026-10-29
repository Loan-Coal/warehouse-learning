# run_all.ps1 : Q-table et PPO à 2, 3, 4 robots, puis le tableau final
$ErrorActionPreference = "Continue"
$py    = "python"
$maxMb = 600          # selon ta RAM libre (voir le tableau)
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"            # logs en UTF-8 (sinon UTF-16 et accents cassés sous PS 5.1)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
New-Item -ItemType Directory -Force -Path logs | Out-Null

# File PPO, en arrière-plan
$ppo = Start-Job -ArgumentList $PSScriptRoot, $py -ScriptBlock {
    param($root, $py)
    Set-Location $root
    foreach ($n in 2, 3, 4) {
        & $py -m rl.run --policy ppo --robots $n --train --envs 8 --episodes 10000000 `
            --max-minutes 60 --no-render --print-every 5000 --save "checkpoints/ppo_$n" *>&1 | Out-File -Encoding utf8 "logs/ppo_$n.log"
    }
}

# File Q-table, au premier plan
$minutes = @{2 = 90; 3 = 60; 4 = 60}
$eps     = @{2 = 15000000; 3 = 9000000; 4 = 6000000}
foreach ($n in 2, 3, 4) {
    Write-Host "`n=== Q-table, $n robots ($($minutes[$n]) min) ===" -ForegroundColor Cyan
    & $py -m rl.run --policy central_q --robots $n --train --episodes 10000000 `
        --max-minutes $minutes[$n] --no-render --print-every 5000 `
        --policy-opt "eps_steps=$($eps[$n])" --policy-opt "max_mb=$maxMb" `
        --save "checkpoints/q_$n" 2>&1 | Tee-Object -Variable out | Out-Host; $out | Out-File -Encoding utf8 "logs/q_$n.log"
}

Write-Host "`n=== Attente de la fin des runs PPO ===" -ForegroundColor Cyan
Wait-Job $ppo | Out-Null

Write-Host "`n=== Tableau final ===" -ForegroundColor Cyan
& $py -m rl.qtable_report --robots 2 3 4 --episodes 100 2>&1 | Tee-Object -Variable out | Out-Host; $out | Out-File -Encoding utf8 "logs/report.log"
Write-Host "`nTerminé. Tableau dans results/qtable_table.md" -ForegroundColor Green