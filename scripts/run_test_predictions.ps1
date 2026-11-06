# Public test predictions for the final ensemble, one GPU job at a time; stops on the first failure.
#  1. YOLOv8-M Slicing + flip TTA (raw per-pass boxes saved for WBF)   ~1 h 55
#  2. GFL epoch 38 (best_mva.pth), 2560x1440, per-image BN               ~50 min
# Log: reports/test_run.log ; final line 'all test predictions complete'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv-5070\Scripts\python.exe'
$log = Join-Path $root 'reports\test_run.log'
function Log($t){ Add-Content -LiteralPath $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $t" -Encoding utf8 }
$jobs = @(
  @{name='yolo_slicing_tta_test'; args='-u scripts/run_qos.py scripts/eval_sahi_tta.py --run slicing_m_e30 --mode tta --split test --chunk 500'},
  @{name='gfl_ep38_test';         args='-u scripts/run_qos.py scripts/gfl_predict.py --split test'}
)
foreach($j in $jobs){
  Log "start $($j.name)"
  $out = Join-Path $root "reports\$($j.name).out.log"; $err = Join-Path $root "reports\$($j.name).err.log"
  $p = Start-Process -FilePath $py -ArgumentList $j.args -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru -Wait
  if($p.ExitCode -ne 0){ Log "FAILED $($j.name) (exit $($p.ExitCode)); see $err"; exit 1 }
  Log "done $($j.name): $((Get-Content -LiteralPath $out -Tail 1))"
}
Log 'all test predictions complete'
