# Waits for run_test_predictions.ps1 to finish, then splits the YOLO TTA test raw file and builds the final WBF submission.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv-5070\Scripts\python.exe'
$log = Join-Path $root 'reports\test_run.log'
function Log($t){ Add-Content -LiteralPath $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $t" -Encoding utf8 }
while($true){
  Start-Sleep -Seconds 60
  $last = Get-Content -LiteralPath $log -Tail 1
  if($last -match 'all test predictions complete$'){ break }
  if($last -match 'FAILED'){ Log 'submission step skipped: a test prediction failed'; exit 1 }
}
foreach($a in @('-X utf8 scripts/export_tta_parts.py reports/sahi_tta/slicing_m_e30_tta_test_raw.npz', '-X utf8 scripts/make_submission.py')){
  Log "start $a"
  $p = Start-Process -FilePath $py -ArgumentList $a -RedirectStandardOutput reports\submission_step.out.log -RedirectStandardError reports\submission_step.err.log -NoNewWindow -PassThru -Wait
  if($p.ExitCode -ne 0){ Log "FAILED $a (exit $($p.ExitCode))"; exit 1 }
  Log "done: $((Get-Content -LiteralPath reports\submission_step.out.log -Tail 1))"
}
Log 'submission ready: reports/submission/final_ensemble_test_predictions.json'
