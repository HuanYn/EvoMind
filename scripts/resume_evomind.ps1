$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$existing = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($repoRoot) -and
    $_.CommandLine -match 'evomind_(run|continue|harness_eval)\.py'
}
if ($existing) { throw 'Existing evomind process found; refusing duplicate launch.' }
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$launchRoot = Join-Path $repoRoot "artifacts\runs\recovery_$runStamp"
New-Item -ItemType Directory -Path $launchRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot 'artifacts\runs\text_official_mini_20260908\state.json') -Destination (Join-Path $launchRoot 'supervisor_state_before.json')
Copy-Item -LiteralPath (Join-Path $repoRoot 'artifacts\evaluation\text_product_20260909\state.json') -Destination (Join-Path $launchRoot 'evaluation_state_before.json')
# Set only this launch environment, never the user's persistent/global settings.
$previousEncoding = $env:PYTHONIOENCODING
$previousUTF8 = $env:PYTHONUTF8
$previousHF = $env:HF_HOME
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
    $env:HF_HOME = Join-Path $repoRoot '.cache\huggingface'
    $process = Start-Process -FilePath (Join-Path $repoRoot '.venv\Scripts\python.exe') -ArgumentList @('-X','utf8','-B','-u','scripts/evomind_run.py','--manifest','configs/text_official_mini.json','--resume') -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $launchRoot 'stdout.log') -RedirectStandardError (Join-Path $launchRoot 'stderr.log') -PassThru
    Write-Output "Recovery supervisor PID: $($process.Id)"
    Write-Output "Logs: $launchRoot"
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
    $env:PYTHONUTF8 = $previousUTF8
    $env:HF_HOME = $previousHF
}
