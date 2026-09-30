param(
    [string]$Python = 'D:\Anaconda3\envs\msmmr\python.exe',
    [string]$ExperimentRoot = '',
    [int]$IntervalSeconds = 1800,
    [string]$TaskName = 'EmoMM-Sequence-Routes-Monitor'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if (-not $ExperimentRoot) { $ExperimentRoot = Join-Path $repoRoot 'artifacts\sequence_routes_v1' }
$experimentPath = [System.IO.Path]::GetFullPath($ExperimentRoot)
$monitorPath = Join-Path $PSScriptRoot 'monitor_sequence_routes.py'
$monitoringPath = Join-Path $experimentPath 'monitoring'
$null = New-Item -ItemType Directory -Path $monitoringPath -Force
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$pythonWindowless = Join-Path (Split-Path -Parent $pythonPath) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonWindowless -PathType Leaf)) { $pythonWindowless = $pythonPath }
if ($IntervalSeconds -lt 60) { throw 'Scheduled checks require an interval of at least 60 seconds.' }
$onceArgs = '"{0}" --once --root "{1}" --interval-seconds {2}' -f $monitorPath, $experimentPath, $IntervalSeconds
$registrationError = $null
try {
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask -and -not ($existingTask.Actions.Arguments -like "*$monitorPath*")) {
        throw "Task name $TaskName is owned by another command; refusing to overwrite it."
    }
    $action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument $onceArgs -WorkingDirectory $repoRoot
    $trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval ([TimeSpan]::FromSeconds($IntervalSeconds))
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Read-only sequence experiment checks. Verifies PID and process creation time; never restarts or kills experiments.'
    $null = Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 3
    $information = Get-ScheduledTaskInfo -TaskName $TaskName
    $installed = [ordered]@{
        mechanism = 'windows_scheduled_task'; task_name = $TaskName; interval_seconds = $IntervalSeconds;
        installed_utc = (Get-Date).ToUniversalTime().ToString('o'); executable = $pythonWindowless;
        arguments = $onceArgs; last_task_result = $information.LastTaskResult;
        next_run_time = $information.NextRunTime.ToString('o');
        persistence = 'Current-user interactive task: checks run while this Windows user is logged in; start-when-available is enabled.'
    }
} catch {
    $registrationError = $_.Exception.Message
    # A hidden local watcher is the fallback when Windows denies registration.
    # Reuse it only after the Python entry point verifies its process identity.
    $watchArgs = '"{0}" --watch --root "{1}" --interval-seconds {2}' -f $monitorPath, $experimentPath, $IntervalSeconds
    $process = Start-Process -FilePath $pythonWindowless -ArgumentList $watchArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $monitoringPath 'watcher.stdout.log') -RedirectStandardError (Join-Path $monitoringPath 'watcher.stderr.log')
    Start-Sleep -Seconds 2
    $process.Refresh()
    if ($process.HasExited) {
        $watcherRecord = Join-Path $monitoringPath 'watcher.json'
        if (-not (Test-Path -LiteralPath $watcherRecord)) { throw 'Local watcher exited without a process record; inspect watcher.stderr.log.' }
        $watcher = Get-Content -LiteralPath $watcherRecord -Raw | ConvertFrom-Json
        $verification = & $pythonPath -c "import importlib.util,json,sys; s=importlib.util.spec_from_file_location('monitor',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.process_identity(int(sys.argv[2]),sys.argv[3])))" $monitorPath $watcher.pid $watcher.created_utc
        if (-not (($verification | ConvertFrom-Json).alive)) { throw 'No verified live local watcher exists; inspect watcher.stderr.log.' }
        $watcherPid = $watcher.pid
        $watcherCreated = $watcher.created_utc
    } else {
        $watcherPid = $process.Id
        $watcherCreated = $process.StartTime.ToUniversalTime().ToString('o')
    }
    $installed = [ordered]@{
        mechanism = 'local_hidden_watcher'; pid = $watcherPid; created_utc = $watcherCreated;
        interval_seconds = $IntervalSeconds; installed_utc = (Get-Date).ToUniversalTime().ToString('o');
        scheduled_task_error = $registrationError; executable = $pythonWindowless; arguments = $watchArgs;
        persistence = 'Only while this process and current Windows session remain alive; not a durable scheduler.'
    }
}
$installed | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $monitoringPath 'installation.json') -Encoding utf8
$installed | ConvertTo-Json -Depth 6
