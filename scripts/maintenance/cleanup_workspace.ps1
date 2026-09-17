param([switch]$Execute, [switch]$CachesOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$webRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot '../EmoMM-Web'))
$allowedRoots = @($projectRoot, $webRoot)
$targets = [Collections.Generic.List[string]]::new()
function Add-Target([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $resolved = (Resolve-Path -LiteralPath $path).Path.TrimEnd('\')
    $inside = @($allowedRoots | Where-Object { $resolved.StartsWith($_.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase) }).Count -gt 0
    if (-not $inside -or $allowedRoots -contains $resolved) { throw "Unsafe target: $resolved" }
    if ($resolved -match '\\(\.git|node_modules|raw|models)($|\\)' -or $resolved -match 'feedback_events|semantic_db|last\.ckpt') { throw "Protected target: $resolved" }
    if (-not $targets.Contains($resolved)) { $targets.Add($resolved) }
}
if (-not $CachesOnly) {
    foreach ($relative in @('PPT_汇报材料/构建过程','PPT_汇报材料/质量检查','PPT_汇报材料/原始模板','PPT_汇报材料/成品/历史版本','PPT_汇报材料/参考资料/结果快照','README.md','THIRD_PARTY_NOTICES.md','.idea','artifacts/releases/retraining_20260911','artifacts/catalog/semantic.npy','artifacts/catalog/affect.npy','artifacts/catalog/preference.npy','artifacts/catalog/factor_track_ids.npy','artifacts/catalog/metadata.npz')) {
        Add-Target (Join-Path $projectRoot $relative)
    }
    $finals = Join-Path $projectRoot 'PPT_汇报材料/成品'
    foreach ($file in Get-ChildItem -LiteralPath $finals -File -Filter '*.pptx') {
        if ($file.Name -ne '近期工作汇报_EmoMM-Sequence_方法讲解完善版_v9.pptx') { Add-Target $file.FullName }
    }
    Add-Target (Join-Path $webRoot 'README.md')
    Add-Target (Join-Path $webRoot '.idea')
}
foreach ($base in $allowedRoots) {
    $queue = [Collections.Generic.Queue[string]]::new()
    $queue.Enqueue($base)
    while ($queue.Count -gt 0) {
        foreach ($directory in Get-ChildItem -LiteralPath $queue.Dequeue() -Directory -Force) {
            if ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($directory.Name -in @('.git','node_modules','raw','models')) { continue }
            if ($directory.Name -in @('__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.cache')) { Add-Target $directory.FullName }
            else { $queue.Enqueue($directory.FullName) }
        }
    }
}
# Vite's disposable cache only; installed packages are not removed.
$viteCache = Join-Path $webRoot 'frontend/node_modules/.vite'
if (Test-Path -LiteralPath $viteCache) {
    $resolved = (Resolve-Path -LiteralPath $viteCache).Path
    if ($resolved -ne [IO.Path]::GetFullPath($viteCache)) { throw 'Unexpected Vite cache target' }
    $targets.Add($resolved)
}
$ordered = @($targets | Sort-Object Length)
$selected = [Collections.Generic.List[string]]::new()
foreach ($target in $ordered) {
    if (@($selected | Where-Object {$target.StartsWith($_+'\',[StringComparison]::OrdinalIgnoreCase)}).Count -eq 0) { $selected.Add($target) }
}
$items = @($selected | ForEach-Object {
    $item = Get-Item -LiteralPath $_ -Force
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $_ -File -Recurse -Force) } else { @($item) }
    [pscustomobject]@{path=$_;directory=$item.PSIsContainer;files=$files.Count;bytes=($files | Measure-Object Length -Sum).Sum;status='planned'}
})
$report = [ordered]@{time=(Get-Date).ToString('o');mode=$(if($Execute){'recycle_bin'}else{'dry_run'});items=$items;files=($items | Measure-Object files -Sum).Sum;bytes=($items | Measure-Object bytes -Sum).Sum}
$recordDirectory = Join-Path $projectRoot 'results/current_route_20260915'
New-Item -ItemType Directory -Force -Path $recordDirectory | Out-Null
$record = Join-Path $recordDirectory ('cleanup_'+$(if($CachesOnly){'caches'}else{'workspace'})+'_'+$(if($Execute){'executed'}else{'plan'})+'.json')
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $record -Encoding UTF8
if ($Execute) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'NOTICE'))) { throw 'Third-party attribution must be retained in NOTICE first' }
    if (-not $CachesOnly -and -not (Test-Path -LiteralPath (Join-Path $projectRoot 'results/current_route_20260915/web_integration.json'))) { throw 'Verify new web integration before removing old release' }
    Add-Type -AssemblyName Microsoft.VisualBasic
    foreach ($item in $items) {
        if ($item.directory) { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($item.path,'OnlyErrorDialogs','SendToRecycleBin') }
        else { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($item.path,'OnlyErrorDialogs','SendToRecycleBin') }
        $item.status = 'recycled'
        $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $record -Encoding UTF8
    }
}
[pscustomobject]@{mode=$report.mode;files=$report.files;MiB=[math]::Round($report.bytes/1MB,2);targets=$items.Count;record=$record}
