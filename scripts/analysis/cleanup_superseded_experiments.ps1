param([ValidateSet('prepare','run')][string]$Action='prepare')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')).TrimEnd('\')
$report=Join-Path $root 'results/cleanup_20260914_ppt'
$planPath=Join-Path $report 'plan.json'
$receiptPath=Join-Path $report 'receipt.json'
$finalDeck=Join-Path $root 'PPT_汇报材料/成品/近期工作汇报_EmoMM-Sequence_MTG查询对齐版.pptx'

function ResolveScoped([string]$path){
 $absolute=[IO.Path]::GetFullPath($path)
 if(-not $absolute.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside project: $absolute"}
 if($absolute -eq $root){throw 'Workspace root cannot be a cleanup target'}
 $item=Get-Item -LiteralPath $absolute -Force
 if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse point rejected: $absolute"}
 if($item.PSIsContainer){
  $links=@(Get-ChildItem -LiteralPath $absolute -Recurse -Force | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint})
  if($links.Count){throw "Nested reparse point rejected: $absolute"}
 }
 return $absolute
}
function ProtectedFiles{
 $items=[Collections.Generic.List[string]]::new()
 $items.Add((Join-Path $root 'configs/system.yaml'))
 $items.Add((Join-Path $root 'artifacts/features/clamp3/clean/mtg_factor_supervision_mapped.npz'))
 $items.Add((Join-Path $root 'results/diagnosis_20260914_tower_freeze/query_form_embeddings.npz'))
 $locked=Get-Content (Join-Path $root 'results/mtg_mapped_final_test_20260914/plan.json') -Raw -Encoding utf8 | ConvertFrom-Json
 foreach($property in $locked.checkpoints.PSObject.Properties){foreach($row in $property.Value){$items.Add((Join-Path $root $row.path))}}
 foreach($file in (Get-ChildItem (Join-Path $root 'artifacts/releases/retraining_20260911') -Recurse -File | Where-Object {$_.Extension -eq '.pt'})){$items.Add($file.FullName)}
 return @($items | Sort-Object -Unique | ForEach-Object {[pscustomobject]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash}})
}

if($Action -eq 'prepare'){
 if(Test-Path -LiteralPath $planPath){throw 'Cleanup plan already exists'}
 $targets=[Collections.Generic.List[object]]::new()
 function AddTarget([string]$path,[string]$reason){
  if(-not(Test-Path -LiteralPath $path)){return}
  $absolute=ResolveScoped $path;$item=Get-Item -LiteralPath $absolute
  $files=if($item.PSIsContainer){@(Get-ChildItem -LiteralPath $absolute -Recurse -File)}else{@($item)}
  $targets.Add([pscustomobject]@{path=$absolute;kind=$(if($item.PSIsContainer){'directory'}else{'file'});reason=$reason;files=$files.Count;bytes=[long](($files|Measure-Object Length -Sum).Sum);sha256=$(if(-not $item.PSIsContainer){(Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash}else{$null})})
 }
 foreach($file in Get-ChildItem (Join-Path $root 'scripts/training') -Filter 'run_optimization*.py'){
  AddTarget $file.FullName 'retired loss-search driver; its experiment outputs were previously removed'
 }
 foreach($name in @('evaluate_optimized_final.py','evaluate_tag_blends.py','evaluate_semantic_blends.py')){
  AddTarget (Join-Path $root ('scripts/evaluation/'+$name)) 'retired optimization/blend evaluation driver'
 }
 foreach($variant in @('both_frozen','both_train','music_only','text_only')){
  foreach($file in Get-ChildItem (Join-Path $root ('artifacts/runs/diagnosis_20260914_tower_freeze/'+$variant)) -Recurse -Filter '*.pt'){
   AddTarget $file.FullName 'finished tower diagnostic weight; metrics and training recipe retained'
  }
 }
 foreach($variant in @('legacy_only','canonical_only','mapped_plus_qwen')){
  foreach($file in Get-ChildItem (Join-Path $root ('artifacts/runs/diagnosis_20260914_query_alignment/'+$variant)) -Recurse -Filter '*.pt'){
   AddTarget $file.FullName 'unselected query-alignment weight; validation metrics retained'
  }
 }
 foreach($dir in @('scripts/training/__pycache__','scripts/evaluation/__pycache__')){
  if(Test-Path (Join-Path $root $dir)){
   foreach($file in Get-ChildItem (Join-Path $root $dir) -File | Where-Object {$_.Name -match '^(run_optimization|evaluate_optimized_final|evaluate_tag_blends|evaluate_semantic_blends)'}){
    AddTarget $file.FullName 'bytecode for a retired experiment driver'
   }
  }
 }
 AddTarget (Join-Path $root 'PPT_汇报材料/构建过程/retraining_20260911/browser-profile') 'temporary browser profile created for historical PPT screenshots'
 foreach($path in @('PPT_汇报材料/构建过程/v2','PPT_汇报材料/构建过程/draft-render','PPT_汇报材料/构建过程/ablation_20260913/layout_v1_render','PPT_汇报材料/构建过程/ablation_20260913/layout_v2_render')){
  AddTarget (Join-Path $root $path) 'superseded presentation draft and preview files'
 }
 foreach($file in Get-ChildItem (Join-Path $root 'PPT_汇报材料/构建过程') -Recurse -File -Filter '*.pptx'){
  if($file.FullName -like '*\mtg_20260914\*' -or $file.FullName -like '*\v2\*'){continue}
  AddTarget $file.FullName 'superseded PPT build intermediate; source template and historical finished decks retained'
 }
 New-Item -ItemType Directory -Force -Path $report | Out-Null
 $plan=[pscustomobject]@{prepared_utc=[datetime]::UtcNow.ToString('o');root=$root;recovery='Windows Recycle Bin';scope='retired experiment drivers, unselected diagnostic weights, and superseded PPT build files';protected_files=@(ProtectedFiles);targets=@($targets);total_files=($targets|Measure-Object files -Sum).Sum;total_bytes=($targets|Measure-Object bytes -Sum).Sum;retained='Final selected model, all final-test baselines, current deployment, data, summaries including negative results, and original templates'}
 $plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding utf8
 [pscustomobject]@{targets=$targets.Count;files=$plan.total_files;MB=[math]::Round($plan.total_bytes/1MB,1);plan=$planPath} | ConvertTo-Json
 exit 0
}

if(Test-Path -LiteralPath $receiptPath){throw 'Cleanup already executed'}
if(-not(Test-Path -LiteralPath $finalDeck)){throw 'Validated final PPT must exist before cleanup'}
$qaPath=Join-Path $root 'PPT_汇报材料/构建过程/mtg_20260914/verification.json'
if(-not(Test-Path -LiteralPath $qaPath)){throw 'Presentation verification is required'}
$qa=Get-Content $qaPath -Raw -Encoding utf8 | ConvertFrom-Json
if(-not $qa.passed){throw 'Presentation verification did not pass'}
if((Get-FileHash -LiteralPath $finalDeck -Algorithm SHA256).Hash -ne $qa.sha256){throw 'Final presentation changed after verification'}
$plan=Get-Content -LiteralPath $planPath -Raw -Encoding utf8 | ConvertFrom-Json
$paths=@()
foreach($target in $plan.targets){
 $absolute=ResolveScoped $target.path
 if($target.kind -eq 'file' -and (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash -ne $target.sha256){throw "Target changed after plan: $absolute"}
 if($target.kind -eq 'directory'){
  $currentFiles=@(Get-ChildItem -LiteralPath $absolute -Recurse -File)
  $currentBytes=[long](($currentFiles | Measure-Object Length -Sum).Sum)
  if($currentFiles.Count -ne $target.files -or $currentBytes -ne $target.bytes){throw "Directory changed after plan: $absolute"}
 }
 foreach($protected in $plan.protected_files){if($protected.path -eq $absolute -or $protected.path.StartsWith($absolute+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Protected dependency in cleanup target: $absolute"}}
 $paths+=,$absolute
}
foreach($protected in $plan.protected_files){if((Get-FileHash -LiteralPath $protected.path -Algorithm SHA256).Hash -ne $protected.sha256){throw "Protected file changed: $($protected.path)"}}
Add-Type -AssemblyName Microsoft.VisualBasic
$deleted=[Collections.Generic.List[object]]::new()
foreach($target in $plan.targets){
 if($target.kind -eq 'directory'){
  [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($target.path,[Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,[Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,[Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
 }else{
  [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($target.path,[Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,[Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,[Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
 }
 if(Test-Path -LiteralPath $target.path){throw "Removal failed: $($target.path)"}
 $deleted.Add($target)
 [pscustomobject]@{completed_utc=[datetime]::UtcNow.ToString('o');status='in_progress';recovery='Windows Recycle Bin';removed=@($deleted)} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $report 'progress.json') -Encoding utf8
}
foreach($protected in $plan.protected_files){if((Get-FileHash -LiteralPath $protected.path -Algorithm SHA256).Hash -ne $protected.sha256){throw "Protected file changed during cleanup: $($protected.path)"}}
$receipt=[pscustomobject]@{completed_utc=[datetime]::UtcNow.ToString('o');status='complete';recovery='Windows Recycle Bin';targets=$deleted.Count;files=$plan.total_files;bytes=$plan.total_bytes;protected_hashes_verified=$true;removed=@($deleted)}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{status='complete';targets=$deleted.Count;files=$plan.total_files;MB=[math]::Round($plan.total_bytes/1MB,1);recovery='Windows Recycle Bin';receipt=$receiptPath} | ConvertTo-Json
