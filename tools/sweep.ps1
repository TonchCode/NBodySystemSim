# Sweep nbody_bench over N x kernel x block x (unroll | q x prefetch) and append everything to ONE CSV.
#
#   & tools\sweep.ps1 -Out results\<dir>\sweep.csv
#   & tools\sweep.ps1 -Out x.csv -N 4096,65536 -Kernels allpairs_multi -Blocks 256 -Qs 0,1,4,8 -Prefetch 0,1 -Runs 20 -Batch 50
#   powershell -File tools\sweep.ps1 -Out x.csv -N "4096,65536"      (also works: lists are split on commas)
#
# Round-2 fix (critic review 1, #9): the round-1 logs printed "--n System.Int32[]". Cause:
# PowerShell variable names are case-insensitive, so the loop variable $n WAS the parameter
# $N, which is typed [int[]]; every assignment in the loop was coerced back to a 1-element
# array. The bench still got the right value (a 1-element array splats to one argument), but
# the log was wrong. Loop variables now have distinct names, and every list parameter is
# parsed explicitly (so `powershell -File` with "a,b" strings works too).
#
# Per kernel family the swept knob is:
#   allpairs_tuned  -> -Unrolls (1,4,8,16,0=full)
#   allpairs_multi  -> -Qs (0=auto,1,2,4,8,16,32) x -Prefetch (1 by default; 0,1 to separate it)
#   allpairs_tiled / allpairs_naive -> block only
# CSV columns are written by nbody_bench (header in src/bench/main.cpp): q_requested is what
# was asked (0 = auto), q is what ran. One row per timed run. Validation and the D-009 gates
# are on by default (-NoValidate for pure timing). A gate FAIL (bench exit code 3) is logged
# and the sweep continues; the script exits 3 at the end if any job failed a gate.
# Environment: run tools\capture_env.ps1 before and after. Keep N small enough that one
# launch stays well under 2 s (TDR).
param(
    [string]   $Out      = "results\sweep.csv",
    [string[]] $N        = @("1000", "4096", "10000", "65536"),
    [string[]] $Kernels  = @("allpairs_tiled", "allpairs_tuned", "allpairs_multi"),
    [string[]] $Blocks   = @("64", "128", "256", "512", "1024"),
    [string[]] $Unrolls  = @("1", "4", "8", "16", "0"),
    [string[]] $Qs       = @("0", "1", "2", "4", "8", "16", "32"),
    [string[]] $Prefetch = @("1"),
    [int]      $Runs     = 10,
    [int]      $Warmup   = 3,
    [int]      $WarmupMs = 300,     # time-based warm-up; 0 disables (cold clocks => bimodal small-N timings)
    [int]      $Batch    = 1,       # launches per timed run (use 20+ for N <= ~10k; see bench --batch)
    [string]   $TimelineDir = "",   # if set: also write one --timeline CSV per job into this folder
    [switch]   $NoValidate
)
$ErrorActionPreference = "Stop"

function ConvertTo-IntList([string[]] $values, [string] $name) {
    $out = @()
    foreach ($v in $values) {
        foreach ($part in ($v -split ',')) {
            $t = $part.Trim()
            if ($t -eq '') { continue }
            $parsed = 0
            if (-not [int]::TryParse($t, [ref]$parsed)) { throw "-${name}: '$t' is not an integer" }
            $out += $parsed
        }
    }
    return ,$out
}
function ConvertTo-StrList([string[]] $values) {
    return ,@($values | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
}

$nList  = ConvertTo-IntList $N "N"
$bList  = ConvertTo-IntList $Blocks "Blocks"
$uList  = ConvertTo-IntList $Unrolls "Unrolls"
$qList  = ConvertTo-IntList $Qs "Qs"
$pfList = ConvertTo-IntList $Prefetch "Prefetch"
$kList  = ConvertTo-StrList $Kernels

$root  = Split-Path -Parent $PSScriptRoot
$bench = Join-Path $root "build\nbody_bench.exe"
if (-not (Test-Path $bench)) { throw "nbody_bench.exe not found; run tools\build.bat Release first" }
$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }
if ($TimelineDir -and -not (Test-Path $TimelineDir)) { New-Item -ItemType Directory -Force $TimelineDir | Out-Null }

$common = @("--runs", $Runs, "--warmup", $Warmup, "--warmup-ms", $WarmupMs, "--batch", $Batch, "--csv", $Out)
if ($NoValidate) { $common += "--no-validate" }

$jobs = @()
foreach ($nBodies in $nList) {
    foreach ($kern in $kList) {
        foreach ($blk in $bList) {
            switch ($kern) {
                "allpairs_tuned" { foreach ($u in $uList) { $jobs += ,@("--kernel", $kern, "--n", $nBodies, "--block", $blk, "--unroll", $u) } }
                "allpairs_multi" { foreach ($qq in $qList) { foreach ($pf in $pfList) {
                                       $jobs += ,@("--kernel", $kern, "--n", $nBodies, "--block", $blk, "--q", $qq, "--prefetch", $pf) } } }
                default          { $jobs += ,@("--kernel", $kern, "--n", $nBodies, "--block", $blk) }
            }
        }
    }
}

$i = 0
$gateFails = @()
foreach ($j in $jobs) {
    $i++
    $argv = $j + $common
    if ($TimelineDir) {
        $tag = (($j | Select-Object -Skip 1) -join "_") -replace '--', '' -replace '[^A-Za-z0-9_]', ''
        $argv += @("--timeline", (Join-Path $TimelineDir "tl_$tag.csv"))
    }
    Write-Host ("[{0}/{1}] {2}" -f $i, $jobs.Count, ($j -join " "))
    & $bench @argv | Where-Object { $_ -match "^kernel=|^validate|^# launch|^gates|^timeline" } | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -eq 3) { $gateFails += ($j -join " "); Write-Host "    ** GATE FAIL **"; continue }
    if ($LASTEXITCODE -ne 0) { throw "nbody_bench failed ($LASTEXITCODE) on: $($j -join ' ')" }
}
Write-Host "wrote $Out"
if ($gateFails.Count -gt 0) {
    Write-Host ("{0} job(s) failed a D-009 gate:" -f $gateFails.Count)
    $gateFails | ForEach-Object { Write-Host "    $_" }
    exit 3
}
