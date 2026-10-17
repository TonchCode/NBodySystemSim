# Sweep nbody_bench over N x kernel x block x (unroll | q) and append everything to ONE CSV.
#
#   From a PowerShell prompt (array arguments like -N 4096,65536 only bind when the script is
#   invoked with & or .\, NOT through `powershell -File`, which passes them as one string):
#   & tools\sweep.ps1 -Out results\<dir>\sweep.csv
#   & tools\sweep.ps1 -Out x.csv -N 4096,65536 -Kernels allpairs_multi -Blocks 256 -Qs 0,1,4,8 -Runs 20 -Batch 50
#
# Per kernel family the swept knob is:
#   allpairs_tuned  -> -Unrolls (1,4,8,16,0=full)
#   allpairs_multi  -> -Qs      (0=auto,1,2,4,8,16,32)
#   allpairs_tiled / allpairs_naive -> block only
# The CSV columns are written by nbody_bench (kernel,n,block,q,unroll,grid,regs,blocks_per_sm,
# occupancy,run,ms,...). One row per timed run. Validation (vs naive + FP64 sample) is on by
# default; pass -NoValidate for pure timing sweeps.
# Environment (power limit, clocks, background load) is NOT captured here: bench-profiler
# records it around the sweep. Keep N small enough that one launch stays well under 2 s (TDR).
param(
    [string]   $Out      = "results\sweep.csv",
    [int[]]    $N        = @(1000, 4096, 10000, 65536),
    [string[]] $Kernels  = @("allpairs_tiled", "allpairs_tuned", "allpairs_multi"),
    [int[]]    $Blocks   = @(64, 128, 256, 512, 1024),
    [int[]]    $Unrolls  = @(1, 4, 8, 16, 0),
    [int[]]    $Qs       = @(0, 1, 2, 4, 8, 16, 32),
    [int]      $Runs     = 10,
    [int]      $Warmup   = 3,
    [int]      $WarmupMs = 300,     # time-based warm-up; 0 disables (cold clocks => bimodal small-N timings)
    [int]      $Batch    = 1,       # launches per timed run (use 20+ for N <= ~10k; see bench --batch)
    [switch]   $NoValidate
)
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$bench = Join-Path $root "build\nbody_bench.exe"
if (-not (Test-Path $bench)) { throw "nbody_bench.exe not found; run tools\build.bat Release first" }
$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }

$common = @("--runs", $Runs, "--warmup", $Warmup, "--warmup-ms", $WarmupMs, "--batch", $Batch, "--csv", $Out)
if ($NoValidate) { $common += "--no-validate" }

$jobs = @()
foreach ($n in $N | ForEach-Object { [int]$_ }) {
    foreach ($k in $Kernels) {
        foreach ($b in $Blocks) {
            switch ($k) {
                "allpairs_tuned" { foreach ($u in $Unrolls) { $jobs += ,@("--kernel", $k, "--n", $n, "--block", $b, "--unroll", $u) } }
                "allpairs_multi" { foreach ($q in $Qs)      { $jobs += ,@("--kernel", $k, "--n", $n, "--block", $b, "--q", $q) } }
                default          { $jobs += ,@("--kernel", $k, "--n", $n, "--block", $b) }
            }
        }
    }
}

$i = 0
foreach ($j in $jobs) {
    $i++
    $argv = $j + $common
    Write-Host ("[{0}/{1}] {2}" -f $i, $jobs.Count, ($j -join " "))
    & $bench @argv | Where-Object { $_ -match "^kernel=|^validate|^# launch" } | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) { throw "nbody_bench failed ($LASTEXITCODE) on: $($j -join ' ')" }
}
Write-Host "wrote $Out"
