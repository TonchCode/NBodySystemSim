# Build, dump and (optionally) run the MUFU-vs-issue timing probe (critic review 1, #2).
#
#   & tools\run_probe.ps1 -OutDir results\2026-10-17_round2\probe -BuildOnly   # SASS + ptxas only
#   & tools\run_probe.ps1 -OutDir results\2026-10-17_round2\probe              # + timed run with clock log
#
# Source: results\2026-10-17_architect_sanity\mufu_probe.cu (round-1 artifact, unchanged, so
# the SASS dumped here is the SASS that produced the round-1 numbers, same nvcc).
# Writes into OutDir:
#   build_ptxas.txt        nvcc -Xptxas -v / --resource-usage output (registers, spills)
#   mufu_probe.sass        cuobjdump -sass of all four probe kernels
#   sass_counts.txt        tools\sass_count.py: opcode counts per kernel, per interaction
#   (run only)  gpu_clock_log.csv  nvidia-smi clocks.sm / power.draw / temp / util every 100 ms
#   (run only)  probe_out.txt      the probe's own output
# No system or driver setting is changed (power limit is only read, in the clock log).
param(
    [string] $OutDir = "results\2026-10-17_round2\probe",
    [switch] $BuildOnly
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $root "results\2026-10-17_architect_sanity\mufu_probe.cu"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
$OutDir = (Resolve-Path $OutDir).Path
$exe  = Join-Path $OutDir "mufu_probe.exe"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -prerelease -latest -products * -property installationPath
if (-not $vs) { throw "Visual Studio not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

# Same flags as the round-1 build line in the probe header, plus resource usage.
# Redirection happens inside cmd: vcvars64 writes a harmless "vswhere.exe not recognized" to
# stderr, which Windows PowerShell 5.1 would turn into a terminating error under "Stop".
$ptxasLog = Join-Path $OutDir "build_ptxas.txt"
$cmd = "call `"$vcvars`" >nul 2>&1 && nvcc -O3 -arch=sm_89 -lineinfo --resource-usage -o `"$exe`" `"$src`" > `"$ptxasLog`" 2>&1"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "probe build failed, see build_ptxas.txt" }
& cuobjdump -sass $exe > (Join-Path $OutDir "mufu_probe.sass")
python (Join-Path $PSScriptRoot "sass_count.py") (Join-Path $OutDir "mufu_probe.sass") --per 128 > (Join-Path $OutDir "sass_counts.txt")
Write-Host "SASS + ptxas written to $OutDir"
if ($BuildOnly) { return }

# Clock / power log in the background for the whole probe run.
$log = Join-Path $OutDir "gpu_clock_log.csv"
# stdout redirection, not "-f": with -f nvidia-smi buffers and Stop-Process kills it before the
# flush (first run on 2026-10-17 produced an empty log).
$smi = Start-Process -FilePath "nvidia-smi" -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -ArgumentList @(
    "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu",
    "--format=csv", "-lms", "100")
Start-Sleep -Milliseconds 1500   # idle baseline in the log
try {
    & $exe | Tee-Object -FilePath (Join-Path $OutDir "probe_out.txt")
} finally {
    Start-Sleep -Milliseconds 1000
    Stop-Process -Id $smi.Id -ErrorAction SilentlyContinue
}
Write-Host "probe output + clock log written to $OutDir"
