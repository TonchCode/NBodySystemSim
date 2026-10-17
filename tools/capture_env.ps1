# Record the measurement environment (CLAUDE.md rule 2) into a folder. Read-only: changes no
# system, driver or power setting.
#
#   & tools\capture_env.ps1 -OutDir results\<dir> -Tag before
#   ... measurements ...
#   & tools\capture_env.ps1 -OutDir results\<dir> -Tag after
#
# Writes env_<tag>_nvidia-smi-q.txt (full nvidia-smi -q, process list removed) and
# env_<tag>.txt: date, driver, cl version, nvcc version, power limit, clocks, and idle GPU
# utilisation / VRAM sampled 10 x 200 ms. If the median idle utilisation is > 5% the file says
# "UNCONTROLLED BACKGROUND LOAD: X%", and every number from that session must carry the label.
param(
    [Parameter(Mandatory = $true)] [string] $OutDir,
    [string] $Tag = "before"
)
$ErrorActionPreference = "Continue"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
$q   = Join-Path $OutDir "env_${Tag}_nvidia-smi-q.txt"
$txt = Join-Path $OutDir "env_${Tag}.txt"

# Full dump minus the process list (application names are personal data, not needed).
$full = & nvidia-smi -q
$keep = @(); $skip = $false
foreach ($l in $full) {
    if ($l -match '^\s{4}Processes') { $skip = $true; continue }
    if ($skip -and $l -match '^\s{4}\S') { $skip = $false }
    if (-not $skip) { $keep += $l }
}
$keep | Set-Content -Encoding utf8 $q

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -prerelease -latest -products * -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$clv = cmd /c "call `"$vcvars`" >nul 2>&1 && cl 2>&1" | Select-Object -First 1
$nvccv = (& nvcc --version | Select-String "release").Line

$samples = @()
for ($i = 0; $i -lt 10; $i++) {
    $samples += (& nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits)
    Start-Sleep -Milliseconds 200
}
$utils = $samples | ForEach-Object { [int](($_ -split ',')[0].Trim()) } | Sort-Object
$medUtil = $utils[[int][math]::Floor($utils.Count / 2)]
$static = & nvidia-smi --query-gpu=name,driver_version,power.limit,power.default_limit,enforced.power.limit,clocks.max.sm,pstate --format=csv

$out = @(
    "date: $(Get-Date -Format o)",
    "tag: $Tag",
    "cl: $clv",
    "nvcc: $nvccv",
    "nvidia-smi static (csv):",
    $static,
    "idle samples (util %, VRAM MiB, SM MHz, W, C) x10 @200ms:",
    $samples,
    "median idle utilisation: $medUtil %"
)
if ($medUtil -gt 5) { $out += "UNCONTROLLED BACKGROUND LOAD: $medUtil% (label every number from this session)" }
else { $out += "background load: controlled (<= 5%)" }
$out | Set-Content -Encoding utf8 $txt
Get-Content $txt
