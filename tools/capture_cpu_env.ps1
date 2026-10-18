# R3 #8: record the host CPU state next to tools\capture_env.ps1 (CLAUDE.md rule 2). READ-ONLY: it
# changes no power plan, power mode, screensaver or any other setting; it only reads them.
#
#   & tools\capture_cpu_env.ps1 -OutDir results\<dir> -Tag before
#
# Writes cpu_env_<tag>.txt:
#   - CPU name, cores, logical processors, base clock (Win32_Processor)
#   - active power plan (powercfg /getactivescheme) and the Windows 11 power-mode overlay
#     (HKLM ...\Power\User\PowerSchemes ActiveOverlayAcPowerScheme). The GUID -> name mapping below is the
#     commonly documented one (not verified against a Microsoft page here): ded574b5-... = Best performance,
#     961cc777-... = Best power efficiency, 00000000-... = Balanced (recommended).
#   - idle CPU utilisation: \Processor(_Total)\% Processor Time, 10 samples x 200 ms, median
#   - screensaver: HKCU:\Control Panel\Desktop SCRNSAVE.EXE / ScreenSaveActive / ScreenSaveTimeOut
#     (reported only; the 2026-10-18 clean re-run lost cells to Mystify.scr)
#   - CPU temperature if readable without admin (MSAcpi_ThermalZoneTemperature; usually denied)
param(
    [Parameter(Mandatory = $true)] [string] $OutDir,
    [string] $Tag = "before"
)
$ErrorActionPreference = "Continue"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
$txt = Join-Path $OutDir "cpu_env_${Tag}.txt"
$out = @("date: $(Get-Date -Format o)", "tag: $Tag")

try {
    $p = Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1
    $out += "cpu: $($p.Name) | cores $($p.NumberOfCores) | logical $($p.NumberOfLogicalProcessors) | MaxClockSpeed $($p.MaxClockSpeed) MHz | CurrentClockSpeed $($p.CurrentClockSpeed) MHz | LoadPercentage $($p.LoadPercentage)"
} catch { $out += "cpu: Win32_Processor not readable ($($_.Exception.Message))" }

$scheme = (& powercfg /getactivescheme 2>&1) -join " "
$out += "power plan: $scheme"
try {
    $ov = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" -ErrorAction Stop
    $ac = "$($ov.ActiveOverlayAcPowerScheme)"
    $name = switch -Regex ($ac) {
        "^ded574b5" { "Best performance" }
        "^961cc777" { "Best power efficiency" }
        "^00000000" { "Balanced (recommended)" }
        default     { "unknown / not set" }
    }
    $out += "power mode overlay (AC): '$ac' -> $name"
} catch { $out += "power mode overlay: not readable ($($_.Exception.Message))" }

try {
    $s = Get-Counter -Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop | Out-Null
    $vals = @()
    for ($i = 0; $i -lt 10; $i++) {
        $c = Get-Counter -Counter '\Processor(_Total)\% Processor Time' -ErrorAction Stop
        $vals += [math]::Round($c.CounterSamples[0].CookedValue, 1)
        Start-Sleep -Milliseconds 200
    }
    $sorted = $vals | Sort-Object
    $out += "idle CPU utilisation samples (%): $($vals -join ', ')"
    $out += "median idle CPU utilisation: $($sorted[[int][math]::Floor($sorted.Count / 2)]) %"
} catch { $out += "idle CPU utilisation: performance counter not readable ($($_.Exception.Message))" }

try {
    $d = Get-ItemProperty -Path "HKCU:\Control Panel\Desktop" -ErrorAction Stop
    $scr = if ($d.PSObject.Properties.Name -contains "SCRNSAVE.EXE") { $d.'SCRNSAVE.EXE' } else { "(not set)" }
    $out += "screensaver: SCRNSAVE.EXE=$scr | ScreenSaveActive=$($d.ScreenSaveActive) | ScreenSaveTimeOut=$($d.ScreenSaveTimeOut) s"
    if ($scr -ne "(not set)" -and "$scr" -ne "") { $out += "WARNING: a screensaver is configured (reported, not changed)" }
} catch { $out += "screensaver: not readable ($($_.Exception.Message))" }

try {
    $t = Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop
    foreach ($z in $t) { $out += "thermal zone $($z.InstanceName): $([math]::Round($z.CurrentTemperature / 10 - 273.15, 1)) C" }
} catch { $out += "CPU temperature: not readable without admin ($($_.Exception.Message.Split([Environment]::NewLine)[0]))" }

$out | Set-Content -Encoding utf8 $txt
Get-Content $txt
