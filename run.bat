@echo off
rem One-click viewer launcher (D-032). Builds Release first if build\src\viewer\nbody_viewer.exe is missing or
rem older than any file under src\ or the top-level CMakeLists.txt, then starts the demo default:
rem two colliding disk galaxies, bh_warp (R3 #7), N = 262144, colour by origin, load-balance stats on,
rem maximised window.
rem Extra arguments are appended and override the defaults, e.g.
rem   run.bat --ic merger3
rem   run.bat --kernel bh_naive --n 65536 --ic plummer --color mass
setlocal
pushd "%~dp0"
set EXE=build\src\viewer\nbody_viewer.exe
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$e='%EXE%'; if (-not (Test-Path $e)) { exit 1 }; $t=(Get-Item $e).LastWriteTime;" ^
  "$new = Get-ChildItem -Path src -Recurse -File | Where-Object { $_.LastWriteTime -gt $t };" ^
  "if ($new -or (Get-Item CMakeLists.txt).LastWriteTime -gt $t) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo [run.bat] viewer missing or out of date: building Release ...
  call tools\build.bat Release
  if errorlevel 1 (
    echo [run.bat] BUILD FAILED
    popd
    pause
    exit /b 1
  )
)
"%EXE%" --ic galaxies --kernel bh_warp --n 262144 --color origin --lb 1 --dist 6 --maximize 1 %*
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" (
  echo [run.bat] viewer exited with code %RC%
  pause
)
exit /b %RC%
