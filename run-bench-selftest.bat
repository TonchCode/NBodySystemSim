@echo off
rem Builds Release if build\nbody_bench.exe is missing or older than any file under src\ (or CMakeLists.txt),
rem then runs the exact-arithmetic selftest of every kernel variant, the tree checks and the IC regression
rem hashes (nbody_bench --selftest). Last line PASS / FAIL; exit code 0 / 3. Takes a few minutes.
rem Set NBODY_NOPAUSE=1 to skip the final pause (scripts).
setlocal
pushd "%~dp0"
set EXE=build\nbody_bench.exe
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$e='%EXE%'; if (-not (Test-Path $e)) { exit 1 }; $t=(Get-Item $e).LastWriteTime;" ^
  "$new = Get-ChildItem -Path src -Recurse -File | Where-Object { $_.LastWriteTime -gt $t };" ^
  "if ($new -or (Get-Item CMakeLists.txt).LastWriteTime -gt $t) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo [run-bench-selftest.bat] bench missing or out of date: building Release ...
  call tools\build.bat Release
  if errorlevel 1 (
    echo [run-bench-selftest.bat] BUILD FAILED
    popd
    if "%NBODY_NOPAUSE%"=="" pause
    exit /b 1
  )
)
"%EXE%" --selftest %*
set RC=%ERRORLEVEL%
popd
echo [run-bench-selftest.bat] exit code %RC%
if "%NBODY_NOPAUSE%"=="" pause
exit /b %RC%
