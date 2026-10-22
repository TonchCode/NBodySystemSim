@echo off
rem Nsight Compute profile of the main force kernels (all-pairs #1, trees #4 / #5 / #7).
rem Run from an ADMINISTRATOR prompt (right-click -> Run as administrator), PC otherwise idle:
rem     tools\ncu_profile.bat
rem Writes results\2026-10-22_ncu\*.ncu-repz (kept local: reports can embed the process
rem environment) and *.csv metric exports (committed). Takes a few minutes.
rem ncu locks the SM clock to base by default (--clock-control base) and flushes caches
rem between replay passes (--cache-control all), so its times are NOT comparable with the bench.
setlocal
set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe"
pushd "%~dp0.."
set OUT=results\2026-10-22_ncu
if not exist %OUT% mkdir %OUT%
set B=build\nbody_bench.exe
set COMMON=--set full --import-source no --launch-skip 2 --launch-count 1 -f

"%NCU%" %COMMON% --kernel-name regex:allPairsTiled -o %OUT%\ap_tiled_65536 %B% --kernel allpairs_tiled --n 65536 --block 256 --runs 3 --no-validate > %OUT%\ap_tiled_65536.log 2>&1
"%NCU%" %COMMON% --kernel-name regex:bhTraverseKernel -o %OUT%\bh_naive_1M %B% --kernel bh_naive --n 1048576 --ic plummer --theta 0.5 --leaf 32 --block 128 --runs 3 --no-validate > %OUT%\bh_naive_1M.log 2>&1
"%NCU%" %COMMON% --kernel-name regex:bhTraverseKernel -o %OUT%\bh_sorted_1M %B% --kernel bh_sorted --n 1048576 --ic plummer --theta 0.5 --leaf 32 --block 256 --runs 3 --no-validate > %OUT%\bh_sorted_1M.log 2>&1
"%NCU%" %COMMON% --kernel-name regex:bhWarpKernel -o %OUT%\bh_warp_1M %B% --kernel bh_warp --n 1048576 --ic plummer --theta 0.5 --leaf 16 --block 64 --runs 3 --no-validate > %OUT%\bh_warp_1M.log 2>&1

for %%f in (ap_tiled_65536 bh_naive_1M bh_sorted_1M bh_warp_1M) do (
  "%NCU%" --import %OUT%\%%f.ncu-repz --page details --csv > %OUT%\%%f_details.csv
  "%NCU%" --import %OUT%\%%f.ncu-repz --page raw --csv > %OUT%\%%f_raw.csv
)
echo done > %OUT%\DONE.txt
popd
echo Finished. Results in results\2026-10-22_ncu
