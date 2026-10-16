@echo off
rem Configure + build with VS 2026 Insiders (vswhere needs -prerelease to find it), CMake + Ninja.
rem Usage: tools\build.bat [Release|Debug] [extra cmake args...]
rem Pin the older MSVC toolset if a VS update outpaces nvcc: set NBODY_VCVARS_VER=14.44
setlocal
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -prerelease -latest -products * -property installationPath`) do set VSPATH=%%i
if not defined VSPATH (echo Visual Studio not found & exit /b 1)

if defined NBODY_VCVARS_VER (
  call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=%NBODY_VCVARS_VER% >nul
) else (
  call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
)

pushd "%~dp0.."
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% %2 %3 %4 %5 || (popd & exit /b 1)
cmake --build build || (popd & exit /b 1)
popd
