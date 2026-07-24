@echo off
setlocal
cd /d "%~dp0"

if "%WAIFUHAT_WSL_DISTRO%"=="" set "WAIFUHAT_WSL_DISTRO=Ubuntu"

where wsl.exe >nul 2>nul || (
  echo [ERROR] WSL2 is not installed.
  pause
  exit /b 1
)

echo Installing WaifuHAT2x in %WAIFUHAT_WSL_DISTRO% with ROCm...
wsl.exe -d "%WAIFUHAT_WSL_DISTRO%" --cd "%CD%" -- env WAIFUHAT_SKIP_MODELS=1 bash scripts/install_wsl.sh
if errorlevel 1 (
  echo.
  echo [ERROR] Installation failed. See docs\ROCM_RUNTIME.md and the output above.
  pause
  exit /b 1
)

echo.
echo Downloading HAT weights with aria2 when available...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\download_hats_aria2.ps1"
if errorlevel 1 (
  echo [WARNING] aria2 could not be used; WSL will fall back to gdown.
)

echo.
echo Verifying and extracting official model weights...
wsl.exe -d "%WAIFUHAT_WSL_DISTRO%" --cd "%CD%" -- bash scripts/project_python.sh scripts/download_models.py
if errorlevel 1 (
  echo [ERROR] Model setup failed.
  pause
  exit /b 1
)

echo.
echo Installation succeeded.
pause
