@echo off
setlocal
cd /d "%~dp0"

if "%WAIFUHAT_WSL_DISTRO%"=="" set "WAIFUHAT_WSL_DISTRO=Ubuntu"
if "%WAIFUHAT_CONFIG%"=="" set "WAIFUHAT_CONFIG=config.toml"
if not exist "%WAIFUHAT_CONFIG%" (
  echo [ERROR] Missing %WAIFUHAT_CONFIG%. Copy config.example.toml to config.toml first.
  pause
  exit /b 1
)

wsl.exe -d "%WAIFUHAT_WSL_DISTRO%" --cd "%CD%" -- bash scripts/project_python.sh scripts/run_with_watchdog.py --config "%WAIFUHAT_CONFIG%" --stall-seconds 600 --max-restarts 2 %*
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo [ERROR] Upscaling ended with exit code %RESULT%.
if "%RESULT%"=="0" echo Upscaling completed successfully.
pause
exit /b %RESULT%
