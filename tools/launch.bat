@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Venv missing. Create it with:
  echo   py -3.12 -m venv .venv
  pause
  exit /b 1
)
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1
REM Hub downloads must land on this drive (D:), not C:\Users\...\.cache\huggingface
set HF_HOME=%CD%\.hf_cache
set HF_HUB_CACHE=%CD%\.hf_cache\hub
set HF_XET_CACHE=%CD%\.hf_cache\xet
REM Windows without Developer Mode cannot create Hub cache symlinks (WinError 1314).
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM Re-apply Windows CUDA JIT workarounds (lost on torch/gsplat reinstall). File I/O only.
".venv\Scripts\python.exe" "%CD%\tools\apply_cuda_jit_workarounds.py"
REM pythonw: desktop window, no extra console. Fall back to python.exe if pythonw is absent.
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m app.main
) else (
  ".venv\Scripts\python.exe" -m app.main
)
endlocal
