@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%CD%\.venv\Scripts;%PATH%"
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1
if not exist "vendor\litegs\litegs\render\__init__.py" (
  echo Cloning LiteGS at pinned SHA (vendor\litegs.sha) ...
  git clone --recurse-submodules https://github.com/MooreThreads/LiteGS.git vendor\litegs
  for /f "usebackq delims=" %%s in ("vendor\litegs.sha") do git -C vendor\litegs checkout %%s
  git -C vendor\litegs submodule update --init --recursive
)
where cl
where nvcc
nvcc --version
".venv\Scripts\python.exe" "%CD%\tools\apply_cuda_jit_workarounds.py"
".venv\Scripts\python.exe" "%CD%\tools\build_litegs.py" %*
endlocal
