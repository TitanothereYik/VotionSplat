@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
where cl
where nvcc
nvcc --version
"%CD%\.venv\Scripts\python.exe" "%CD%\tools\apply_cuda_jit_workarounds.py"
"%CD%\.venv\Scripts\python.exe" "%CD%\tools\env_report.py"
