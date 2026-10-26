@echo off
setlocal
cd /d "%~dp0"
title Experiment Platform
echo Starting the application from:
echo %CD%
if not exist ".venv-windows\Scripts\python.exe" (
    py -3.11 -m venv .venv-windows
    if errorlevel 1 goto python_error
)
".venv-windows\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3, 11), 'This app requires Python 3.11.'"
if errorlevel 1 goto python_error
".venv-windows\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_error
".venv-windows\Scripts\python.exe" scripts\check_installation.py --smoke-test
if errorlevel 1 goto diagnostic_error
".venv-windows\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto diagnostic_error
exit /b 0
:python_error
echo Install 64-bit Python 3.11 with the Python launcher, then try again.
echo If .venv-windows uses another Python version, rename that folder first.
goto failure
:install_error
echo Dependency installation failed. Check the error above and your package-index access.
goto failure
:diagnostic_error
echo Startup check or application failed. Copy the error text above for troubleshooting.
:failure
pause
exit /b 1
