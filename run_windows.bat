@echo off
setlocal
cd /d "%~dp0"
title Runway

rem Find Python 3.11 or newer
set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
  echo Runway needs Python 3.11 or newer. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH", then double-click this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment in .venv ...
  %PY% -m venv .venv || goto fail
)

echo Installing Runway's packages. The first run takes a few minutes ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto fail

if not exist ".env" (
  echo GEMINI_API_KEY=> .env
  echo ELEVENLABS_API_KEY=>> .env
)
echo.
echo Checking connections (put your keys in .env to use Gemini and ElevenLabs):
".venv\Scripts\python.exe" llm.py
".venv\Scripts\python.exe" voice.py
echo.
echo Starting Runway at http://localhost:8501  (close this window to stop it)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:fail
echo.
echo Something went wrong. Scroll up to see the error.
pause
exit /b 1
