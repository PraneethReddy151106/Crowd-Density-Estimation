@echo off
REM ============================================================================
REM One-Click Launcher for Crowd Density Estimation & People Counting Web App
REM ============================================================================

echo ======================================================================
echo Starting Crowd Density Estimation & People Counting System
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if virtual environment exists, activate if present
if exist "venv\Scripts\activate.bat" (
    echo Activating local virtual environment (venv)...
    call venv\Scripts\activate.bat
)

REM Generate sample assets if not already present
if not exist "sample_data\sample_sparse_crowd.jpg" (
    echo Generating sample validation assets...
    python sample_data_generator.py
)

REM Launch Streamlit web application
echo Launching Streamlit Web Application on http://localhost:8501 ...
python -m streamlit run app.py

pause
