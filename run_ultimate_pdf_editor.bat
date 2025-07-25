@echo off
title Ultimate PDF Editor - Professional Edition

echo ========================================
echo 🎯 ULTIMATE PDF EDITOR - STARTING UP
echo ========================================
echo.
echo Features:
echo  🖌️ Brush Erasing - Remove unwanted content
echo  📝 Text Editing - Add and style text
echo  🔍 OCR Recognition - Edit detected text
echo.
echo ========================================

:: Check if virtual environment exists
if not exist "..\venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run setup first or ensure venv exists in parent directory.
    pause
    exit /b 1
)

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call ..\venv\Scripts\activate.bat

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in virtual environment!
    pause
    exit /b 1
)

:: Install/Update requirements if needed
echo 📦 Checking dependencies...
pip install -r requirements.txt --quiet

:: Check if required modules exist
echo 🔍 Verifying modules...
if not exist "brush_eraser.py" (
    echo ❌ Missing brush_eraser.py module!
    pause
    exit /b 1
)

if not exist "text_editor.py" (
    echo ❌ Missing text_editor.py module!
    pause
    exit /b 1
)

if not exist "ocr_editor.py" (
    echo ❌ Missing ocr_editor.py module!
    pause
    exit /b 1
)

if not exist "ultimate_pdf_editor_integrated.py" (
    echo ❌ Missing main application file!
    pause
    exit /b 1
)

:: Create directories if they don't exist
if not exist "temp_files" mkdir temp_files
if not exist "downloads" mkdir downloads
if not exist "static" mkdir static
if not exist "templates" mkdir templates

echo ✅ All checks passed!
echo.
echo ========================================
echo 🚀 STARTING ULTIMATE PDF EDITOR
echo ========================================
echo.
echo 🌐 Server will be available at:
echo    http://localhost:8000
echo.
echo 📖 How to use:
echo  1. Open http://localhost:8000 in your browser
echo  2. Upload a PDF file
echo  3. Use the tools to edit your PDF:
echo     - Brush Tool: Click to erase content
echo     - Add Text: Click to add styled text
echo     - Select Text: Edit OCR-detected text
echo  4. Generate final PDF when done
echo.
echo 🛑 Press Ctrl+C to stop the server
echo ========================================
echo.

:: Start the server
python ultimate_pdf_editor_integrated.py

:: If we reach here, the server was stopped
echo.
echo ========================================
echo 🛑 Ultimate PDF Editor Stopped
echo ========================================
echo.
echo Thank you for using Ultimate PDF Editor!
echo.
pause 