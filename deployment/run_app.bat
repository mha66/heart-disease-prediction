@echo off
REM === Start Streamlit App ===
start cmd /k "streamlit run ..\ui\app.py"

REM Small delay to make sure Streamlit starts before Ngrok
timeout /t 5 /nobreak >nul

REM === Start Ngrok ===
start cmd /k "ngrok http 8501"

exit
