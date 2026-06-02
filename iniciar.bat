@echo off
cd /d "%~dp0"
"%USERPROFILE%\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run app.py --server.headless true &
timeout /t 3 /nobreak >nul
start "" http://localhost:8501
pause
