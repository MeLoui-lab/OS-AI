cd /d C:\AI\OS_AI
call venv\Scripts\activate.bat
uvicorn main:app --host 0.0.0.0 --port 8000 --reload