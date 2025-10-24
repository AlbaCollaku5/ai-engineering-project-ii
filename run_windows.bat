@echo off
echo Starting AI Knowledge Assistant on Windows...

REM Set environment variables
set OPENROUTER_API_KEY=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
set OPENROUTER_MODEL=openai/gpt-oss-20b:free

echo Environment variables set:
echo   - OPENROUTER_API_KEY: %OPENROUTER_API_KEY:~0,20%...
echo   - OPENROUTER_MODEL: %OPENROUTER_MODEL%

echo.
echo Checking dependencies...

REM Check if requirements are installed
python -c "import fastapi, uvicorn, chromadb, sentence_transformers, langchain" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install strict requirements. Trying flexible versions...
        pip install -r requirements_flexible.txt
        if errorlevel 1 (
            echo Failed to install flexible requirements. Trying minimal requirements...
            pip install -r requirements_minimal.txt
        )
    )
) else (
    echo Dependencies already installed.
)

echo.
echo Checking for ChromaDB issues...

REM Check if ChromaDB directory exists and might be corrupted
if exist "chroma_db" (
    echo Found existing ChromaDB database...
    echo If you get ChromaDB errors, run: powershell -ExecutionPolicy Bypass -File fix_chromadb.ps1
)

echo.
echo Starting FastAPI server...
echo Open your browser to: http://localhost:8000
echo.

REM Start the server
uvicorn api:app --host 0.0.0.0 --port 8000

pause
