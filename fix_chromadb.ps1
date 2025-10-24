# PowerShell script to fix ChromaDB database corruption
Write-Host "Fixing ChromaDB database corruption..." -ForegroundColor Yellow

$chromaPath = "./chroma_db"

if (Test-Path $chromaPath) {
    Write-Host "Found existing ChromaDB at $chromaPath" -ForegroundColor Yellow
    Write-Host "Removing corrupted database..." -ForegroundColor Red
    
    try {
        Remove-Item -Path $chromaPath -Recurse -Force
        Write-Host "Removed corrupted ChromaDB database" -ForegroundColor Green
    }
    catch {
        Write-Host "Error removing database: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "No existing ChromaDB database found" -ForegroundColor Yellow
}

# Create fresh directory
try {
    New-Item -ItemType Directory -Path $chromaPath -Force | Out-Null
    Write-Host "Created fresh ChromaDB directory" -ForegroundColor Green
    Write-Host "ChromaDB database fixed! You can now run the application." -ForegroundColor Green
}
catch {
    Write-Host "Error creating directory: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
