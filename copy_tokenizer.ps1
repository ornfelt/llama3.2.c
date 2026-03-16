param(
    [string]$ModelDir = "D:\my_files\my_docs\ai\models\small_models\Llama3.2-1B-Instruct"
)

# Example usage:
# .\copy_tokenizer.ps1
# .\copy_tokenizer.ps1 -ModelDir "D:\some\other\model"

$dataDir = Join-Path (Get-Location) "data"

if (-not (Test-Path -LiteralPath $dataDir -PathType Container)) {
    Write-Host "Data directory does not exist: $dataDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) {
    Write-Host "Model directory does not exist: $ModelDir" -ForegroundColor Red
    exit 1
}

$tokenizerJson  = Join-Path $ModelDir "tokenizer.json"
$tokenizerModel = Join-Path $ModelDir "tokenizer.model"

if (-not (Test-Path -LiteralPath $tokenizerJson -PathType Leaf)) {
    Write-Host "Missing file: $tokenizerJson" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -LiteralPath $tokenizerModel -PathType Leaf)) {
    Write-Host "Missing file: $tokenizerModel" -ForegroundColor Red
    exit 1
}

Copy-Item -LiteralPath $tokenizerJson -Destination (Join-Path $dataDir "tokenizer.json") -Force
Copy-Item -LiteralPath $tokenizerModel -Destination (Join-Path $dataDir "llama3_tokenizer.model") -Force

Write-Host "Copied files to: $dataDir" -ForegroundColor Green

