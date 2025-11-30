# Quick single test - Test Case 1: Clear Healthy
# Run with: .\quick_test.ps1

$body = '{"visit":1,"mr_delay":0,"sex":1,"hand":1,"age":70,"education_years":16,"ses":3.0,"mmse":29.0,"cdr":0.0,"etiv":1500,"nwbv":0.75,"asf":1.2}'

Write-Host "Testing API with Clear Healthy Case..." -ForegroundColor Cyan
Write-Host "Expected: Nondemented`n"

try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
    
    Write-Host "Result:" -ForegroundColor Green
    Write-Host "  Predicted Class: $($response.predicted_class)"
    Write-Host "  Detection %: $($response.detection_percentage)%"
    Write-Host "  Alzheimer's Detected: $($response.alzheimers_detected)"
    Write-Host "`n  Probabilities:"
    $response.probabilities.PSObject.Properties | ForEach-Object { 
        Write-Host "    $($_.Name): $([math]::Round($_.Value * 100, 2))%" 
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Make sure your API server is running: uvicorn src.api:app --reload" -ForegroundColor Yellow
}

