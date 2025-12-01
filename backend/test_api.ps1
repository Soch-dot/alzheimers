# PowerShell script to test the API with hard test cases
# Run with: .\test_api.ps1

$baseUrl = "http://127.0.0.1:8000/predict"

Write-Host "=" -NoNewline
Write-Host ("=" * 79)
Write-Host "TESTING API WITH HARD CASES"
Write-Host ("=" * 80)

# Test Case 1: Clear Healthy
Write-Host "`n[TEST CASE 1] Clear Nondemented Case (Healthy)" -ForegroundColor Cyan
$body1 = '{"visit":1,"mr_delay":0,"sex":1,"hand":1,"age":70,"education_years":16,"ses":3.0,"mmse":29.0,"cdr":0.0,"etiv":1500,"nwbv":0.75,"asf":1.2}'
$response1 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body1 -ContentType "application/json"
Write-Host "Expected: Nondemented"
Write-Host "Predicted: $($response1.predicted_class)" -ForegroundColor $(if($response1.predicted_class -eq "Nondemented"){"Green"}else{"Red"})
Write-Host "Detection %: $($response1.detection_percentage)%"
Write-Host "Probabilities:"
$response1.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 2: Clear Demented
Write-Host "`n[TEST CASE 2] Clear Demented Case (Severe)" -ForegroundColor Cyan
$body2 = '{"visit":1,"mr_delay":0,"sex":0,"hand":1,"age":85,"education_years":12,"ses":2.0,"mmse":15.0,"cdr":2.0,"etiv":1400,"nwbv":0.65,"asf":1.1}'
$response2 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body2 -ContentType "application/json"
Write-Host "Expected: Demented"
Write-Host "Predicted: $($response2.predicted_class)" -ForegroundColor $(if($response2.predicted_class -eq "Demented"){"Green"}else{"Red"})
Write-Host "Detection %: $($response2.detection_percentage)%"
Write-Host "Probabilities:"
$response2.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 3: Borderline
Write-Host "`n[TEST CASE 3] Borderline Converted Case" -ForegroundColor Cyan
$body3 = '{"visit":1,"mr_delay":0,"sex":1,"hand":1,"age":78,"education_years":14,"ses":2.0,"mmse":22.0,"cdr":0.5,"etiv":1600,"nwbv":0.70,"asf":1.15}'
$response3 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body3 -ContentType "application/json"
Write-Host "Expected: Converted or Demented"
Write-Host "Predicted: $($response3.predicted_class)" -ForegroundColor $(if($response3.predicted_class -in @("Converted","Demented")){"Green"}else{"Red"})
Write-Host "Detection %: $($response3.detection_percentage)%"
Write-Host "Probabilities:"
$response3.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 4: Very Old but Healthy
Write-Host "`n[TEST CASE 4] Very Old but High MMSE" -ForegroundColor Cyan
$body4 = '{"visit":1,"mr_delay":0,"sex":0,"hand":1,"age":90,"education_years":18,"ses":4.0,"mmse":28.0,"cdr":0.0,"etiv":1300,"nwbv":0.72,"asf":1.3}'
$response4 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body4 -ContentType "application/json"
Write-Host "Expected: Nondemented"
Write-Host "Predicted: $($response4.predicted_class)" -ForegroundColor $(if($response4.predicted_class -eq "Nondemented"){"Green"}else{"Red"})
Write-Host "Detection %: $($response4.detection_percentage)%"
Write-Host "Probabilities:"
$response4.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 5: Young but Low MMSE
Write-Host "`n[TEST CASE 5] Young but Low MMSE" -ForegroundColor Cyan
$body5 = '{"visit":1,"mr_delay":0,"sex":1,"hand":1,"age":65,"education_years":10,"ses":1.0,"mmse":18.0,"cdr":1.0,"etiv":1700,"nwbv":0.68,"asf":1.0}'
$response5 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body5 -ContentType "application/json"
Write-Host "Expected: Demented or Converted"
Write-Host "Predicted: $($response5.predicted_class)" -ForegroundColor $(if($response5.predicted_class -in @("Converted","Demented")){"Green"}else{"Red"})
Write-Host "Detection %: $($response5.detection_percentage)%"
Write-Host "Probabilities:"
$response5.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 6: High MMSE, Clear Dementia Signs (TRICKY)
Write-Host "`n[TEST CASE 6] High MMSE, Clear Dementia Signs (TRICKY)" -ForegroundColor Yellow
$body6 = '{"visit":2,"mr_delay":12,"sex":1,"hand":1,"age":79,"education_years":18,"ses":4.0,"mmse":26.0,"cdr":1.0,"etiv":1340,"nwbv":0.62,"asf":1.12}'
$response6 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body6 -ContentType "application/json"
Write-Host "Expected: Demented"
Write-Host "Predicted: $($response6.predicted_class)" -ForegroundColor $(if($response6.predicted_class -eq "Demented"){"Green"}else{"Red"})
Write-Host "Detection %: $($response6.detection_percentage)%"
Write-Host "Probabilities:"
$response6.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 7: Low Education, Low MMSE, Not Dementia (TRICKY)
Write-Host "`n[TEST CASE 7] Low Education, Low MMSE, Not Dementia (TRICKY)" -ForegroundColor Yellow
$body7 = '{"visit":1,"mr_delay":0,"sex":0,"hand":1,"age":70,"education_years":5,"ses":2.0,"mmse":20.0,"cdr":0.0,"etiv":1510,"nwbv":0.80,"asf":1.30}'
$response7 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body7 -ContentType "application/json"
Write-Host "Expected: Nondemented"
Write-Host "Predicted: $($response7.predicted_class)" -ForegroundColor $(if($response7.predicted_class -eq "Nondemented"){"Green"}else{"Red"})
Write-Host "Detection %: $($response7.detection_percentage)%"
Write-Host "Probabilities:"
$response7.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 8: High Volume, Early Clinical Decline (TRICKY)
Write-Host "`n[TEST CASE 8] High Volume, Early Clinical Decline (TRICKY)" -ForegroundColor Yellow
$body8 = '{"visit":1,"mr_delay":0,"sex":1,"hand":1,"age":58,"education_years":15,"ses":3.0,"mmse":27.0,"cdr":0.5,"etiv":1600,"nwbv":0.82,"asf":1.18}'
$response8 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body8 -ContentType "application/json"
Write-Host "Expected: Converted"
Write-Host "Predicted: $($response8.predicted_class)" -ForegroundColor $(if($response8.predicted_class -eq "Converted"){"Green"}else{"Yellow"})
Write-Host "Detection %: $($response8.detection_percentage)%"
Write-Host "Probabilities:"
$response8.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 9: Old Age, Low Brain Volume, Stable Cognition (TRICKY)
Write-Host "`n[TEST CASE 9] Old Age, Low Brain Volume, Stable Cognition (TRICKY)" -ForegroundColor Yellow
$body9 = '{"visit":3,"mr_delay":24,"sex":0,"hand":1,"age":91,"education_years":12,"ses":2.0,"mmse":25.0,"cdr":0.0,"etiv":1200,"nwbv":0.59,"asf":1.05}'
$response9 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body9 -ContentType "application/json"
Write-Host "Expected: Nondemented"
Write-Host "Predicted: $($response9.predicted_class)" -ForegroundColor $(if($response9.predicted_class -eq "Nondemented"){"Green"}else{"Red"})
Write-Host "Detection %: $($response9.detection_percentage)%"
Write-Host "Probabilities:"
$response9.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

# Test Case 10: Normal Metrics, Behavioral Dementia (TRICKY)
Write-Host "`n[TEST CASE 10] Normal Metrics, Behavioral Dementia (TRICKY)" -ForegroundColor Yellow
$body10 = '{"visit":1,"mr_delay":0,"sex":1,"hand":1,"age":67,"education_years":16,"ses":4.0,"mmse":28.0,"cdr":1.0,"etiv":1505,"nwbv":0.79,"asf":1.21}'
$response10 = Invoke-RestMethod -Uri $baseUrl -Method Post -Body $body10 -ContentType "application/json"
Write-Host "Expected: Demented"
Write-Host "Predicted: $($response10.predicted_class)" -ForegroundColor $(if($response10.predicted_class -eq "Demented"){"Green"}else{"Red"})
Write-Host "Detection %: $($response10.detection_percentage)%"
Write-Host "Probabilities:"
$response10.probabilities.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $([math]::Round($_.Value * 100, 2))%" }

Write-Host "`n" + ("=" * 80)
Write-Host "TESTING COMPLETE - 10 Test Cases (5 Original + 5 Tricky)"
Write-Host ("=" * 80)

