$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location "D:\MachineLearning\new_animal_model"

function Assert-NativeSuccess {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

$cropsRoot = "D:\datasets\animal_alive_yolo26_detector_crops_top3\crops"
$runRoot = "D:\MachineLearning\new_animal_model\checkpoints\classification_model\resnet18"

Write-Host "[$(Get-Date -Format s)] Step 1/3: creating YOLO detector crops"
uv run python classifier\make_yolo_detector_crops.py --config classifier\params_yolo_detector_crops.yaml
Assert-NativeSuccess "YOLO detector crop generation"
if (-not (Test-Path $cropsRoot)) {
    throw "Crop generation finished but crops root does not exist: $cropsRoot"
}

Write-Host "[$(Get-Date -Format s)] Step 2/3: training ResNet18 classifier on YOLO crops"
$trainingStarted = Get-Date
uv run python classifier\train_resnet18_classifier.py --config classifier\params_resnet18_yolo_crops_classifier.yaml
Assert-NativeSuccess "ResNet18 classifier training"

$latestRun = Get-ChildItem $runRoot -Directory |
    Where-Object { $_.LastWriteTime -ge $trainingStarted.AddMinutes(-2) } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $latestRun) {
    throw "No new ResNet18 classifier run found after $trainingStarted"
}
$classifier = Join-Path $latestRun.FullName "weights\best.pt"
if (-not (Test-Path $classifier)) {
    throw "Classifier best.pt not found: $classifier"
}

Write-Host "[$(Get-Date -Format s)] Step 3/3: running detector+classifier video pipeline with $classifier"
uv run python classifier\predict_video_detector_classifier.py `
    --detector yolo26n.pt `
    --classifier $classifier `
    --source "D:\MachineLearning\new_animal_model\zwijnen_close.mp4" `
    --save "D:\MachineLearning\new_animal_model\video_predictions\zwijnen_close_yolo_crops_resnet18_448_smoothed.mp4" `
    --det-conf 0.25 `
    --det-iou 0.5 `
    --cls-conf 0.70 `
    --combined-conf 0.35 `
    --min-track-votes 2 `
    --crop-margin 0.18 `
    --classify-every 3 `
    --device 0
Assert-NativeSuccess "Detector+classifier video prediction"

Write-Host "[$(Get-Date -Format s)] Done"
