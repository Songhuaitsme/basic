param(
    [int]$TrainingProcessId = 31668
)

$ErrorActionPreference = "Stop"
$workspace = "D:\project\DQN\basic"
$python = "E:\miniconda3\python.exe"
$model = Join-Path $workspace "artifacts\v1\pilot\candidate_dqn_pilot_seed7_b1.pt"
$outputDirectory = Join-Path $workspace "artifacts\v1\evaluation\v1\pilot_seed7_unseen4"
$seeds = @(1, 20, 31, 34)

Set-Location -LiteralPath $workspace

$trainingProcess = Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
if ($null -ne $trainingProcess) {
    Wait-Process -Id $TrainingProcessId
}

if (-not (Test-Path -LiteralPath $model)) {
    throw "Pilot process exited without creating the final model: $model"
}
if ((Get-Item -LiteralPath $model).Length -le 0) {
    throw "Pilot final model is empty: $model"
}

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$baselineReports = @()
$treatmentReports = @()
foreach ($seed in $seeds) {
    $baseline = Join-Path $outputDirectory "seed_${seed}_equal_weight.json"
    $treatment = Join-Path $outputDirectory "seed_${seed}_candidate_dqn.json"

    & $python -u -m v1.evaluate_v1 `
        --policy equal_weight `
        --arrival-cutoff 1.0 `
        --seed $seed `
        --safety-cap 1000000 `
        --device cpu `
        --candidate-chunk-size 4096 `
        --output $baseline
    if ($LASTEXITCODE -ne 0) {
        throw "equal_weight evaluation failed for seed $seed"
    }

    & $python -u -m v1.evaluate_v1 `
        --policy candidate_dqn `
        --arrival-cutoff 1.0 `
        --seed $seed `
        --safety-cap 1000000 `
        --model-path $model `
        --device cpu `
        --candidate-chunk-size 4096 `
        --output $treatment
    if ($LASTEXITCODE -ne 0) {
        throw "candidate_dqn evaluation failed for seed $seed"
    }

    $baselineReports += $baseline
    $treatmentReports += $treatment
}

$analysisArguments = @(
    "-u",
    "-m",
    "v1.analyze_v1"
)
for ($index = 0; $index -lt $seeds.Count; $index++) {
    $analysisArguments += @(
        "--baseline",
        $baselineReports[$index],
        "--treatment",
        $treatmentReports[$index]
    )
}
$analysisArguments += @(
    "--output-prefix",
    (Join-Path $outputDirectory "pilot_seed7_unseen4_effect")
)

& $python @analysisArguments
if ($LASTEXITCODE -ne 0) {
    throw "paired pilot analysis failed"
}

Write-Output "POST_PILOT_DIAGNOSTIC_COMPLETE"
