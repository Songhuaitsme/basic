$ErrorActionPreference = 'Stop'

$pythonExe = 'E:\miniconda3\python.exe'
$outputDir = 'artifacts/v1/benchmarks/cuda_compare'
$trials = @(
    @{ Trial = 2; Device = 'cuda' },
    @{ Trial = 2; Device = 'cpu' },
    @{ Trial = 3; Device = 'cpu' },
    @{ Trial = 3; Device = 'cuda' }
)

foreach ($item in $trials) {
    $trial = $item.Trial
    $device = $item.Device
    $stem = "$outputDir/candidate_dqn_seed14_${device}_trial${trial}"
    & $pythonExe -u -m v1.train_v1 `
        --steps 100 `
        --seed 14 `
        --output "$stem.pt" `
        --device $device `
        --candidate-chunk-size 4096 `
        --batch-size 1 `
        --min-replay-size 1 `
        --updates-per-transition 1 `
        --checkpoint-every 100 `
        --log-every 100 `
        --profile `
        --profile-output "$stem.profile.json"
    if ($LASTEXITCODE -ne 0) {
        throw "Trial $trial on $device failed with exit code $LASTEXITCODE"
    }
}
