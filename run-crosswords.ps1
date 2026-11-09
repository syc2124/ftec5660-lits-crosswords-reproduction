param([ValidateRange(1,3)][int]$Repeats = 3, [ValidateRange(1,30)][int]$Count = 30)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:OPENAI_API_KEY = $env:DEEPSEEK_API_KEY
if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY','User')
}
if (-not $env:OPENAI_API_KEY) { throw 'Set DEEPSEEK_API_KEY before running.' }
$env:OPENAI_API_BASE = 'https://api.deepseek.com'
$env:LITS_NETWORK_RECOVERY_TIMEOUT = '25'
$env:PYTHONPATH = (Resolve-Path demos).Path

foreach ($repeat in 1..$Repeats) {
    $conditions = if ($repeat % 2) { @(10,5) } else { @(5,10) }
    foreach ($steps in $conditions) {
        $result = "crossword_results/r${repeat}_steps${steps}"
        if (Test-Path "$result/eval_results.json") { continue }
        $tokens = Get-ChildItem -Path . -Directory |
            Where-Object Name -Match '^(pilot|crossword_results)' |
            Get-ChildItem -Recurse -Filter inferencelogger.log |
            Get-Content | ForEach-Object {
                $entry = $_ | ConvertFrom-Json
                $entry.input_tokens + $entry.output_tokens
            } | Measure-Object -Sum
        # Stop between runs at a deliberately conservative 10 CNY/M blended estimate.
        if ($tokens.Sum -ge 1000000) { throw 'Conservative token budget threshold reached.' }

        & .venv/Scripts/lits-chain.exe --dataset crosswords --include lits_benchmark.crosswords --dataset-arg data_file=demos/crosswords/data/mini0505.json --policy-model openai/deepseek-flash --cfg "max_steps=$steps" --var offset=1 "limit=$Count" --output-dir $result
        if ($LASTEXITCODE) { throw "Inference failed for $result." }
        & .venv/Scripts/lits-eval-chain.exe --result_dir $result --benchmark crosswords --include lits_benchmark.crosswords
        if ($LASTEXITCODE) { throw "Evaluation failed for $result." }
    }
}

