param(
    [Parameter(Mandatory = $true)][string[]]$Runs,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$Python = 'python',
    [ValidateSet('training', 'split')][string]$Unit = 'training',
    [ValidateSet('mae', 'rmse', 'mse', 'nmse', 'r2')][string]$Metric = 'rmse',
    [ValidateSet('train', 'validation', 'test')][string]$Phase = 'test',
    [ValidateSet('sd', 'ci95')][string]$Interval = 'sd',
    [switch]$NoPlots
)

# Relative input/output paths are resolved from the repository root.
$taskRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskOldPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $taskRoot
try {
    $env:PYTHONPATH = @((Join-Path $taskRoot 'src'), $taskOldPythonPath) -join ';'
    $taskArguments = @('-m', 'graphml_lichess.analysis', '--runs') + $Runs + @(
        '--output', $Output, '--unit', $Unit, '--metric', $Metric,
        '--phase', $Phase, '--interval', $Interval
    )
    if ($NoPlots) { $taskArguments += '--no-plots' }
    & $Python @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Analysis exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $taskOldPythonPath
    Pop-Location
}
