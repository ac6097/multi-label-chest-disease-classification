$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location -LiteralPath $projectDir

function Invoke-HybridModule {
    param(
        [Parameter(Mandatory = $true)][string]$ModuleName,
        [string[]]$ModuleArgs = @()
    )
    Write-Host "===== python -m $ModuleName $($ModuleArgs -join ' ') ====="
    & python -m $ModuleName @ModuleArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Hybrid step failed with exit code $LASTEXITCODE: $ModuleName"
    }
}

function Get-ResumeArgs {
    param([Parameter(Mandatory = $true)][string]$RunDirectory)
    $lastCheckpoint = Join-Path $projectDir (Join-Path $RunDirectory "last.pth")
    if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
        return @("--resume", $lastCheckpoint)
    }
    return @()
}

try {
    $moduleArgs = @("--experiment", "h0") + (Get-ResumeArgs "outputs/hybrid/h0")
    Invoke-HybridModule -ModuleName "hybrid.train_classifier" -ModuleArgs $moduleArgs

    $moduleArgs = @("--method", "h1_pixel") + (Get-ResumeArgs "outputs/hybrid/h1_pretrain")
    Invoke-HybridModule -ModuleName "hybrid.pretrain" -ModuleArgs $moduleArgs
    $moduleArgs = @("--experiment", "h1") + (Get-ResumeArgs "outputs/hybrid/h1")
    Invoke-HybridModule -ModuleName "hybrid.train_classifier" -ModuleArgs $moduleArgs

    $moduleArgs = @("--method", "h2_latent") + (Get-ResumeArgs "outputs/hybrid/h2_pretrain")
    Invoke-HybridModule -ModuleName "hybrid.pretrain" -ModuleArgs $moduleArgs
    $moduleArgs = @("--experiment", "h2") + (Get-ResumeArgs "outputs/hybrid/h2")
    Invoke-HybridModule -ModuleName "hybrid.train_classifier" -ModuleArgs $moduleArgs
    Invoke-HybridModule -ModuleName "hybrid.compare_results"
}
finally {
    Pop-Location
}
