$ErrorActionPreference = "Stop"

$SourceDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $SourceDirectory
$EnvironmentPath = Join-Path $ProjectRoot ".venv_training"
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
$RequirementsPath = Join-Path $SourceDirectory "requirements-training.txt"

if (-not (Test-Path -LiteralPath $RequirementsPath)) {
    throw "Requirements file not found: $RequirementsPath"
}

# Prefer a regular Python installation. Microsoft Store aliases frequently
# create virtual environments that later point to an unavailable executable.
$CandidateCommands = @(
    @{ Command = "py"; Arguments = @("-3.12") },
    @{ Command = "py"; Arguments = @("-3.11") },
    @{ Command = "python"; Arguments = @() }
)

$BaseCommand = $null
$BaseArguments = @()
foreach ($Candidate in $CandidateCommands) {
    try {
        $Resolved = Get-Command $Candidate.Command -ErrorAction Stop
        & $Resolved.Source @($Candidate.Arguments) -c "import sys; print(sys.executable)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $BaseCommand = $Resolved.Source
            $BaseArguments = $Candidate.Arguments
            break
        }
    }
    catch {
        continue
    }
}

if (-not $BaseCommand) {
    throw "No working Python 3.11 or 3.12 installation was found. Install Python from python.org, then rerun this script."
}

if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    Write-Host "Creating training environment at $EnvironmentPath"
    & $BaseCommand @BaseArguments -m venv $EnvironmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual-environment creation failed."
    }
}

Write-Host "Bootstrapping pip..."
& $EnvironmentPython -m ensurepip --upgrade
& $EnvironmentPython -m pip install --upgrade pip

Write-Host "Installing training dependencies..."
& $EnvironmentPython -m pip install -r $RequirementsPath
& $EnvironmentPython -m pip install ipykernel
& $EnvironmentPython -m pip check

Write-Host "Verifying TensorFlow..."
& $EnvironmentPython -c "import sys, tensorflow as tf, numpy as np, scipy; print('Python:', sys.executable); print('TensorFlow:', tf.__version__); print('NumPy:', np.__version__); print('SciPy:', scipy.__version__)"

Write-Host "Registering the Jupyter kernel..."
& $EnvironmentPython -m ipykernel install --user --name eeai-training --display-name "EEAI Training"

Write-Host "Training environment is ready. Select the 'EEAI Training' kernel in VS Code."
