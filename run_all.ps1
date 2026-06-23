param(
    [ValidateSet("entorno", "verificar", "resumen", "todo")]
    [string]$Modo = "todo"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Invoke-RepoPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        & $venvPython @Args
        if ($LASTEXITCODE -ne 0) { throw "Fallo Python: $Args" }
        return
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source @Args
        if ($LASTEXITCODE -ne 0) { throw "Fallo Python: $Args" }
        return
    }

    $localPythonRoots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
    )
    foreach ($candidate in $localPythonRoots) {
        if (Test-Path $candidate) {
            & $candidate @Args
            if ($LASTEXITCODE -ne 0) { throw "Fallo Python: $Args" }
            return
        }
    }

    $pythonFiles = Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA "Programs\Python") -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pythonFiles) {
        & $pythonFiles.FullName @Args
        if ($LASTEXITCODE -ne 0) { throw "Fallo Python: $Args" }
        return
    }

    $codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $codexPython) {
        & $codexPython @Args
        if ($LASTEXITCODE -ne 0) { throw "Fallo Python: $Args" }
        return
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source @Args
        if ($LASTEXITCODE -ne 0) { throw "Fallo py launcher: $Args" }
        return
    }

    throw "No se ha encontrado Python. Instalar Python 3.10 o superior."
}

if ($Modo -eq "entorno" -or $Modo -eq "todo") {
    Invoke-RepoPython scripts\comprobar_entorno.py
}

if ($Modo -eq "verificar" -or $Modo -eq "todo") {
    Invoke-RepoPython scripts\verificar_repositorio.py --modo rapido
}

if ($Modo -eq "resumen" -or $Modo -eq "todo") {
    Invoke-RepoPython scripts\resumen_resultados.py
}
