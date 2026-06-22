$ErrorActionPreference = "Stop"

$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe"
$script = Join-Path $raiz "ejecutar_excel_juan.py"
$registro = Join-Path $raiz "ejecucion_completa.log"

& $python $script --workers 4 *>> $registro

