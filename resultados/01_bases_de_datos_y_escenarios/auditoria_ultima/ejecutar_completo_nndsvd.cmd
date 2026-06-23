@echo off
cd /d "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA"
echo Inicio ejecucion completa NNDSVD: %DATE% %TIME% > "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA\auditoria\ejecucion_completa_nndsvd.log"
echo. > "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA\auditoria\ejecucion_completa_nndsvd.err"
echo Antes de Python: %DATE% %TIME% >> "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA\auditoria\ejecucion_completa_nndsvd.log"
"C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe" -u ejecutar.py >> "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA\auditoria\ejecucion_completa_nndsvd.log" 2>> "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA\auditoria\ejecucion_completa_nndsvd.err"
echo Despues de Python errorlevel=%ERRORLEVEL%: %DATE% %TIME% >> "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA\auditoria\ejecucion_completa_nndsvd.log"
echo Fin ejecucion completa NNDSVD: %DATE% %TIME% >> "C:\Users\armga\OneDrive\Escritorio\TFG\Ultima implementacio Juan\ULTIMA\auditoria\ejecucion_completa_nndsvd.log"
