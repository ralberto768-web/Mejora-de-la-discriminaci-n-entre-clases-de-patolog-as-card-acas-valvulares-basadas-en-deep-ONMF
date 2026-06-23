# Resumen de verificacion de evidencia

Esta carpeta conserva manifiestos y resumenes usados para comprobar que la evidencia experimental incluida en el repositorio esta completa.

## Contenido

- `MANIFIESTO_ARCHIVOS.csv`: relacion de archivos de evidencia, tamanos y hashes cuando estan disponibles.
- `README.md`: explicacion de la finalidad de esta carpeta.

## Uso

La comprobacion publica se ejecuta desde la raiz del repositorio:

```powershell
.\run_all.bat todo
```

Ese comando no reentrena modelos. Su objetivo es comprobar que la descarga conserva los archivos obligatorios y que las evidencias principales se pueden localizar.
