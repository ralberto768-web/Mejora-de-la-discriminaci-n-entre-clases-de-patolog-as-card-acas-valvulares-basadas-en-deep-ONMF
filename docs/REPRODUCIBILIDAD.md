# Reproducibilidad

El repositorio permite dos niveles de comprobacion: revision de resultados ya generados y reproduccion completa con datos externos.

## Comprobacion del paquete descargado

Desde la raiz del repositorio:

```powershell
.\run_all.bat todo
```

Este comando comprueba:

- dependencias basicas de Python;
- existencia de archivos obligatorios;
- coherencia con el manifiesto del repositorio;
- localizacion de evidencias principales.

## Reproduccion completa

Para repetir entrenamientos o extracciones desde cero se necesita:

1. Instalar dependencias de `requirements.txt` o `environment.yml`.
2. Descargar correctamente archivos gestionados por Git LFS.
3. Colocar las bases de datos externas en `datos_externos/`.
4. Revisar configuraciones y scripts en `metodologia/` y `resultados/`.
5. Ejecutar los flujos correspondientes segun el experimento que se quiera repetir.

## Datos externos

Los audios fuente no estan incluidos directamente en GitHub. La carpeta `datos_externos/` contiene la plantilla de colocacion y `docs/DATOS_EXTERNOS.md` explica como prepararla.

## Integridad

- `github/MANIFIESTO_REPOSITORIO.csv` lista archivos versionados y tamanos.
- `github/ARCHIVOS_GRANDES_GIT_LFS.csv` lista archivos grandes esperados en Git LFS.
- `verificacion/` conserva manifiestos de la evidencia experimental incluida.

La comprobacion rapida no reentrena modelos. Su objetivo es confirmar que el paquete descargado esta completo y que la documentacion apunta a archivos existentes.
