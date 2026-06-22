# Instrucciones para subir a GitHub

## Repositorio recomendado

`mejora-discriminacion-patologias-cardiacas-valvulares-deep-onmf`

## Paso 1: instalar Git LFS

Este repositorio contiene archivos grandes (`.npy`, `.npz`, etc.). Antes de subir:

```powershell
git lfs install
git config core.longpaths true
```

## Paso 2: inicializar repositorio

Desde esta carpeta:

```powershell
git init
git config core.longpaths true
git add .gitattributes
git add .
git commit -m "Repositorio TFG reproducible"
```

## Paso 3: crear repositorio remoto

Crear en GitHub un repositorio llamado:

`mejora-discriminacion-patologias-cardiacas-valvulares-deep-onmf`

Despues:

```powershell
git remote add origin https://github.com/TU_USUARIO/mejora-discriminacion-patologias-cardiacas-valvulares-deep-onmf.git
git branch -M main
git push -u origin main
```

## Importante

No subir desde la web de GitHub arrastrando carpetas si hay archivos grandes. Usar Git + Git LFS.

Si Windows se queja de rutas largas, ejecutar antes:

```powershell
git config --global core.longpaths true
```
