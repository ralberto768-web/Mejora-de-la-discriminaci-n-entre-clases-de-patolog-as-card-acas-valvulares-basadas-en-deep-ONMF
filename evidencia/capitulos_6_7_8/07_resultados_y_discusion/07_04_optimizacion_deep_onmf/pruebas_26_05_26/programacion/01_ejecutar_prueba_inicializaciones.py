from __future__ import annotations

"""Prueba comparativa Deep ONMF con inicializaciones aleatoria, NNDSVD, NNDSVDa y NNDSVDar."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def localizar_raiz_objetivo() -> Path:
    archivo = Path(__file__).resolve()
    candidatos = [archivo.parent, *archivo.parents, Path.cwd(), *Path.cwd().parents]
    for base in candidatos:
        if (base / "src" / "tfg_deep_onmf").exists() and (base / "Bases de Datos").exists():
            return base
        candidato = base / "Programacion objetivo"
        if (candidato / "src" / "tfg_deep_onmf").exists() and (candidato / "Bases de Datos").exists():
            return candidato
    raise RuntimeError("No se ha encontrado la carpeta 'Programacion objetivo'.")


RAIZ_OBJETIVO = localizar_raiz_objetivo()
SRC_OBJETIVO = RAIZ_OBJETIVO / "src"
if str(SRC_OBJETIVO) not in sys.path:
    sys.path.insert(0, str(SRC_OBJETIVO))

CARPETA_PRUEBA = RAIZ_OBJETIVO / "pruebas para el 26-05-26"
CARPETA_PROGRAMACION = CARPETA_PRUEBA / "programacion"
CARPETA_RESULTADOS = CARPETA_PRUEBA / "resultados comparativos"
CARPETA_FOTOS_SEPARADAS = CARPETA_RESULTADOS / "01_fotos_por_separado"
CARPETA_EXISTENTES = CARPETA_RESULTADOS / "00_resultados_existentes_usados"
MPL_CACHE = CARPETA_RESULTADOS / ".cache_matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

from tfg_deep_onmf.audio import construir_matriz_clase, descubrir_audios
from tfg_deep_onmf.configuracion import Configuracion
from tfg_deep_onmf.estadistica import caracteristicas_por_audio, distancias_figura_7, resumen_auditoria, tabla_2_desde_w
from tfg_deep_onmf.graficos import COLORES, figura_5_sbv, figura_7_distancias, figura_11d_tsne, tabla_2_imagen
from tfg_deep_onmf.onmf import CapaONMF, ResultadoONMF


CLASES = ("N", "AS", "MR", "MS", "MVP")
EPS = 1e-12


@dataclass(frozen=True)
class MetodoInicializacion:
    codigo: str
    nombre: str
    descripcion: str
    carpeta: str


METODOS = (
    MetodoInicializacion(
        codigo="aleatoria_actual",
        nombre="Aleatoria actual",
        descripcion=(
            "Es la implementacion que ya estaba en el proyecto: W y H se inicializan "
            "con numeros aleatorios positivos usando la semilla base."
        ),
        carpeta="00_aleatoria_actual",
    ),
    MetodoInicializacion(
        codigo="nndsvd",
        nombre="NNDSVD",
        descripcion=(
            "Inicializacion por SVD no negativa. Deja ceros estructurales donde la "
            "descomposicion no aporta valores positivos."
        ),
        carpeta="01_NNDSVD",
    ),
    MetodoInicializacion(
        codigo="nndsvda",
        nombre="NNDSVDa",
        descripcion=(
            "Variante de NNDSVD que sustituye los ceros por la media de la matriz X. "
            "Suele evitar bloqueos por ceros en las actualizaciones multiplicativas."
        ),
        carpeta="02_NNDSVDa",
    ),
    MetodoInicializacion(
        codigo="nndsvdar",
        nombre="NNDSVDar",
        descripcion=(
            "Variante de NNDSVD que sustituye los ceros por valores aleatorios muy "
            "pequenos. Mantiene la estructura SVD y anade una perturbacion controlada."
        ),
        carpeta="03_NNDSVDar",
    ),
)


def crear_carpetas() -> None:
    for carpeta in (CARPETA_PRUEBA, CARPETA_PROGRAMACION, CARPETA_RESULTADOS, CARPETA_FOTOS_SEPARADAS, CARPETA_EXISTENTES):
        carpeta.mkdir(parents=True, exist_ok=True)


def normalizar_columnas_w(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    escala = np.linalg.norm(w, axis=0)
    escala = np.maximum(escala, EPS)
    return w / escala[None, :], h * escala[:, None]


def ortogonalidad_media(h: np.ndarray) -> float:
    normas = np.linalg.norm(h, axis=1, keepdims=True)
    h_norm = h / np.maximum(normas, EPS)
    gramo = h_norm @ h_norm.T
    mascara = ~np.eye(gramo.shape[0], dtype=bool)
    return float(np.mean(np.abs(gramo[mascara])))


def inicializar_aleatorio(x: np.ndarray, rango: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    filas, columnas = x.shape
    w = rng.random((filas, rango)) + EPS
    h = rng.random((rango, columnas)) + EPS
    return normalizar_columnas_w(w, h)


def inicializar_nndsvd(
    x: np.ndarray,
    rango: int,
    variante: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    filas, columnas = x.shape
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    limite = min(rango, len(s))
    w = np.zeros((filas, rango), dtype=np.float64)
    h = np.zeros((rango, columnas), dtype=np.float64)

    if limite > 0:
        raiz = np.sqrt(s[0])
        w[:, 0] = raiz * np.abs(u[:, 0])
        h[0, :] = raiz * np.abs(vt[0, :])

    for componente in range(1, limite):
        vector_u = u[:, componente]
        vector_v = vt[componente, :]
        u_pos = np.maximum(vector_u, 0.0)
        u_neg = np.maximum(-vector_u, 0.0)
        v_pos = np.maximum(vector_v, 0.0)
        v_neg = np.maximum(-vector_v, 0.0)

        norma_pos = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
        norma_neg = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)

        if norma_pos >= norma_neg:
            u_elegido, v_elegido, norma = u_pos, v_pos, norma_pos
        else:
            u_elegido, v_elegido, norma = u_neg, v_neg, norma_neg

        if norma <= EPS:
            continue
        coeficiente = np.sqrt(s[componente] * norma)
        w[:, componente] = coeficiente * u_elegido / np.maximum(np.linalg.norm(u_elegido), EPS)
        h[componente, :] = coeficiente * v_elegido / np.maximum(np.linalg.norm(v_elegido), EPS)

    media = float(np.mean(x))
    if variante == "nndsvda":
        w[w == 0.0] = media
        h[h == 0.0] = media
    elif variante == "nndsvdar":
        w[w == 0.0] = rng.random(np.count_nonzero(w == 0.0)) * media / 100.0
        h[h == 0.0] = rng.random(np.count_nonzero(h == 0.0)) * media / 100.0

    w = np.maximum(w, EPS)
    h = np.maximum(h, EPS)
    return normalizar_columnas_w(w, h)


def inicializar(x: np.ndarray, rango: int, metodo: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if metodo == "aleatoria_actual":
        return inicializar_aleatorio(x, rango, rng)
    if metodo in {"nndsvd", "nndsvda", "nndsvdar"}:
        return inicializar_nndsvd(x, rango, metodo, rng)
    raise ValueError(f"Metodo de inicializacion no soportado: {metodo}")


def factorizar_onmf_con_inicializacion(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
    metodo: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.maximum(matriz, EPS).astype(np.float64, copy=False)
    rng = np.random.default_rng(semilla)
    w, h = inicializar(x, rango, metodo, rng)

    for _ in range(iteraciones):
        numerador_w = x @ h.T
        denominador_w = w @ (h @ h.T) + EPS
        w *= numerador_w / denominador_w
        w = np.maximum(w, EPS)

        numerador_h = w.T @ x + penalizacion_ortogonal * h
        denominador_h = (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + EPS
        h *= numerador_h / denominador_h
        h = np.maximum(h, EPS)

        w, h = normalizar_columnas_w(w, h)

    reconstruida = w @ h
    error = float(np.linalg.norm(x - reconstruida, ord="fro") / np.maximum(np.linalg.norm(x, ord="fro"), EPS))
    return w, h, error, ortogonalidad_media(h)


def deep_onmf_con_inicializacion(
    matriz: np.ndarray,
    rangos: tuple[int, int, int],
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
    metodo: str,
) -> ResultadoONMF:
    entrada = matriz
    matrices_w: list[np.ndarray] = []
    capas: list[CapaONMF] = []

    for indice, rango in enumerate(rangos, start=1):
        inicio = time.perf_counter()
        w, h, error, ortogonalidad = factorizar_onmf_con_inicializacion(
            entrada,
            rango=rango,
            iteraciones=iteraciones,
            penalizacion_ortogonal=penalizacion_ortogonal,
            semilla=semilla + indice * 1000,
            metodo=metodo,
        )
        segundos = time.perf_counter() - inicio
        matrices_w.append(w)
        capas.append(
            CapaONMF(
                indice=indice,
                rango=rango,
                forma_entrada=entrada.shape,
                forma_w=w.shape,
                forma_h=h.shape,
                error_relativo=error,
                ortogonalidad_media=ortogonalidad,
                segundos=segundos,
            )
        )
        entrada = h

    w_final = matrices_w[0] @ matrices_w[1] @ matrices_w[2]
    normas = np.maximum(np.linalg.norm(w_final, axis=0), EPS)
    w_final = w_final / normas[None, :]
    h_final = entrada * normas[:, None]
    x = np.maximum(matriz, EPS)
    error_final = float(np.linalg.norm(x - w_final @ h_final, ord="fro") / np.maximum(np.linalg.norm(x, ord="fro"), EPS))
    return ResultadoONMF(w_final=w_final, h_final=h_final, capas=capas, error_relativo_final=error_final)


def dataframe_capas(resultados: dict[str, ResultadoONMF], metodo: MetodoInicializacion) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    for clase, resultado in resultados.items():
        for capa in resultado.capas:
            filas.append(
                {
                    "metodo": metodo.nombre,
                    "codigo_metodo": metodo.codigo,
                    "clase": clase,
                    "capa": capa.indice,
                    "rango": capa.rango,
                    "entrada": f"{capa.forma_entrada[0]} x {capa.forma_entrada[1]}",
                    "W": f"{capa.forma_w[0]} x {capa.forma_w[1]}",
                    "H": f"{capa.forma_h[0]} x {capa.forma_h[1]}",
                    "error_relativo": capa.error_relativo,
                    "ortogonalidad_media": capa.ortogonalidad_media,
                    "segundos": capa.segundos,
                }
            )
    return pd.DataFrame(filas)


def metricas_separacion(caracteristicas: pd.DataFrame, coordenadas: pd.DataFrame) -> dict[str, float]:
    columnas = [f"SBV_{indice}" for indice in range(1, 8)]
    x = caracteristicas[columnas].to_numpy(dtype=float)
    etiquetas = caracteristicas["clase"].to_numpy()
    x_escalada = StandardScaler().fit_transform(x)
    coords = coordenadas[["tSNE_1", "tSNE_2"]].to_numpy(dtype=float)
    return {
        "silhouette_features": float(silhouette_score(x_escalada, etiquetas)),
        "davies_bouldin_features": float(davies_bouldin_score(x_escalada, etiquetas)),
        "silhouette_tsne": float(silhouette_score(coords, etiquetas)),
        "davies_bouldin_tsne": float(davies_bouldin_score(coords, etiquetas)),
    }


def copiar_archivo(origen: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)


def nombre_seguro(origen_base: Path, ruta: Path, limite: int = 145) -> str:
    relativo = ruta.relative_to(origen_base)
    base = "__".join(relativo.parts)
    base = "".join(caracter if caracter.isalnum() or caracter in "._- " else "_" for caracter in base)
    if len(base) <= limite:
        return base
    sufijo = ruta.suffix
    return base[: limite - len(sufijo) - 1].rstrip("._- ") + sufijo


def escribir_json(ruta: Path, datos: object) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def ejecutar_metodo(
    metodo: MetodoInicializacion,
    configuracion: Configuracion,
    datos_por_clase: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, dict[str, Path]]:
    print(f"\n=== Ejecutando {metodo.nombre} ===")
    inicio_metodo = time.perf_counter()
    carpeta = CARPETA_RESULTADOS / metodo.carpeta
    carpeta_figuras = carpeta / "figuras"
    carpeta_datos = carpeta / "datos"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)
    carpeta_datos.mkdir(parents=True, exist_ok=True)

    resultados: dict[str, ResultadoONMF] = {}
    w_por_clase: dict[str, np.ndarray] = {}
    h_por_clase: dict[str, np.ndarray] = {}

    for posicion, clase in enumerate(CLASES, start=1):
        print(f"[{metodo.nombre}] Clase {clase} ({posicion}/{len(CLASES)})")
        resultado = deep_onmf_con_inicializacion(
            datos_por_clase[clase].matriz,
            rangos=configuracion.rangos_onmf,
            iteraciones=configuracion.iteraciones_onmf,
            penalizacion_ortogonal=configuracion.penalizacion_ortogonal,
            semilla=configuracion.semilla + posicion * 100,
            metodo=metodo.codigo,
        )
        resultados[clase] = resultado
        w_por_clase[clase] = resultado.w_final
        h_por_clase[clase] = resultado.h_final

    caracteristicas = caracteristicas_por_audio(datos_por_clase, h_por_clase)
    tabla_2 = tabla_2_desde_w(w_por_clase, CLASES)
    distancias = distancias_figura_7(caracteristicas, CLASES)

    caracteristicas.to_csv(carpeta_datos / "caracteristicas_sbv_por_audio.csv", index=False, encoding="utf-8-sig")
    tabla_2.to_csv(carpeta_datos / "tabla_2_estadistica_sbv.csv", index=False, encoding="utf-8-sig")
    for nombre, tabla in distancias.items():
        tabla.to_csv(carpeta_datos / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    np.savez(carpeta_datos / "matrices_w_finales_por_clase.npz", **{f"W_{clase}": w for clase, w in w_por_clase.items()})

    detalle_capas = dataframe_capas(resultados, metodo)
    detalle_capas.to_csv(carpeta_datos / "detalle_capas_onmf.csv", index=False, encoding="utf-8-sig")
    escribir_json(
        carpeta_datos / "errores_finales_por_clase.json",
        {clase: resultado.error_relativo_final for clase, resultado in resultados.items()},
    )

    figura_5 = carpeta_figuras / f"{metodo.carpeta}_Figura_5_SBV_por_clase.png"
    tabla_2_png = carpeta_figuras / f"{metodo.carpeta}_Tabla_2_estadistica_SBV.png"
    figura_7 = carpeta_figuras / f"{metodo.carpeta}_Figura_7_distancias_euclideas.png"
    figura_11 = carpeta_figuras / f"{metodo.carpeta}_Figura_11D_tSNE_deep_ONMF.png"
    coords_csv = carpeta_datos / "coordenadas_figura_11D_tSNE.csv"

    figura_5_sbv(w_por_clase, CLASES, configuracion.frecuencia_esperada_hz, figura_5)
    tabla_2_imagen(tabla_2, tabla_2_png)
    figura_7_distancias(distancias, figura_7)
    figura_11d_tsne(caracteristicas, CLASES, figura_11, coords_csv)

    coordenadas = pd.read_csv(coords_csv, encoding="utf-8-sig")
    metricas = metricas_separacion(caracteristicas, coordenadas)
    segundos = time.perf_counter() - inicio_metodo
    errores = [resultado.error_relativo_final for resultado in resultados.values()]
    metricas_fila: dict[str, object] = {
        "metodo": metodo.nombre,
        "codigo_metodo": metodo.codigo,
        "descripcion": metodo.descripcion,
        "segundos": segundos,
        "error_relativo_final_medio": float(np.mean(errores)),
        "error_relativo_final_maximo": float(np.max(errores)),
        "ortogonalidad_media_capas": float(detalle_capas["ortogonalidad_media"].mean()),
        **metricas,
    }

    rutas = {
        "figura_5": figura_5,
        "tabla_2_png": tabla_2_png,
        "figura_7": figura_7,
        "figura_11": figura_11,
        "coords_csv": coords_csv,
    }
    copiar_archivo(figura_5, CARPETA_FOTOS_SEPARADAS / f"{metodo.carpeta}_Figura_5_SBV_por_clase.png")
    copiar_archivo(tabla_2_png, CARPETA_FOTOS_SEPARADAS / f"{metodo.carpeta}_Tabla_2_estadistica_SBV.png")
    copiar_archivo(figura_7, CARPETA_FOTOS_SEPARADAS / f"{metodo.carpeta}_Figura_7_distancias_euclideas.png")
    copiar_archivo(figura_11, CARPETA_FOTOS_SEPARADAS / f"{metodo.carpeta}_Figura_11D_tSNE_deep_ONMF.png")

    escribir_explicacion_metodo(carpeta / "explicacion_resultado.md", metodo, metricas_fila, rutas)
    print(
        f"{metodo.nombre}: silhouette t-SNE={metricas_fila['silhouette_tsne']:.4f}, "
        f"DB t-SNE={metricas_fila['davies_bouldin_tsne']:.4f}, "
        f"error medio={metricas_fila['error_relativo_final_medio']:.4f}"
    )
    return metricas_fila, detalle_capas, rutas


def escribir_explicacion_metodo(
    ruta: Path,
    metodo: MetodoInicializacion,
    metricas: dict[str, object],
    rutas: dict[str, Path],
) -> None:
    texto = f"""# Resultado {metodo.nombre}

## Que se ha cambiado

{metodo.descripcion}

El resto del procedimiento se mantiene igual: misma base de datos, mismas clases, tramas de 2 segundos, solape de 1 segundo, rangos ONMF 9-8-7, 120 iteraciones por capa y penalizacion ortogonal 0.05.

## Como leer sus resultados

- `Figura_5_SBV_por_clase.png`: muestra las bases espectrales aprendidas por clase. Si las curvas son mas limpias y diferenciadas, la base W es mas interpretable.
- `Tabla_2_estadistica_SBV.png`: resume medias, desviaciones y p-valores de los SBV. P-valores mas pequenos indican mayor diferencia estadistica entre clases.
- `Figura_7_distancias_euclideas.png`: compara separacion entre clases y dispersion dentro de clase.
- `Figura_11D_tSNE_deep_ONMF.png`: visualiza los audios en 2D. Es la foto clave para ver si las clases quedan mas separadas.

## Metricas obtenidas

| Metrica | Valor |
|---|---:|
| Error relativo final medio | {metricas['error_relativo_final_medio']:.6f} |
| Error relativo final maximo | {metricas['error_relativo_final_maximo']:.6f} |
| Ortogonalidad media de capas | {metricas['ortogonalidad_media_capas']:.6f} |
| Silhouette en rasgos SBV | {metricas['silhouette_features']:.6f} |
| Davies-Bouldin en rasgos SBV | {metricas['davies_bouldin_features']:.6f} |
| Silhouette en t-SNE | {metricas['silhouette_tsne']:.6f} |
| Davies-Bouldin en t-SNE | {metricas['davies_bouldin_tsne']:.6f} |

## Archivos de este metodo

- Figura 5: `{rutas['figura_5'].name}`
- Tabla 2 imagen: `{rutas['tabla_2_png'].name}`
- Figura 7: `{rutas['figura_7'].name}`
- Figura 11D: `{rutas['figura_11'].name}`
- Coordenadas t-SNE: `{rutas['coords_csv'].name}`

"""
    ruta.write_text(texto, encoding="utf-8")


def copiar_resultados_existentes() -> None:
    fuentes = [
        RAIZ_OBJETIVO / "comparacion final" / "04_prueba_ajustada_codigo_y_resultados" / "resultados" / "final_clave",
        RAIZ_OBJETIVO / "comparacion final" / "04_prueba_ajustada_codigo_y_resultados" / "resultados",
        RAIZ_OBJETIVO / "comparacion final" / "02_resultados",
    ]
    extensiones = {".png", ".jpg", ".jpeg", ".pdf", ".csv", ".md", ".json", ".zip"}
    copiados: list[Path] = []
    for fuente in fuentes:
        if not fuente.exists():
            continue
        for ruta in fuente.rglob("*"):
            if not ruta.is_file() or ruta.suffix.lower() not in extensiones:
                continue
            destino = CARPETA_EXISTENTES / fuente.name / nombre_seguro(fuente, ruta)
            try:
                copiar_archivo(ruta, destino)
            except OSError as error:
                aviso = CARPETA_EXISTENTES / "archivos_no_copiados_por_ruta_larga.txt"
                aviso.parent.mkdir(parents=True, exist_ok=True)
                with aviso.open("a", encoding="utf-8") as manejador:
                    manejador.write(f"{ruta} -> {error}\n")
                continue
            copiados.append(destino)

    lineas = [
        "# Resultados existentes reutilizados",
        "",
        "Esta carpeta conserva copias de los resultados previos que se usan como contexto.",
        "La prueba nueva de Juan no los modifica: solo los deja al lado para poder comparar.",
        "",
        "## Archivos copiados",
        "",
    ]
    for ruta in sorted(copiados):
        tipo = "foto/figura" if ruta.suffix.lower() in {".png", ".jpg", ".jpeg"} else "resultado/documento"
        lineas.append(f"- `{ruta.relative_to(CARPETA_EXISTENTES)}`: {tipo} existente usado como referencia.")
    (CARPETA_EXISTENTES / "00_EXPLICACION_RESULTADOS_EXISTENTES.md").write_text("\n".join(lineas), encoding="utf-8")


def copiar_programacion() -> None:
    destino_script = CARPETA_PROGRAMACION / "01_ejecutar_prueba_inicializaciones.py"
    origen_script = Path(__file__).resolve()
    if origen_script != destino_script:
        copiar_archivo(origen_script, destino_script)

    bat = """@echo off
cd /d "%~dp0"
set "PYTHON313=C:\\Users\\armga\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"
if exist "%PYTHON313%" (
    "%PYTHON313%" 01_ejecutar_prueba_inicializaciones.py
) else (
    python 01_ejecutar_prueba_inicializaciones.py
)
pause
"""
    (CARPETA_PROGRAMACION / "02_EJECUTAR_PRUEBA_JUAN_INICIALIZACIONES.bat").write_text(bat, encoding="utf-8")
    requisitos = "\n".join(["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]) + "\n"
    (CARPETA_PROGRAMACION / "requirements_prueba_juan.txt").write_text(requisitos, encoding="utf-8")

    destino_src = CARPETA_PROGRAMACION / "src_tfg_deep_onmf_usado"
    if SRC_OBJETIVO.exists():
        shutil.copytree(
            SRC_OBJETIVO / "tfg_deep_onmf",
            destino_src / "tfg_deep_onmf",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    explicacion = f"""# Programacion de la prueba para Juan

Esta carpeta contiene el codigo necesario para repetir la prueba pedida para el 26/05/2026.

## Archivos

- `01_ejecutar_prueba_inicializaciones.py`: ejecuta Deep ONMF con cuatro inicializaciones: aleatoria actual, NNDSVD, NNDSVDa y NNDSVDar.
- `02_EJECUTAR_PRUEBA_JUAN_INICIALIZACIONES.bat`: acceso directo para lanzarlo en Windows.
- `requirements_prueba_juan.txt`: librerias necesarias.
- `src_tfg_deep_onmf_usado/`: copia del codigo base del proyecto que se ha usado como apoyo.

## Que cambia respecto al Deep ONMF actual

En el codigo original, `W` y `H` se inicializan asi:

```python
w = rng.random((filas, rango)) + eps
h = rng.random((rango, columnas)) + eps
```

En esta prueba se conserva esa version como referencia y se comparan tres inicializaciones comunes:

- `NNDSVD`
- `NNDSVDa`
- `NNDSVDar`

Los resultados se guardan en:

`{CARPETA_RESULTADOS}`
"""
    (CARPETA_PROGRAMACION / "00_LEEME_PROGRAMACION.md").write_text(explicacion, encoding="utf-8")


def tabla_markdown_metricas(metricas: pd.DataFrame) -> str:
    columnas = [
        "metodo",
        "error_relativo_final_medio",
        "ortogonalidad_media_capas",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    tabla = metricas[columnas].copy()
    for columna in columnas[1:]:
        tabla[columna] = tabla[columna].map(lambda valor: f"{valor:.6f}")
    lineas = [
        "| " + " | ".join(tabla.columns) + " |",
        "| " + " | ".join("---" for _ in tabla.columns) + " |",
    ]
    for fila in tabla.itertuples(index=False, name=None):
        lineas.append("| " + " | ".join(str(valor) for valor in fila) + " |")
    return "\n".join(lineas)


def dataframe_a_markdown(tabla: pd.DataFrame) -> str:
    tabla_texto = tabla.copy()
    for columna in tabla_texto.columns:
        tabla_texto[columna] = tabla_texto[columna].map(
            lambda valor: f"{valor:.6f}" if isinstance(valor, float) else str(valor)
        )
    lineas = [
        "| " + " | ".join(tabla_texto.columns) + " |",
        "| " + " | ".join("---" for _ in tabla_texto.columns) + " |",
    ]
    for fila in tabla_texto.itertuples(index=False, name=None):
        lineas.append("| " + " | ".join(str(valor) for valor in fila) + " |")
    return "\n".join(lineas)


def crear_figura_comparativa_tsne(rutas_por_metodo: dict[str, dict[str, Path]], metricas: pd.DataFrame, ruta: Path) -> None:
    fig, ejes = plt.subplots(2, 2, figsize=(15, 12))
    for eje, metodo in zip(ejes.ravel(), METODOS):
        coords = pd.read_csv(rutas_por_metodo[metodo.codigo]["coords_csv"], encoding="utf-8-sig")
        fila = metricas.loc[metricas["codigo_metodo"] == metodo.codigo].iloc[0]
        for clase in CLASES:
            mascara = coords["clase"] == clase
            eje.scatter(
                coords.loc[mascara, "tSNE_1"],
                coords.loc[mascara, "tSNE_2"],
                s=20,
                alpha=0.82,
                color=COLORES.get(clase, "#666666"),
                label=clase,
                edgecolors="none",
            )
        eje.set_title(
            f"{metodo.nombre}\nSil t-SNE={fila['silhouette_tsne']:.3f} | DB t-SNE={fila['davies_bouldin_tsne']:.3f}"
        )
        eje.set_xlabel("t-SNE 1")
        eje.set_ylabel("t-SNE 2")
        eje.grid(True, alpha=0.2)
    handles, labels = ejes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CLASES), title="Clase")
    fig.suptitle("Comparativa Figura 11D - Deep ONMF segun inicializacion W/H", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(ruta, dpi=220)
    plt.close(fig)


def crear_figura_tabla_metricas(metricas: pd.DataFrame, ruta: Path) -> None:
    columnas = [
        "metodo",
        "error_relativo_final_medio",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    tabla = metricas[columnas].copy()
    for columna in columnas[1:]:
        tabla[columna] = tabla[columna].map(lambda valor: f"{valor:.4f}")
    fig, eje = plt.subplots(figsize=(16, 4.8))
    eje.axis("off")
    objeto = eje.table(cellText=tabla.values, colLabels=tabla.columns, cellLoc="center", loc="center")
    objeto.auto_set_font_size(False)
    objeto.set_fontsize(8)
    objeto.scale(1, 1.8)
    eje.set_title("Tabla comparativa de inicializaciones Deep ONMF", fontsize=13, pad=18)
    fig.tight_layout()
    fig.savefig(ruta, dpi=220)
    plt.close(fig)


def escribir_informe_general(metricas: pd.DataFrame, auditoria: pd.DataFrame) -> None:
    mejor_tsne_sil = metricas.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_tsne_db = metricas.sort_values("davies_bouldin_tsne", ascending=True).iloc[0]
    mejor_error = metricas.sort_values("error_relativo_final_medio", ascending=True).iloc[0]
    lineas = [
        "# Prueba para Juan - Inicializaciones Deep ONMF",
        "",
        "## Objetivo",
        "",
        "Comparar el Deep ONMF actual con tres inicializaciones comunes de matrices W y H:",
        "",
        "- Implementacion actual con inicializacion aleatoria.",
        "- NNDSVD.",
        "- NNDSVDa.",
        "- NNDSVDar.",
        "",
        "La prueba mantiene constantes el resto de parametros para que la comparacion sea justa.",
        "",
        "## Parametros usados",
        "",
        "- Trama PCG: 2 segundos.",
        "- Solape: 1 segundo.",
        "- Rangos Deep ONMF: 9, 8 y 7.",
        "- Iteraciones por capa: 120.",
        "- Penalizacion ortogonal: 0.05.",
        "- Semilla base: 42.",
        "- Audios cortos: se rellenan con ceros para no eliminar muestras.",
        "",
        "## Auditoria de datos",
        "",
        dataframe_a_markdown(auditoria),
        "",
        "## Tabla comparativa",
        "",
        tabla_markdown_metricas(metricas),
        "",
        "## Lectura de la tabla",
        "",
        "- `error_relativo_final_medio`: cuanto menor, mejor reconstruye Deep ONMF las matrices de espectrograma.",
        "- `silhouette_features`: cuanto mayor, mejor separacion de clases en los rasgos SBV antes de t-SNE.",
        "- `davies_bouldin_features`: cuanto menor, mejor separacion/compactacion en rasgos SBV.",
        "- `silhouette_tsne`: cuanto mayor, mejor separacion visible en la Figura 11D.",
        "- `davies_bouldin_tsne`: cuanto menor, mejor separacion visible en la Figura 11D.",
        "",
        "## Resultado principal",
        "",
        f"- Mejor `silhouette_tsne`: **{mejor_tsne_sil['metodo']}** con {mejor_tsne_sil['silhouette_tsne']:.6f}.",
        f"- Mejor `davies_bouldin_tsne`: **{mejor_tsne_db['metodo']}** con {mejor_tsne_db['davies_bouldin_tsne']:.6f}.",
        f"- Menor error relativo medio: **{mejor_error['metodo']}** con {mejor_error['error_relativo_final_medio']:.6f}.",
        "",
        "## Donde mirar las fotos",
        "",
        "- `01_fotos_por_separado/`: todas las figuras de cada inicializacion por separado.",
        "- `comparativa_tSNE_inicializaciones.png`: las cuatro Figuras 11D una al lado de otra.",
        "- `tabla_metricas_inicializaciones.png`: tabla visual con las metricas clave.",
        "- `00_INFORME_COMPARATIVO_INICIALIZACIONES.pdf`: PDF final con explicacion y comparativa.",
        "",
        "## Nota para defenderlo",
        "",
        "Esta prueba no cambia la arquitectura Deep ONMF ni la base de datos. Solo cambia la forma de arrancar W y H.",
        "Si una inicializacion mejora la foto o las metricas, la explicacion es que el algoritmo empieza desde una base",
        "mas informativa que el azar y evita algunos minimos locales de las actualizaciones multiplicativas.",
        "",
    ]
    (CARPETA_RESULTADOS / "00_INFORME_COMPARATIVO.md").write_text("\n".join(lineas), encoding="utf-8")


def agregar_texto_pdf(pdf: PdfPages, titulo: str, lineas: list[str]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.94, titulo, fontsize=18, weight="bold", va="top")
    y = 0.88
    for linea in lineas:
        fig.text(0.06, y, linea, fontsize=10.5, va="top", wrap=True)
        y -= 0.036
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11.69, 8.27))
            y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def agregar_imagen_pdf(pdf: PdfPages, ruta: Path, titulo: str) -> None:
    imagen = plt.imread(ruta)
    fig, eje = plt.subplots(figsize=(11.69, 8.27))
    eje.imshow(imagen)
    eje.axis("off")
    eje.set_title(titulo, fontsize=14, pad=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def crear_pdf(metricas: pd.DataFrame, rutas_por_metodo: dict[str, dict[str, Path]]) -> None:
    ruta_pdf = CARPETA_RESULTADOS / "00_INFORME_COMPARATIVO_INICIALIZACIONES.pdf"
    mejor_tsne = metricas.sort_values("silhouette_tsne", ascending=False).iloc[0]
    lineas_portada = [
        "Prueba solicitada por Juan para comparar inicializaciones de W y H en Deep ONMF.",
        "Metodos comparados: aleatoria actual, NNDSVD, NNDSVDa y NNDSVDar.",
        "Todos usan la misma base, los mismos parametros y la misma generacion de figuras.",
        "",
        f"Mejor separacion visual por silhouette t-SNE: {mejor_tsne['metodo']} ({mejor_tsne['silhouette_tsne']:.4f}).",
        "",
        "El PDF incluye la tabla de metricas y las figuras principales. En las carpetas de resultados",
        "queda cada imagen por separado con su explicacion en Markdown.",
    ]
    with PdfPages(ruta_pdf) as pdf:
        agregar_texto_pdf(pdf, "Inicializaciones Deep ONMF - prueba 26/05/2026", lineas_portada)
        agregar_imagen_pdf(pdf, CARPETA_RESULTADOS / "tabla_metricas_inicializaciones.png", "Tabla de metricas")
        agregar_imagen_pdf(pdf, CARPETA_RESULTADOS / "comparativa_tSNE_inicializaciones.png", "Figura 11D comparada")
        for metodo in METODOS:
            agregar_imagen_pdf(pdf, rutas_por_metodo[metodo.codigo]["figura_11"], f"Figura 11D - {metodo.nombre}")
            agregar_imagen_pdf(pdf, rutas_por_metodo[metodo.codigo]["figura_5"], f"Figura 5 - {metodo.nombre}")
            agregar_imagen_pdf(pdf, rutas_por_metodo[metodo.codigo]["figura_7"], f"Figura 7 - {metodo.nombre}")


def main() -> int:
    crear_carpetas()
    copiar_programacion()
    copiar_resultados_existentes()

    configuracion = Configuracion(
        raiz=RAIZ_OBJETIVO,
        duracion_trama_s=2.0,
        solape_trama_s=1.0,
        rangos_onmf=(9, 8, 7),
        iteraciones_onmf=120,
        penalizacion_ortogonal=0.05,
        semilla=42,
        rellenar_audios_cortos=True,
    )

    print(f"Raiz objetivo: {RAIZ_OBJETIVO}")
    print(f"Carpeta de salida: {CARPETA_PRUEBA}")
    registros = descubrir_audios(configuracion.carpeta_base_datos, CLASES)
    datos_por_clase = {
        clase: construir_matriz_clase(clase, registros, configuracion)
        for clase in CLASES
    }
    auditoria = resumen_auditoria(registros, datos_por_clase, CLASES)
    auditoria.to_csv(CARPETA_RESULTADOS / "auditoria_datos_usados.csv", index=False, encoding="utf-8-sig")

    filas_metricas: list[dict[str, object]] = []
    detalle_capas: list[pd.DataFrame] = []
    rutas_por_metodo: dict[str, dict[str, Path]] = {}
    for metodo in METODOS:
        fila, capas, rutas = ejecutar_metodo(metodo, configuracion, datos_por_clase)
        filas_metricas.append(fila)
        detalle_capas.append(capas)
        rutas_por_metodo[metodo.codigo] = rutas

    metricas = pd.DataFrame(filas_metricas)
    metricas.to_csv(CARPETA_RESULTADOS / "metricas_inicializaciones_deep_onmf.csv", index=False, encoding="utf-8-sig")
    pd.concat(detalle_capas, ignore_index=True).to_csv(
        CARPETA_RESULTADOS / "detalle_capas_inicializaciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    escribir_json(
        CARPETA_RESULTADOS / "parametros_prueba_juan.json",
        {
            **configuracion.como_diccionario(),
            "metodos": [metodo.codigo for metodo in METODOS],
            "objetivo": "comparar inicializaciones W/H en Deep ONMF",
        },
    )

    crear_figura_comparativa_tsne(
        rutas_por_metodo,
        metricas,
        CARPETA_RESULTADOS / "comparativa_tSNE_inicializaciones.png",
    )
    crear_figura_tabla_metricas(metricas, CARPETA_RESULTADOS / "tabla_metricas_inicializaciones.png")
    copiar_archivo(
        CARPETA_RESULTADOS / "comparativa_tSNE_inicializaciones.png",
        CARPETA_FOTOS_SEPARADAS / "comparativa_tSNE_inicializaciones.png",
    )
    copiar_archivo(
        CARPETA_RESULTADOS / "tabla_metricas_inicializaciones.png",
        CARPETA_FOTOS_SEPARADAS / "tabla_metricas_inicializaciones.png",
    )

    escribir_informe_general(metricas, auditoria)
    crear_pdf(metricas, rutas_por_metodo)

    leeme = f"""# Pruebas para el 26-05-26

Carpeta creada para responder al correo de Juan.

## Estructura

- `programacion/`: codigo necesario para repetir la prueba.
- `resultados comparativos/`: figuras, tablas, metricas, explicaciones y PDF final.

## Resultado rapido

Consulta primero:

- `resultados comparativos/00_INFORME_COMPARATIVO.md`
- `resultados comparativos/00_INFORME_COMPARATIVO_INICIALIZACIONES.pdf`
- `resultados comparativos/metricas_inicializaciones_deep_onmf.csv`
- `resultados comparativos/comparativa_tSNE_inicializaciones.png`

"""
    (CARPETA_PRUEBA / "00_LEEME_PRUEBAS_26-05-26.md").write_text(leeme, encoding="utf-8")

    print("\nPrueba terminada.")
    print(f"Informe: {CARPETA_RESULTADOS / '00_INFORME_COMPARATIVO.md'}")
    print(f"PDF: {CARPETA_RESULTADOS / '00_INFORME_COMPARATIVO_INICIALIZACIONES.pdf'}")
    print(metricas[["metodo", "silhouette_tsne", "davies_bouldin_tsne", "error_relativo_final_medio"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
