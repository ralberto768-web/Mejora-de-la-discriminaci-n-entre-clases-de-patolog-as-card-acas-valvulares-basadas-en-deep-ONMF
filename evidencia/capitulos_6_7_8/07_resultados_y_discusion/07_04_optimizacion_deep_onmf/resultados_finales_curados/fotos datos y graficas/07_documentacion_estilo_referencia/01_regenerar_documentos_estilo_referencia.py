from __future__ import annotations

import json
import math
import shutil
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


FINAL = Path(__file__).resolve().parent
ROOT = FINAL.parent
SRC = ROOT / "src"
RESULTADOS = FINAL / "RESULTADOS"
MEDIA = RESULTADOS / "fotos datos y graficas"
OUT = MEDIA / "07_documentacion_estilo_referencia"
FIGS = OUT / "figuras"
TABLAS = OUT / "tablas"
VERIF = OUT / "verificacion"
F8_INIT = MEDIA / "08_f8_inicializaciones_profesor"
RESULTADO8_OUT = MEDIA / "09_resultado8_sin_descartar_2s"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_deep_onmf.audio import construir_matriz_audio, construir_matriz_clase, descubrir_audios  # noqa: E402
from tfg_deep_onmf.configuracion import Configuracion  # noqa: E402
from tfg_deep_onmf.estadistica import resumen_auditoria  # noqa: E402
from tfg_deep_onmf.onmf import proyectar_sobre_w  # noqa: E402

HIST_ROOT = ROOT / "comparacion final"
HIST_AJUSTADA = HIST_ROOT / "04_prueba_ajustada_codigo_y_resultados" / "resultados"
HIST_METRICAS = HIST_ROOT / "02_resultados" / "03_datos_y_metricas" / "metricas_comparacion_figura11.csv"
HIST_BARRIDO = HIST_AJUSTADA / "metricas_barrido_deep_onmf.csv"
HIST_MEJOR = HIST_AJUSTADA / "mejor_candidato_deep_onmf.json"
HIST_FINAL_CLAVE = HIST_AJUSTADA / "final_clave"
HIST_ZIP_COMPLETO = HIST_AJUSTADA / "comparacion_ajustada_completa.zip"
HIST_F8_RASGOS = HIST_AJUSTADA / "01_rasgos_candidatos" / "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8.csv"
HIST_F8_TSNE = HIST_AJUSTADA / "02_coordenadas_barrido" / "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8__tsne.csv"
HIST_CODIGO_BARRIDO = HIST_ROOT / "04_prueba_ajustada_codigo_y_resultados" / "codigo" / "05_barrido_rasgos_deep_onmf_ajustados.py"
HIST_CODIGO_PIPELINE = HIST_ROOT / "04_prueba_ajustada_codigo_y_resultados" / "codigo" / "src" / "tfg_deep_onmf" / "pipeline.py"
HIST_CODIGO_COMPARADOR = HIST_ROOT / "04_prueba_ajustada_codigo_y_resultados" / "codigo" / "comparador_figura11" / "01_codigos_ordenados" / "01_comparar_figura11.py"
HIST_CODIGO_COMPARAR_RASGOS = HIST_ROOT / "04_prueba_ajustada_codigo_y_resultados" / "codigo" / "03_comparar_con_rasgos_deep_onmf_ajustados.py"
CODIGO_GENERACION_FINAL = MEDIA / "01_codigo_generacion_final" / "01_generar_entrega_final.py"

ACTUAL_COMP = MEDIA / "03_comparativa_principal" / "metricas_comparativa_principal.csv"
ACTUAL_VARIANTES = MEDIA / "04_deep_onmf_mejorado_inicializaciones" / "metricas_deep_onmf_variantes.csv"
ACTUAL_CONV = MEDIA / "05_convergencia_0_300" / "convergencia_0_120_vs_120_300.csv"
ACTUAL_AUDITORIA = MEDIA / "02_implementacion_fiel_articulo" / "auditoria_articulo_fiel.csv"
ACTUAL_CAPAS = MEDIA / "02_implementacion_fiel_articulo" / "capas_deep_onmf_aleatoria.csv"

COLORES = {
    "N": "#1b9e77",
    "AS": "#d95f02",
    "MR": "#7570b3",
    "MS": "#e7298a",
    "MVP": "#66a61e",
}
CLASES = ("N", "AS", "MR", "MS", "MVP")
EPS = 1e-12


def preparar_salida() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for carpeta in (OUT, FIGS, TABLAS, VERIF):
        carpeta.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), OUT / "01_regenerar_documentos_estilo_referencia.py")


def leer_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def guardar_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def extraer_funcion_codigo(path: Path, nombre: str, max_lineas: int = 28) -> str:
    lineas = path.read_text(encoding="utf-8").splitlines()
    inicio = None
    for i, linea in enumerate(lineas):
        if linea.startswith(f"def {nombre}("):
            inicio = i
            break
    if inicio is None:
        return f"# No se encontro def {nombre} en {path.name}"
    fin = len(lineas)
    for j in range(inicio + 1, len(lineas)):
        if lineas[j].startswith("def ") and lineas[j][0] != " ":
            fin = j
            break
    return "\n".join(lineas[inicio : min(fin, inicio + max_lineas)])


def extraer_lineas_con_patron(path: Path, patrones: list[str], contexto: int = 2) -> str:
    lineas = path.read_text(encoding="utf-8").splitlines()
    indices: set[int] = set()
    for i, linea in enumerate(lineas):
        if any(p in linea for p in patrones):
            for j in range(max(0, i - contexto), min(len(lineas), i + contexto + 1)):
                indices.add(j)
    if not indices:
        return "# No se encontraron las lineas solicitadas"
    return "\n".join(lineas[i] for i in sorted(indices))


def fmt(x: object, dec: int = 4) -> str:
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{dec}f}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def etiqueta_corta(texto: str) -> str:
    return str(texto)


def slug(nombre: str) -> str:
    return (
        nombre.lower()
        .replace("+", "mas")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def columnas_rasgos(df: pd.DataFrame) -> list[str]:
    ignoradas = {"clase", "archivo", "ruta", "duracion_s", "variante_deep_onmf"}
    return [c for c in df.columns if c not in ignoradas and not c.startswith("error_")]


def normalizar_columnas_w(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    escala = np.maximum(np.linalg.norm(w, axis=0), EPS)
    return w / escala[None, :], h * escala[:, None]


def error_relativo(x: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    return float(np.linalg.norm(x - w @ h, ord="fro") / max(np.linalg.norm(x, ord="fro"), EPS))


def ortogonalidad_media(h: np.ndarray) -> float:
    normas = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    h_norm = h / normas
    gramo = h_norm @ h_norm.T
    mascara = ~np.eye(gramo.shape[0], dtype=bool)
    return float(np.mean(np.abs(gramo[mascara])))


def nndsvd_inicializar(x: np.ndarray, rango: int, variante: str, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(x.astype(np.float64, copy=False), EPS)
    media = float(np.mean(x))
    u, s, vt = randomized_svd(x, n_components=rango, random_state=semilla)
    w = np.zeros((x.shape[0], rango), dtype=np.float64)
    h = np.zeros((rango, x.shape[1]), dtype=np.float64)

    w[:, 0] = math.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = math.sqrt(s[0]) * np.abs(vt[0, :])

    for j in range(1, rango):
        uj = u[:, j]
        vj = vt[j, :]
        uj_pos, uj_neg = np.maximum(uj, 0), np.maximum(-uj, 0)
        vj_pos, vj_neg = np.maximum(vj, 0), np.maximum(-vj, 0)
        norm_pos = np.linalg.norm(uj_pos) * np.linalg.norm(vj_pos)
        norm_neg = np.linalg.norm(uj_neg) * np.linalg.norm(vj_neg)
        if norm_pos >= norm_neg:
            uu = uj_pos / max(np.linalg.norm(uj_pos), EPS)
            vv = vj_pos / max(np.linalg.norm(vj_pos), EPS)
            sigma = norm_pos
        else:
            uu = uj_neg / max(np.linalg.norm(uj_neg), EPS)
            vv = vj_neg / max(np.linalg.norm(vj_neg), EPS)
            sigma = norm_neg
        escala = math.sqrt(s[j] * sigma)
        w[:, j] = escala * uu
        h[j, :] = escala * vv

    if variante == "nndsvda":
        w[w <= EPS] = media
        h[h <= EPS] = media
    elif variante == "nndsvdar":
        rng = np.random.default_rng(semilla)
        w[w <= EPS] = media * rng.random(np.count_nonzero(w <= EPS)) / 100.0
        h[h <= EPS] = media * rng.random(np.count_nonzero(h <= EPS)) / 100.0
    elif variante != "nndsvd":
        raise ValueError(f"Variante NNDSVD no soportada: {variante}")

    return normalizar_columnas_w(np.maximum(w, EPS), np.maximum(h, EPS))


def factorizar_onmf_local(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
    metodo_init: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.maximum(matriz, EPS).astype(np.float64, copy=False)
    w, h = nndsvd_inicializar(x, rango, metodo_init, semilla)
    for _ in range(iteraciones):
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        w = np.maximum(w, EPS)
        h *= (w.T @ x + penalizacion_ortogonal * h) / (
            (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + EPS
        )
        h = np.maximum(h, EPS)
        w, h = normalizar_columnas_w(w, h)
    return w, h, error_relativo(x, w, h), ortogonalidad_media(h)


def entrenar_w_f8_init(datos_por_clase: dict[str, object], config: Configuracion, metodo: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    w_por_clase: dict[str, np.ndarray] = {}
    filas_capas: list[dict[str, object]] = []
    for pos, clase in enumerate(config.clases):
        entrada = np.maximum(datos_por_clase[clase].matriz, EPS)
        matrices_w: list[np.ndarray] = []
        for indice, rango in enumerate(config.rangos_onmf, start=1):
            inicio = time.perf_counter()
            w, h, error, ort = factorizar_onmf_local(
                entrada,
                rango=rango,
                iteraciones=config.iteraciones_onmf,
                penalizacion_ortogonal=config.penalizacion_ortogonal,
                semilla=config.semilla + pos * 17 + indice * 1000,
                metodo_init=metodo,
            )
            filas_capas.append(
                {
                    "metodo_init": metodo,
                    "clase": clase,
                    "capa": indice,
                    "rango": rango,
                    "entrada": f"{entrada.shape[0]}x{entrada.shape[1]}",
                    "W": f"{w.shape[0]}x{w.shape[1]}",
                    "H": f"{h.shape[0]}x{h.shape[1]}",
                    "error_relativo": error,
                    "ortogonalidad_media": ort,
                    "segundos": time.perf_counter() - inicio,
                }
            )
            matrices_w.append(w)
            entrada = h
        w_final = matrices_w[0] @ matrices_w[1] @ matrices_w[2]
        normas = np.maximum(np.linalg.norm(w_final, axis=0), EPS)
        w_por_clase[clase] = w_final / normas[None, :]
    return w_por_clase, pd.DataFrame(filas_capas)


def softmin_errores_f8(errores: np.ndarray, fuerza: float = 8.0) -> np.ndarray:
    relativos = errores / np.maximum(errores.min(axis=1, keepdims=True), EPS)
    logits = -fuerza * (relativos - 1.0)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def rasgos_f8_desde_w(registros: list[object], w_por_clase: dict[str, np.ndarray], config: Configuracion, nombre: str) -> pd.DataFrame:
    filas = []
    for i, registro in enumerate(registros, start=1):
        if i == 1 or i % 100 == 0 or i == len(registros):
            print(f"[f8-init] {nombre}: proyecciones {i}/{len(registros)}")
        matriz = construir_matriz_audio(registro, config)
        errores = []
        fila: dict[str, object] = {
            "clase": registro.clase,
            "archivo": registro.ruta.name,
            "ruta": str(registro.ruta),
            "duracion_s": registro.duracion_s,
            "variante_deep_onmf": nombre,
        }
        for clase_modelo in config.clases:
            _, err = proyectar_sobre_w(matriz, w_por_clase[clase_modelo])
            errores.append(err)
            fila[f"error_vs_{clase_modelo}"] = err
        afinidades = softmin_errores_f8(np.asarray(errores, dtype=np.float64)[None, :])[0]
        for indice, valor in enumerate(afinidades, start=1):
            fila[f"F_{indice:03d}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar_metodo_local(nombre: str, df: pd.DataFrame, carpeta: Path) -> tuple[pd.Series, pd.DataFrame]:
    cols = columnas_rasgos(df)
    x_original = df[cols].to_numpy(dtype=np.float64)
    labels = df["clase"].to_numpy()
    x = StandardScaler().fit_transform(x_original)
    if x.shape[1] > 50:
        x_tsne = PCA(n_components=50, random_state=42).fit_transform(x)
        dim_tsne = 50
    else:
        x_tsne = x
        dim_tsne = x.shape[1]
    perplexity = min(30, max(5, (len(df) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
        max_iter=1000,
    ).fit_transform(x_tsne)
    coords_df = df[["clase", "archivo"]].copy()
    coords_df["tsne_1"] = coords[:, 0]
    coords_df["tsne_2"] = coords[:, 1]
    guardar_csv(coords_df, carpeta / f"coordenadas_tsne_{slug(nombre)}.csv")
    return (
        pd.Series(
            {
                "metodo": nombre,
                "muestras": len(df),
                "rasgos_originales": len(cols),
                "rasgos_entrada_tsne": dim_tsne,
                "perplexity": perplexity,
                "silhouette_features": silhouette_score(x, labels),
                "davies_bouldin_features": davies_bouldin_score(x, labels),
                "silhouette_tsne": silhouette_score(coords, labels),
                "davies_bouldin_tsne": davies_bouldin_score(coords, labels),
            }
        ),
        coords_df,
    )


def preparar_f8_inicializaciones_profesor() -> pd.DataFrame:
    F8_INIT.mkdir(parents=True, exist_ok=True)
    metricas_path = F8_INIT / "metricas_f8_inicializaciones_profesor.csv"
    if metricas_path.exists():
        return leer_csv(metricas_path)

    config = Configuracion(raiz=ROOT, rellenar_audios_cortos=True)
    registros = descubrir_audios(config.carpeta_base_datos, config.clases)
    datos_por_clase = {clase: construir_matriz_clase(clase, registros, config) for clase in config.clases}
    guardar_csv(resumen_auditoria(registros, datos_por_clase, config.clases), F8_INIT / "auditoria_f8_1000_audios.csv")

    filas_metricas = []
    for metodo, sufijo in [("nndsvd", "NNDSVD"), ("nndsvda", "NNDSVDa"), ("nndsvdar", "NNDSVDar")]:
        nombre = f"resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8 + {sufijo}"
        print(f"[f8-init] entrenando {nombre}")
        w_por_clase, capas = entrenar_w_f8_init(datos_por_clase, config, metodo)
        guardar_csv(capas, F8_INIT / f"capas_{metodo}.csv")
        np.savez_compressed(F8_INIT / f"matrices_w_finales_{metodo}.npz", **w_por_clase)
        rasgos = rasgos_f8_desde_w(registros, w_por_clase, config, nombre)
        guardar_csv(rasgos, F8_INIT / f"rasgos_{slug(nombre)}.csv")
        fila, _ = evaluar_metodo_local(nombre, rasgos, F8_INIT)
        filas_metricas.append(fila)
    metricas = pd.DataFrame(filas_metricas)
    guardar_csv(metricas, metricas_path)
    return metricas


def preparar_resultado8_sin_descartar_2s() -> dict[str, object]:
    if RESULTADO8_OUT.exists():
        shutil.rmtree(RESULTADO8_OUT)
    datos_dir = RESULTADO8_OUT / "datos_extraidos"
    codigo_dir = RESULTADO8_OUT / "codigo_usado"
    figuras_dir = RESULTADO8_OUT / "figuras"
    for carpeta in (RESULTADO8_OUT, datos_dir, codigo_dir, figuras_dir):
        carpeta.mkdir(parents=True, exist_ok=True)

    csvs = [
        "coordenadas_tsne_cnn.csv",
        "coordenadas_tsne_dwt.csv",
        "coordenadas_tsne_mfcc.csv",
        "coordenadas_tsne_stft.csv",
        "coordenadas_tsne_deep_onmf.csv",
        "rasgos_cnn.csv",
        "rasgos_dwt.csv",
        "rasgos_mfcc.csv",
        "rasgos_stft.csv",
        "rasgos_deep_onmf.csv",
        "metricas_comparacion_figura11.csv",
    ]
    with zipfile.ZipFile(HIST_ZIP_COMPLETO) as zf:
        for nombre in csvs:
            miembro = f"02_resultados/03_datos_y_metricas/{nombre}"
            with zf.open(miembro) as src, (datos_dir / nombre).open("wb") as dst:
                shutil.copyfileobj(src, dst)

    if HIST_CODIGO_COMPARADOR.exists():
        shutil.copyfile(HIST_CODIGO_COMPARADOR, codigo_dir / "01_comparar_figura11.py")
    if HIST_CODIGO_COMPARAR_RASGOS.exists():
        shutil.copyfile(HIST_CODIGO_COMPARAR_RASGOS, codigo_dir / "03_comparar_con_rasgos_deep_onmf_ajustados.py")

    coords = {
        "CNN": datos_dir / "coordenadas_tsne_cnn.csv",
        "DWT": datos_dir / "coordenadas_tsne_dwt.csv",
        "MFCC": datos_dir / "coordenadas_tsne_mfcc.csv",
        "STFT": datos_dir / "coordenadas_tsne_stft.csv",
        "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8": datos_dir / "coordenadas_tsne_deep_onmf.csv",
    }
    return {
        "base": RESULTADO8_OUT,
        "datos": datos_dir,
        "codigo": codigo_dir,
        "figuras": figuras_dir,
        "coords": coords,
        "metricas_zip": datos_dir / "metricas_comparacion_figura11.csv",
    }


def calcular_mejoras_951_vs_1000(actual: pd.DataFrame, resultado8: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    for metodo in ["CNN", "DWT", "MFCC", "STFT"]:
        fila_951 = actual.loc[actual["metodo"] == metodo].iloc[0]
        fila_1000 = resultado8.loc[resultado8["metodo"] == metodo].iloc[0]
        sil_951 = float(fila_951["silhouette_tsne"])
        sil_1000 = float(fila_1000["silhouette_tsne"])
        db_951 = float(fila_951["davies_bouldin_tsne"])
        db_1000 = float(fila_1000["davies_bouldin_tsne"])
        filas.append(
            {
                "metodo": metodo,
                "sil_tSNE_951": sil_951,
                "sil_tSNE_1000": sil_1000,
                "delta_sil_tSNE": sil_1000 - sil_951,
                "mejora_pct_sil_tSNE": ((sil_1000 / max(sil_951, EPS)) - 1.0) * 100.0,
                "DB_tSNE_951": db_951,
                "DB_tSNE_1000": db_1000,
                "delta_DB_tSNE": db_1000 - db_951,
                "reduccion_pct_DB_tSNE": (1.0 - db_1000 / max(db_951, EPS)) * 100.0,
            }
        )
    return pd.DataFrame(filas)


def mejoras_951_1000_pdf(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "metodo",
            "sil_tSNE_951",
            "sil_tSNE_1000",
            "mejora_pct_sil_tSNE",
            "DB_tSNE_951",
            "DB_tSNE_1000",
            "reduccion_pct_DB_tSNE",
        ]
    ].rename(
        columns={
            "sil_tSNE_951": "sil 951",
            "sil_tSNE_1000": "sil 1000",
            "mejora_pct_sil_tSNE": "% sil",
            "DB_tSNE_951": "DB 951",
            "DB_tSNE_1000": "DB 1000",
            "reduccion_pct_DB_tSNE": "% DB",
        }
    )


def cargar_datos() -> dict[str, object]:
    hist_metricas = leer_csv(HIST_METRICAS)
    hist_barrido = leer_csv(HIST_BARRIDO)
    hist_final = leer_csv(HIST_FINAL_CLAVE / "metricas_comparacion_ajustada.csv")
    actual_comp = leer_csv(ACTUAL_COMP)
    actual_variantes = leer_csv(ACTUAL_VARIANTES)
    actual_conv = leer_csv(ACTUAL_CONV)
    auditoria = leer_csv(ACTUAL_AUDITORIA)
    capas = leer_csv(ACTUAL_CAPAS)
    mejor = json.loads(HIST_MEJOR.read_text(encoding="utf-8"))
    f8_init = preparar_f8_inicializaciones_profesor()
    resultado8_assets = preparar_resultado8_sin_descartar_2s()

    f8_exacto = hist_final.loc[hist_final["metodo"] == "Deep ONMF"].copy()
    f8_exacto["metodo"] = "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8"
    f8_exacto["protocolo"] = "resultado8_deep_onmf_sin_descartar_menores_2s"
    f8_exacto["bloque"] = "F8 exacto con 1000 audios"

    f8_init = f8_init.copy()
    f8_init["protocolo"] = "resultado8_deep_onmf_sin_descartar_menores_2s"
    f8_init["bloque"] = "F8 + NNDSVD/NNDSVDa/NNDSVDar con 1000 audios"

    actual_variantes = actual_variantes.copy()
    actual_variantes["protocolo"] = "Fiel al articulo"
    actual_variantes["bloque"] = "Deep-ONMF mejorado + NNDSVD/NNDSVDa/NNDSVDar con 951 audios"
    actual_comp = actual_comp.copy()
    actual_comp["protocolo"] = "Fiel al articulo"
    actual_comp["bloque"] = "Protocolo fiel al articulo (951 audios)"

    v1_resultado8 = hist_final.copy()
    v1_resultado8["protocolo"] = "resultado8_deep_onmf_sin_descartar_menores_2s"
    v1_resultado8["bloque"] = "Protocolo resultado8 (1000 audios)"
    v1_resultado8["metodo"] = v1_resultado8["metodo"].replace(
        {"Deep ONMF": "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8"}
    )
    cols_v1 = [
        "bloque",
        "protocolo",
        "metodo",
        "muestras",
        "rasgos_originales",
        "rasgos_entrada_tsne",
        "perplexity",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    v1_total = pd.concat([actual_comp[cols_v1], v1_resultado8[cols_v1]], ignore_index=True)
    mejoras_951_1000 = calcular_mejoras_951_vs_1000(actual_comp, v1_resultado8)

    cols_comp = [
        "bloque",
        "protocolo",
        "metodo",
        "muestras",
        "rasgos_originales",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    comparacion_clave = pd.concat(
        [
            f8_exacto[cols_comp],
            f8_init[cols_comp],
            actual_variantes[cols_comp],
        ],
        ignore_index=True,
    )
    comparacion_clave["nombre_corto"] = comparacion_clave["metodo"].map(etiqueta_corta)

    ranking_barrido = hist_barrido.copy()
    ranking_barrido["nombre_corto"] = ranking_barrido["variante"].map(etiqueta_corta)
    ranking_barrido = ranking_barrido.sort_values("silhouette_tsne", ascending=False)

    protocolos = pd.DataFrame(
        [
            {
                "protocolo": "resultado8_deep_onmf_sin_descartar_menores_2s",
                "muestras": 1000,
                "tratamiento audios cortos": "no se descartan; se rellenan con ceros hasta 2 s",
                "mejor fila": "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8",
                "uso": "resultado f8 defendible y comparacion con inicializaciones del profesor",
            },
            {
                "protocolo": "Fiel al articulo",
                "muestras": int(actual_comp["muestras"].iloc[0]),
                "tratamiento audios cortos": "se descartan menores de 2 s: 16 MR, 14 MS y 19 MVP",
                "mejor fila": "Deep-ONMF mejorado + NNDSVD",
                "uso": "reproduccion metodologica estricta con 951 audios",
            },
        ]
    )

    return {
        "hist_metricas": hist_metricas,
        "hist_final": hist_final,
        "hist_barrido": hist_barrido,
        "f8_exacto": f8_exacto,
        "f8_init": f8_init,
        "actual_comp": actual_comp,
        "actual_variantes": actual_variantes,
        "actual_conv": actual_conv,
        "auditoria": auditoria,
        "capas": capas,
        "mejor": mejor,
        "comparacion_clave": comparacion_clave,
        "ranking_barrido": ranking_barrido,
        "protocolos": protocolos,
        "v1_resultado8": v1_resultado8,
        "v1_total": v1_total,
        "resultado8_assets": resultado8_assets,
        "mejoras_951_1000": mejoras_951_1000,
    }


def guardar_tablas(datos: dict[str, object]) -> None:
    for nombre in ("comparacion_clave", "ranking_barrido", "protocolos", "v1_total"):
        guardar_csv(datos[nombre], TABLAS / f"{nombre}.csv")
    guardar_csv(metricas_v2_decision(datos["comparacion_clave"]), TABLAS / "metricas_v2_comparativa_decision.csv")
    guardar_csv(convergencia_v2_tabla(datos["actual_conv"]), TABLAS / "tabla_v2_convergencia_resumen.csv")
    guardar_csv(datos["v1_total"], TABLAS / "metricas_v1_comparativa_total.csv")
    guardar_csv(metricas_v1_rasgos(datos["v1_total"]), TABLAS / "metricas_v1_rasgos.csv")
    guardar_csv(metricas_v1_tsne(datos["v1_total"]), TABLAS / "metricas_v1_tsne.csv")
    guardar_csv(lectura_docente_v1(), TABLAS / "lectura_docente_v1.csv")
    guardar_csv(diferencias_deep_f8(), TABLAS / "diferencias_deep_onmf_f8.csv")
    guardar_csv(ejemplo_softmin_f8(), TABLAS / "ejemplo_softmin_errores_f8.csv")
    guardar_csv(datos["mejoras_951_1000"], TABLAS / "mejoras_v1_951_vs_1000.csv")
    guardar_csv(mejoras_951_1000_pdf(datos["mejoras_951_1000"]), TABLAS / "mejoras_v1_951_vs_1000_compacta.csv")
    guardar_csv(datos["mejoras_951_1000"], RESULTADO8_OUT / "mejoras_v1_951_vs_1000.csv")
    guardar_csv(datos["v1_total"], RESULTADO8_OUT / "metricas_v1_951_vs_1000.csv")
    guardar_csv(resumen_capas(datos["capas"]), TABLAS / "resumen_capas_pdf.csv")


def resumen_capas(capas: pd.DataFrame) -> pd.DataFrame:
    resumen = (
        capas.groupby(["capa", "rango"], as_index=False)
        .agg(
            error_medio=("error_relativo", "mean"),
            error_min=("error_relativo", "min"),
            error_max=("error_relativo", "max"),
            ortogonalidad_media=("ortogonalidad_media", "mean"),
            segundos_totales=("segundos", "sum"),
        )
        .sort_values("capa")
    )
    resumen["lectura"] = [
        "primera compresion espectral amplia",
        "segunda capa: patrones mas compactos",
        "tercera capa: 7 SBV finales",
    ][: len(resumen)]
    return resumen


def nombre_v1(texto: str) -> str:
    return str(texto).replace(
        "resultado8_deep_onmf_sin_descartar_menores_2s__",
        "resultado8_deep_onmf_sin_descartar_menores_2s__\n",
    )


def protocolo_corto(texto: str) -> str:
    if str(texto) == "Fiel al articulo":
        return "Fiel articulo (951)"
    if str(texto) == "resultado8_deep_onmf_sin_descartar_menores_2s":
        return "resultado8 (1000)"
    return str(texto)


def metricas_v1_rasgos(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "metodo",
            "muestras",
            "rasgos_originales",
            "silhouette_features",
            "davies_bouldin_features",
        ]
    ].copy()
    out["metodo"] = out["metodo"].map(nombre_v1)
    return out.rename(
        columns={
            "rasgos_originales": "rasgos",
            "silhouette_features": "sil rasgos",
            "davies_bouldin_features": "DB rasgos",
        }
    )


def metricas_v1_tsne(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "metodo",
            "muestras",
            "rasgos_entrada_tsne",
            "perplexity",
            "silhouette_tsne",
            "davies_bouldin_tsne",
        ]
    ].copy()
    out["metodo"] = out["metodo"].map(nombre_v1)
    return out.rename(
        columns={
            "rasgos_entrada_tsne": "rasgos t-SNE",
            "silhouette_tsne": "sil t-SNE",
            "davies_bouldin_tsne": "DB t-SNE",
        }
    )


def lectura_docente_v1() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metodo": "CNN",
                "idea": "aprende filtros 3x3 sobre espectrogramas",
                "lectura": "separa si las texturas tiempo-frecuencia son estables, pero depende del entrenamiento supervisado",
            },
            {
                "metodo": "DWT",
                "idea": "descompone la senal con coif5 por escalas",
                "lectura": "detecta transitorios, aunque la mezcla entre clases aumenta Davies-Bouldin cuando las escalas no son exclusivas",
            },
            {
                "metodo": "MFCC",
                "idea": "resume la envolvente con 40 filtros Mel y 13 coeficientes",
                "lectura": "es compacto y robusto, pero puede perder informacion temporal de soplos y cierres valvulares",
            },
            {
                "metodo": "STFT",
                "idea": "mide energia tiempo-frecuencia con Hamming 150, salto 75 y FFT 250",
                "lectura": "sirve como base interpretable, aunque como rasgo directo no aprende diccionarios por clase",
            },
            {
                "metodo": "Deep-ONMF normal",
                "idea": "aprende bases no negativas W1 W2 W3 con rangos 9-8-7",
                "lectura": "mejora la estructura espectral, pero leer solo SBV/activaciones no explota toda la separacion por clase",
            },
            {
                "metodo": "Deep-ONMF mejorado",
                "idea": "resume cada audio como afinidades de reconstruccion por clase",
                "lectura": "plantea una pregunta mas discriminante: que base reconstruye mejor el audio",
            },
            {
                "metodo": "perfil_softmin_errores_f8",
                "idea": "convierte errores relativos contra bases de clase en un perfil softmin de 5 rasgos",
                "lectura": "es la version con mejor equilibrio global encontrado: sil t-SNE 0.2602 y DB t-SNE 1.3756",
            },
        ]
    )


def diferencias_deep_f8() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bloque": "Deep-ONMF normal",
                "salida": "7 SBV",
                "pregunta que responde": "que bases espectrales se activan en el audio",
                "efecto": "representacion interpretable, pero no compara directamente contra todas las clases",
            },
            {
                "bloque": "Deep-ONMF mejorado",
                "salida": "5 afinidades",
                "pregunta que responde": "que base de clase reconstruye mejor cada audio",
                "efecto": "mejora la discriminacion porque convierte reconstruccion en evidencia por clase",
            },
            {
                "bloque": "perfil_softmin_errores_f8",
                "salida": "5 afinidades con fuerza 8",
                "pregunta que responde": "cuanto gana la mejor reconstruccion frente a las demas",
                "efecto": "acentua diferencias relativas y logra 0.2602 de silhouette t-SNE con 1000 audios",
            },
        ]
    )


def ejemplo_softmin_f8() -> pd.DataFrame:
    errores = np.asarray([0.10, 0.12, 0.30, 0.40, 0.50], dtype=np.float64)
    relativos = errores / errores.min()
    logits = -8.0 * (relativos - 1.0)
    logits -= logits.max()
    afinidades = np.exp(logits)
    afinidades = afinidades / afinidades.sum()
    return pd.DataFrame(
        {
            "clase": ["N", "AS", "MR", "MS", "MVP"],
            "error ejemplo": errores,
            "error relativo": relativos,
            "logit f8": logits,
            "afinidad softmin": afinidades,
        }
    )


def figura_tabla_metricas(df: pd.DataFrame, path: Path, titulo: str) -> None:
    mostrar = df.copy()
    for col in mostrar.columns:
        if pd.api.types.is_float_dtype(mostrar[col]):
            mostrar[col] = mostrar[col].map(lambda x: f"{x:.4f}")
        else:
            mostrar[col] = mostrar[col].map(lambda x: "\n".join(textwrap.wrap(str(x), width=30)))
    alto = max(4.8, 0.55 * len(mostrar) + 1.4)
    fig, ax = plt.subplots(figsize=(7.4, alto))
    ax.axis("off")
    tabla = ax.table(cellText=mostrar.values, colLabels=mostrar.columns, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(6.7)
    tabla.scale(1, 1.65)
    for (row, _col), cell in tabla.get_celld().items():
        cell.set_edgecolor("#b8bec8")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#d8e6f3")
            cell.set_text_props(weight="bold", color="#0a2942")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f7f9fb")
    ax.set_title(titulo, fontsize=11, pad=16, color="#0a2942")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figura_ranking_tsne(comparacion: pd.DataFrame, path: Path) -> None:
    data = comparacion.sort_values("silhouette_tsne", ascending=True).copy()
    colores_bloque = {
        "F8 exacto con 1000 audios": "#2f6f9f",
        "F8 + NNDSVD/NNDSVDa/NNDSVDar con 1000 audios": "#427f58",
        "Deep-ONMF mejorado + NNDSVD/NNDSVDa/NNDSVDar con 951 audios": "#7a9e3f",
    }
    colors = [colores_bloque.get(b, "#7a9e3f") for b in data["bloque"]]
    labels = ["\n".join(textwrap.wrap(str(m), width=38)) for m in data["metodo"]]
    fig, ax = plt.subplots(figsize=(8.0, max(7.6, 0.72 * len(data))))
    ax.barh(labels, data["silhouette_tsne"], color=colors)
    ax.set_xlabel("Silhouette t-SNE")
    ax.set_title("Ranking de separacion visual t-SNE")
    ax.grid(True, axis="x", alpha=0.25)
    for i, v in enumerate(data["silhouette_tsne"]):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.set_xlim(0, max(data["silhouette_tsne"]) * 1.20)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figura_ranking_db(comparacion: pd.DataFrame, path: Path) -> None:
    data = comparacion.sort_values("davies_bouldin_tsne", ascending=False).copy()
    colores_bloque = {
        "F8 exacto con 1000 audios": "#2f6f9f",
        "F8 + NNDSVD/NNDSVDa/NNDSVDar con 1000 audios": "#427f58",
        "Deep-ONMF mejorado + NNDSVD/NNDSVDa/NNDSVDar con 951 audios": "#7a9e3f",
    }
    colors = [colores_bloque.get(b, "#7a9e3f") for b in data["bloque"]]
    labels = ["\n".join(textwrap.wrap(str(m), width=38)) for m in data["metodo"]]
    fig, ax = plt.subplots(figsize=(8.0, max(7.6, 0.72 * len(data))))
    ax.barh(labels, data["davies_bouldin_tsne"], color=colors)
    ax.set_xlabel("Davies-Bouldin t-SNE")
    ax.set_title("Ranking de compacidad visual: menor es mejor")
    ax.grid(True, axis="x", alpha=0.25)
    for i, v in enumerate(data["davies_bouldin_tsne"]):
        ax.text(v + 0.04, i, f"{v:.4f}", va="center", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figura_barrido_historico(ranking: pd.DataFrame, path: Path) -> None:
    top = ranking.head(10).sort_values("silhouette_tsne", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    ax.barh(top["variante"], top["silhouette_tsne"], color="#2f6f9f")
    ax.set_xlabel("Silhouette t-SNE")
    ax.set_title("Barrido historico de rasgos Deep-ONMF")
    ax.grid(True, axis="x", alpha=0.25)
    for i, v in enumerate(top["silhouette_tsne"]):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=7.8)
    ax.set_xlim(0, max(top["silhouette_tsne"]) * 1.22)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figura_mejora_relativa(hist_barrido: pd.DataFrame, path: Path) -> pd.DataFrame:
    base = hist_barrido.loc[hist_barrido["variante"] == "sbv_media_base"].iloc[0]
    rows = []
    for _, row in hist_barrido.iterrows():
        if row["variante"] in {
            "sbv_media_base",
            "perfil_softmin_errores_f4",
            "perfil_softmin_errores_f8",
            "perfil_softmin_errores_f12",
            "perfil_softmin_errores_f16",
            "perfil_softmin_errores_f24",
        }:
            rows.append(
                {
                    "variante": row["variante"],
                    "silhouette_tsne": row["silhouette_tsne"],
                    "factor_silhouette_vs_base": row["silhouette_tsne"] / max(base["silhouette_tsne"], 1e-12),
                    "reduccion_db_vs_base": base["davies_bouldin_tsne"] / max(row["davies_bouldin_tsne"], 1e-12),
                }
            )
    df = pd.DataFrame(rows).sort_values("factor_silhouette_vs_base", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    ax.barh(df["variante"], df["factor_silhouette_vs_base"], color="#315f72")
    ax.axvline(1.0, color="#8b1e3f", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Factor de mejora de silhouette t-SNE frente a SBV base")
    ax.set_title("Cuanto mejora cada perfil frente al Deep-ONMF normal")
    ax.grid(True, axis="x", alpha=0.25)
    for i, v in enumerate(df["factor_silhouette_vs_base"]):
        ax.text(v + 0.04, i, f"x{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    guardar_csv(df, TABLAS / "mejora_relativa_vs_sbv_base.csv")
    return df


def figura_esquema_softmin(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    ax.axis("off")
    boxes = [
        ("Audio PCG", 0.5, 0.90, "#d9e8f5"),
        ("STFT\ntramas 2 s, solape 1 s", 0.5, 0.76, "#e7f0d6"),
        ("Bases Deep-ONMF por clase\nW_N, W_AS, W_MR, W_MS, W_MVP", 0.5, 0.60, "#f5ead7"),
        ("Errores de reconstruccion\ncontra cada base", 0.5, 0.43, "#f0d7df"),
        ("Softmin f8\nconvierte errores en afinidades", 0.5, 0.27, "#e2ddf2"),
        ("Vector final de 5 rasgos\nperfil de pertenencia por clase", 0.5, 0.12, "#d8efe5"),
    ]
    for text, x, y, color in boxes:
        ax.add_patch(
            plt.Rectangle((0.12, y - 0.055), 0.76, 0.11, facecolor=color, edgecolor="#345", linewidth=1.2)
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        _, x1, y1, _ = boxes[i]
        _, x2, y2, _ = boxes[i + 1]
        ax.annotate("", xy=(x2, y2 + 0.065), xytext=(x1, y1 - 0.065), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("Por que perfil_softmin_errores_f8 separa mejor", fontsize=14, pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def coords_path_actual(nombre: str) -> Path:
    slug = (
        nombre.lower()
        .replace("+", "mas")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    if "nndsvd" in slug or "normal" in slug or "mejorado" in slug:
        base = MEDIA / "04_deep_onmf_mejorado_inicializaciones"
    else:
        base = MEDIA / "03_comparativa_principal"
    return base / f"coordenadas_tsne_{slug}.csv"


def coords_path_f8_init(nombre: str) -> Path:
    return F8_INIT / f"coordenadas_tsne_{slug(nombre)}.csv"


def figura_tsne_desde_csv(metodos_y_paths: list[tuple[str, Path]], metricas: pd.DataFrame, path: Path, titulo: str) -> None:
    n = len(metodos_y_paths)
    if n == 1:
        fig, ax_unico = plt.subplots(1, 1, figsize=(7.2, 5.6))
        axes = np.asarray([ax_unico])
    else:
        filas = math.ceil(n / 2)
        fig, axes = plt.subplots(filas, 2, figsize=(7.4, 3.8 * filas))
        axes = np.asarray(axes).reshape(-1)
    for ax, (metodo, ruta_csv) in zip(axes, metodos_y_paths):
        if not ruta_csv.exists():
            ax.axis("off")
            ax.set_title("\n".join(textwrap.wrap(metodo, width=42)), fontsize=8)
            continue
        df = leer_csv(ruta_csv)
        df = df.rename(
            columns={
                "tSNE_1": "tsne_1",
                "tSNE_2": "tsne_2",
                "t-SNE 1": "tsne_1",
                "t-SNE 2": "tsne_2",
                "tsne1": "tsne_1",
                "tsne2": "tsne_2",
            }
        )
        fila_metrica = metricas.loc[metricas["metodo"] == metodo]
        subtitulo = "\n".join(textwrap.wrap(metodo, width=42))
        if not fila_metrica.empty:
            m = fila_metrica.iloc[0]
            subtitulo += f"\nSil={m['silhouette_tsne']:.4f} | DB={m['davies_bouldin_tsne']:.4f}"
        for clase in CLASES:
            mask = df["clase"] == clase
            ax.scatter(df.loc[mask, "tsne_1"], df.loc[mask, "tsne_2"], s=8, alpha=0.76, color=COLORES[clase], label=clase)
        ax.set_title(subtitulo, fontsize=8)
        ax.set_xlabel("t-SNE 1", fontsize=7.5)
        ax.set_ylabel("t-SNE 2", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.18)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CLASES), fontsize=8, title="Clase")
    fig.suptitle("\n".join(textwrap.wrap(titulo, width=74)), fontsize=12.0)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def extraer_figura_comparacion_ajustada(path: Path) -> Path | None:
    destino = path / "Comparacion_Figura11_lado_a_lado_f8.png"
    if destino.exists():
        return destino
    if not HIST_ZIP_COMPLETO.exists():
        return None
    miembro = "02_resultados/01_figuras_separadas/04_Figura11D_Deep_ONMF.png"
    lado_a_lado = "02_resultados/02_pdf_comparativo/Comparacion_Figura11_lado_a_lado.png"
    with zipfile.ZipFile(HIST_ZIP_COMPLETO) as zf:
        for candidato in (lado_a_lado, miembro):
            if candidato in zf.namelist():
                with zf.open(candidato) as src, destino.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                return destino
    return None


def figura_tsne_vertical(metodos: list[str], path: Path, titulo: str) -> None:
    n = len(metodos)
    filas = math.ceil(n / 2)
    fig, axes = plt.subplots(filas, 2, figsize=(7.4, 3.7 * filas))
    axes = np.asarray(axes).reshape(-1)
    for ax, metodo in zip(axes, metodos):
        p = coords_path_actual(metodo)
        if not p.exists():
            ax.axis("off")
            ax.set_title(etiqueta_corta(metodo))
            continue
        df = leer_csv(p)
        for clase in CLASES:
            mask = df["clase"] == clase
            ax.scatter(df.loc[mask, "tsne_1"], df.loc[mask, "tsne_2"], s=7, alpha=0.75, color=COLORES[clase], label=clase)
        ax.set_title(etiqueta_corta(metodo), fontsize=9)
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.18)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figura_tsne_v1_metodos_articulo_y_f8(datos: dict[str, object], path: Path) -> None:
    metodo_f8 = "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8"
    metodos_y_paths = [
        ("CNN", coords_path_actual("CNN")),
        ("DWT", coords_path_actual("DWT")),
        ("MFCC", coords_path_actual("MFCC")),
        ("STFT", coords_path_actual("STFT")),
        ("Deep-ONMF normal", coords_path_actual("Deep-ONMF normal")),
        ("Deep-ONMF mejorado", coords_path_actual("Deep-ONMF mejorado")),
        (metodo_f8, HIST_F8_TSNE),
    ]
    metricas = pd.concat([datos["actual_comp"], datos["f8_exacto"]], ignore_index=True, sort=False)
    figura_tsne_desde_csv(
        metodos_y_paths,
        metricas,
        path,
        "t-SNE de CNN, DWT, MFCC, STFT, Deep-ONMF normal, Deep-ONMF mejorado y perfil_softmin_errores_f8",
    )


def figura_tsne_resultado8_sin_descartar_1000(datos: dict[str, object], path: Path) -> None:
    coords = datos["resultado8_assets"]["coords"]
    metodos = [
        "CNN",
        "DWT",
        "MFCC",
        "STFT",
        "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8",
    ]
    figura_tsne_desde_csv(
        [(metodo, coords[metodo]) for metodo in metodos],
        datos["v1_resultado8"],
        path,
        "t-SNE del protocolo resultado8_deep_onmf_sin_descartar_menores_2s con 1000 audios",
    )


def figura_mejora_951_vs_1000(mejoras: pd.DataFrame, path: Path, tipo: str) -> None:
    data = mejoras.copy()
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if tipo == "silhouette":
        col_951, col_1000 = "sil_tSNE_951", "sil_tSNE_1000"
        etiqueta = "Silhouette t-SNE"
        titulo = "Comparacion silhouette t-SNE: 951 vs 1000 audios"
        anot = data["mejora_pct_sil_tSNE"].map(lambda v: f"{v:+.2f}%")
        mayor_mejor = True
    else:
        col_951, col_1000 = "DB_tSNE_951", "DB_tSNE_1000"
        etiqueta = "Davies-Bouldin t-SNE"
        titulo = "Comparacion Davies-Bouldin t-SNE: 951 vs 1000 audios"
        anot = data["reduccion_pct_DB_tSNE"].map(lambda v: f"{v:+.2f}%")
        mayor_mejor = False
    ax.bar(x - width / 2, data[col_951], width, label="951 audios", color="#7a9e3f")
    ax.bar(x + width / 2, data[col_1000], width, label="1000 audios", color="#2f6f9f")
    ax.set_xticks(x)
    ax.set_xticklabels(data["metodo"])
    ax.set_ylabel(etiqueta)
    ax.set_title(titulo)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for i, texto in enumerate(anot):
        y = max(float(data.loc[i, col_951]), float(data.loc[i, col_1000]))
        ax.text(i, y * 1.03, texto, ha="center", va="bottom", fontsize=8)
    nota = "mayor es mejor" if mayor_mejor else "menor es mejor"
    ax.text(0.99, 0.02, nota, transform=ax.transAxes, ha="right", fontsize=8, color="#405060")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def generar_figuras(datos: dict[str, object]) -> dict[str, Path]:
    paths = {
        "ranking_sil": FIGS / "ranking_silhouette_tsne_vertical.png",
        "ranking_db": FIGS / "ranking_davies_bouldin_vertical.png",
        "barrido": FIGS / "barrido_historico_softmin_vertical.png",
        "mejora_relativa": FIGS / "mejora_relativa_softmin_vs_base.png",
        "esquema_softmin": FIGS / "esquema_softmin_errores_f8.png",
        "tsne_actual": FIGS / "tsne_actual_vertical.png",
        "tsne_variantes": FIGS / "tsne_variantes_vertical.png",
        "tsne_f8": FIGS / "tsne_resultado8_perfil_softmin_errores_f8.png",
        "tsne_f8_inits": FIGS / "tsne_resultado8_perfil_softmin_errores_f8_inicializaciones.png",
        "comparacion_ajustada_f8": FIGS / "Comparacion_Figura11_lado_a_lado_f8.png",
        "v1_tsne_total": FIGS / "tsne_v1_metodos_articulo_y_f8.png",
        "v1_tabla_rasgos": FIGS / "tabla_v1_metricas_rasgos.png",
        "v1_tabla_tsne": FIGS / "tabla_v1_metricas_tsne.png",
        "resultado8_tsne_1000": RESULTADO8_OUT / "tsne_v1_resultado8_sin_descartar_1000.png",
        "mejora_sil_951_1000": RESULTADO8_OUT / "comparativa_v1_951_vs_1000_silhouette_tsne.png",
        "mejora_db_951_1000": RESULTADO8_OUT / "comparativa_v1_951_vs_1000_davies_bouldin_tsne.png",
        "v2_comparativa_sil": FIGS / "comparativa_v2_f8_inicializaciones_y_mejorado_silhouette.png",
        "v2_comparativa_db": FIGS / "comparativa_v2_f8_inicializaciones_y_mejorado_davies_bouldin.png",
        "conv_error": MEDIA / "05_convergencia_0_300" / "convergencia_error_0_300.png",
        "conv_error_clara": FIGS / "convergencia_error_0_300_optimizada.png",
        "conv_mejora": MEDIA / "05_convergencia_0_300" / "convergencia_mejora_0_120_vs_120_300.png",
    }
    figura_ranking_tsne(datos["comparacion_clave"], paths["ranking_sil"])
    figura_ranking_db(datos["comparacion_clave"], paths["ranking_db"])
    figura_barrido_historico(datos["ranking_barrido"], paths["barrido"])
    figura_mejora_relativa(datos["hist_barrido"], paths["mejora_relativa"])
    figura_esquema_softmin(paths["esquema_softmin"])
    figura_tsne_vertical(
        ["CNN", "DWT", "MFCC", "STFT", "Deep-ONMF normal", "Deep-ONMF mejorado"],
        paths["tsne_actual"],
        "t-SNE actual en formato vertical",
    )
    figura_tsne_v1_metodos_articulo_y_f8(datos, paths["v1_tsne_total"])
    figura_tsne_resultado8_sin_descartar_1000(datos, paths["resultado8_tsne_1000"])
    figura_mejora_951_vs_1000(datos["mejoras_951_1000"], paths["mejora_sil_951_1000"], "silhouette")
    figura_mejora_951_vs_1000(datos["mejoras_951_1000"], paths["mejora_db_951_1000"], "db")
    figura_v2_comparativa_decision(datos["comparacion_clave"], paths["v2_comparativa_sil"], "silhouette_tsne")
    figura_v2_comparativa_decision(datos["comparacion_clave"], paths["v2_comparativa_db"], "davies_bouldin_tsne")
    figura_convergencia_clara(datos["actual_conv"], paths["conv_error_clara"])
    figura_tabla_metricas(metricas_v1_rasgos(datos["v1_total"]), paths["v1_tabla_rasgos"], "Metricas de rasgos para V1")
    figura_tabla_metricas(metricas_v1_tsne(datos["v1_total"]), paths["v1_tabla_tsne"], "Metricas t-SNE para V1")
    figura_tsne_vertical(
        [
            "Deep-ONMF normal",
            "Deep-ONMF mejorado",
            "Deep-ONMF mejorado + NNDSVD",
            "Deep-ONMF mejorado + NNDSVDa",
            "Deep-ONMF mejorado + NNDSVDar",
        ],
        paths["tsne_variantes"],
        "Variantes Deep-ONMF actuales",
    )
    metricas_f8 = pd.concat([datos["f8_exacto"], datos["f8_init"]], ignore_index=True)
    metodo_f8 = "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8"
    figura_tsne_desde_csv(
        [(metodo_f8, HIST_F8_TSNE)],
        metricas_f8,
        paths["tsne_f8"],
        "t-SNE de resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8",
    )
    metodos_f8_init = list(datos["f8_init"]["metodo"])
    figura_tsne_desde_csv(
        [(m, coords_path_f8_init(m)) for m in metodos_f8_init],
        metricas_f8,
        paths["tsne_f8_inits"],
        "t-SNE de perfil_softmin_errores_f8 con NNDSVD, NNDSVDa y NNDSVDar",
    )
    extraida = extraer_figura_comparacion_ajustada(FIGS)
    if extraida is not None:
        paths["comparacion_ajustada_f8"] = extraida
    return paths


class PDFDoc:
    def __init__(self, path: Path, title: str, subtitle: str) -> None:
        self.path = path
        self.title_text = title
        self.doc = fitz.open()
        self.page_no = 0
        self.page = None
        self.y = 0.0
        self.new_page(header=False)
        self.cover(title, subtitle)

    @property
    def left(self) -> float:
        return 52

    @property
    def right(self) -> float:
        return 543

    @property
    def bottom(self) -> float:
        return 790

    def new_page(self, header: bool = True) -> None:
        self.page = self.doc.new_page(width=595.3, height=841.9)
        self.page_no += 1
        self.y = 52
        if header:
            self.header()

    def header(self) -> None:
        self.page.insert_text((self.left, 28), "Documento explicativo - comparativas deep-ONMF", fontsize=8.5, color=(0.22, 0.28, 0.36))
        self.page.draw_line((self.left, 36), (self.right, 36), color=(0.75, 0.79, 0.84), width=0.5)
        self.page.insert_text((self.right - 34, 818), str(self.page_no), fontsize=8, color=(0.35, 0.35, 0.35))

    def ensure(self, height: float) -> None:
        if self.y + height > self.bottom:
            self.new_page()

    def cover(self, title: str, subtitle: str) -> None:
        self.y = 82
        self.page.insert_textbox(
            fitz.Rect(self.left, self.y, self.right, self.y + 105),
            title,
            fontsize=20,
            fontname="helv",
            color=(0.04, 0.16, 0.29),
            align=fitz.TEXT_ALIGN_CENTER,
        )
        self.y += 125
        self.page.insert_textbox(
            fitz.Rect(self.left + 20, self.y, self.right - 20, self.y + 80),
            subtitle,
            fontsize=12,
            fontname="helv",
            color=(0.10, 0.15, 0.20),
            align=fitz.TEXT_ALIGN_CENTER,
            lineheight=1.25,
        )
        self.y += 115
        self.callout(
            "Documento redactado en formato academico y autonomo. Las explicaciones se presentan con tono docente, "
            "separando metodologia, resultados e interpretacion critica para facilitar la defensa del TFG."
        )
        self.y = 760
        self.page.insert_textbox(
            fitz.Rect(self.left, self.y, self.right, self.y + 30),
            "TFG - Analisis de fonocardiogramas mediante deep-ONMF",
            fontsize=9.5,
            color=(0.28, 0.32, 0.38),
            align=fitz.TEXT_ALIGN_CENTER,
        )
        self.new_page()

    def heading(self, text: str, level: int = 1) -> None:
        sizes = {1: 15.5, 2: 12.5, 3: 10.5}
        size = sizes.get(level, 11)
        self.ensure(34)
        color = (0.04, 0.16, 0.29) if level == 1 else (0.08, 0.20, 0.32)
        self.page.insert_text((self.left, self.y), text, fontsize=size, fontname="helv", color=color)
        self.y += size + 11

    def paragraph(self, text: str, font: float = 9.2, gap: float = 5) -> None:
        for para in text.split("\n"):
            if not para.strip():
                self.y += gap
                continue
            lines = textwrap.wrap(para, width=96)
            for line in lines:
                self.ensure(font + 6)
                self.page.insert_text((self.left, self.y), line, fontsize=font, fontname="helv", color=(0.05, 0.05, 0.05))
                self.y += font + 4.2
            self.y += gap

    def callout(self, text: str) -> None:
        lines = textwrap.wrap(text, width=86)
        h = 20 + len(lines) * 13
        self.ensure(h + 8)
        rect = fitz.Rect(self.left, self.y, self.right, self.y + h)
        self.page.draw_rect(rect, color=(0.67, 0.75, 0.84), fill=(0.91, 0.95, 0.98), width=0.7)
        y = self.y + 16
        for line in lines:
            self.page.insert_text((self.left + 12, y), line, fontsize=9.2, color=(0.04, 0.14, 0.24))
            y += 13
        self.y += h + 12

    def code_block(self, text: str, title: str | None = None) -> None:
        if title:
            self.heading(title, level=3)
        lines: list[str] = []
        for raw in text.strip("\n").splitlines():
            lines.extend(textwrap.wrap(raw, width=94, replace_whitespace=False) or [""])
        line_h = 10.5
        h = 16 + len(lines) * line_h
        self.ensure(h + 10)
        rect = fitz.Rect(self.left, self.y, self.right, self.y + h)
        self.page.draw_rect(rect, color=(0.62, 0.67, 0.72), fill=(0.96, 0.97, 0.98), width=0.45)
        y = self.y + 13
        for line in lines:
            self.page.insert_text((self.left + 8, y), line, fontsize=7.4, fontname="cour", color=(0.05, 0.06, 0.07))
            y += line_h
        self.y += h + 12

    def bullets(self, items: list[str]) -> None:
        for item in items:
            lines = textwrap.wrap(item, width=90)
            for j, line in enumerate(lines):
                self.ensure(15)
                prefix = "- " if j == 0 else "  "
                self.page.insert_text((self.left + 8, self.y), prefix + line, fontsize=9.1, color=(0.05, 0.05, 0.05))
                self.y += 13.4
            self.y += 2

    def table(self, df: pd.DataFrame, title: str, columns: list[str] | None = None, max_rows: int | None = None) -> None:
        data = df.copy()
        if columns is not None:
            data = data[columns]
        if max_rows is not None:
            data = data.head(max_rows)
        data = data.fillna("")
        self.heading(title, level=3)
        cols = list(data.columns)
        widths = self._col_widths(cols)
        font = 6.7 if len(cols) >= 6 else 7.3
        header_h = 30

        def draw_header() -> None:
            x = self.left
            for col, width in zip(cols, widths):
                rect = fitz.Rect(x, self.y, x + width, self.y + header_h)
                self.page.draw_rect(rect, color=(0.68, 0.72, 0.78), fill=(0.84, 0.90, 0.97), width=0.45)
                self._cell_text(str(col), rect, font, max_lines=2)
                x += width
            self.y += header_h

        self.ensure(header_h + 44)
        draw_header()
        for row in data.itertuples(index=False, name=None):
            line_counts = []
            for value, width in zip(row, widths):
                chars = max(8, int(width / (font * 0.53)))
                line_counts.append(max(1, min(4, len(textwrap.wrap(fmt(value), width=chars)))))
            row_h = 18 + max(line_counts) * (font + 3.4)
            if self.y + row_h > self.bottom:
                self.new_page()
                self.heading(title + " (continuacion)", level=3)
                draw_header()
            x = self.left
            for value, width in zip(row, widths):
                rect = fitz.Rect(x, self.y, x + width, self.y + row_h)
                self.page.draw_rect(rect, color=(0.73, 0.75, 0.79), fill=None, width=0.35)
                self._cell_text(fmt(value), rect, font, max_lines=4)
                x += width
            self.y += row_h
        self.y += 12

    def _col_widths(self, cols: list[str]) -> list[float]:
        total_w = self.right - self.left
        if len(cols) == 2:
            weights = [1, 2.2]
        elif len(cols) == 3:
            weights = [1.2, 1, 2]
        elif len(cols) == 4:
            weights = [1.25, 1, 1, 1.35]
        elif len(cols) == 5:
            weights = [1.45, 0.72, 0.72, 0.82, 1.15]
        else:
            weights = [1.55] + [0.85] * (len(cols) - 1)
        s = sum(weights)
        return [total_w * w / s for w in weights]

    def _cell_text(self, text: str, rect: fitz.Rect, font: float, max_lines: int = 3) -> None:
        chars = max(8, int(rect.width / (font * 0.53)))
        lines: list[str] = []
        for parte in str(text).splitlines():
            lines.extend(textwrap.wrap(parte, width=chars, break_long_words=True, break_on_hyphens=False) or [""])
        lines = lines[:max_lines] or [""]
        y = rect.y0 + 10
        for line in lines:
            self.page.insert_text((rect.x0 + 3, y), line, fontsize=font, fontname="helv", color=(0, 0, 0))
            y += font + 3.4

    def image(self, path: Path, title: str, caption: str, height: float = 310) -> None:
        self.ensure(height + 80)
        self.heading(title, level=3)
        rect = fitz.Rect(self.left, self.y, self.right, self.y + height)
        self.page.insert_image(rect, filename=str(path), keep_proportion=True)
        self.y += height + 8
        self.paragraph(caption, font=8.4, gap=4)

    def save(self) -> None:
        self.doc.save(self.path, deflate=True, garbage=4, clean=True)
        self.doc.close()


def glosario_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sigla": "PCG", "significado": "Phonocardiogram", "lectura": "senal acustica del corazon"},
            {"sigla": "STFT", "significado": "Short-Time Fourier Transform", "lectura": "mapa tiempo-frecuencia"},
            {"sigla": "DWT", "significado": "Discrete Wavelet Transform", "lectura": "analisis multirresolucion"},
            {"sigla": "MFCC", "significado": "Mel-Frequency Cepstral Coefficients", "lectura": "resumen cepstral compacto"},
            {"sigla": "CNN", "significado": "Convolutional Neural Network", "lectura": "red convolucional sobre espectrogramas"},
            {"sigla": "ONMF", "significado": "Orthogonal Non-negative Matrix Factorization", "lectura": "factorizacion con bases no redundantes"},
            {"sigla": "SBV", "significado": "Spectral Basis Vector", "lectura": "patron espectral aprendido por W"},
        ]
    )


def metricas_compactas(df: pd.DataFrame, metodo_col: str = "metodo") -> pd.DataFrame:
    out = df.copy()
    out[metodo_col] = out[metodo_col].map(etiqueta_corta)
    cols = [metodo_col, "muestras", "rasgos_originales", "silhouette_tsne", "davies_bouldin_tsne"]
    return out[cols].rename(
        columns={
            metodo_col: "metodo",
            "rasgos_originales": "rasgos",
            "silhouette_tsne": "sil t-SNE",
            "davies_bouldin_tsne": "DB t-SNE",
        }
    )


def metricas_v2_tabla(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bloque",
        "metodo",
        "muestras",
        "rasgos_originales",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    out = df[cols].copy()
    out["metodo"] = out["metodo"].str.replace(
        "resultado8_deep_onmf_sin_descartar_menores_2s__",
        "resultado8_deep_onmf_sin_descartar_menores_2s__\n",
        regex=False,
    )
    return out.rename(
        columns={
            "rasgos_originales": "rasgos",
            "silhouette_features": "sil rasgos",
            "davies_bouldin_features": "DB rasgos",
            "silhouette_tsne": "sil t-SNE",
            "davies_bouldin_tsne": "DB t-SNE",
        }
    )


def metricas_v2_decision(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bloque",
        "protocolo",
        "metodo",
        "muestras",
        "rasgos_originales",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    out = df[cols].copy()
    out["protocolo"] = out["protocolo"].map(protocolo_corto)
    return out.rename(
        columns={
            "rasgos_originales": "rasgos",
            "silhouette_features": "sil rasgos",
            "davies_bouldin_features": "DB rasgos",
            "silhouette_tsne": "sil t-SNE",
            "davies_bouldin_tsne": "DB t-SNE",
        }
    )


def convergencia_v2_tabla(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lecturas = {
        1: "la base espectral gruesa queda practicamente estabilizada antes de 120",
        2: "la capa intermedia sigue mejorando, pero la mayor ganancia ya esta antes de 120",
        3: "la representacion final de 7 SBV apenas cambia despues de 120",
    }
    out["lectura"] = out["capa"].map(lecturas)
    cols = [
        "capa",
        "rango",
        "error_iter_0",
        "error_iter_120",
        "error_iter_300",
        "mejora_pct_0_a_120",
        "porcentaje_mejora_total_antes_120",
        "lectura",
    ]
    out = out[cols].rename(
        columns={
            "error_iter_0": "error 0",
            "error_iter_120": "error 120",
            "error_iter_300": "error 300",
            "mejora_pct_0_a_120": "mejora 0-120 (%)",
            "porcentaje_mejora_total_antes_120": "mejora total antes 120 (%)",
        }
    )
    return out


def etiqueta_v2_pdf(nombre: str) -> str:
    texto = str(nombre)
    prefijo = "resultado8_deep_onmf_sin_descartar_menores_2s__"
    if texto.startswith(prefijo):
        return texto.replace(prefijo, "")
    return texto


def figura_v2_comparativa_decision(comparacion: pd.DataFrame, path: Path, metrica: str) -> None:
    data = comparacion.copy()
    asc = metrica == "davies_bouldin_tsne"
    data = data.sort_values(metrica, ascending=asc).reset_index(drop=True)
    etiquetas = [textwrap.fill(str(m), width=48) for m in data["metodo"]]
    colores = data["bloque"].map(
        {
            "F8 exacto con 1000 audios": "#2f6f9f",
            "F8 + NNDSVD/NNDSVDa/NNDSVDar con 1000 audios": "#43815a",
            "Deep-ONMF mejorado + NNDSVD/NNDSVDa/NNDSVDar con 951 audios": "#7a9e3f",
        }
    ).fillna("#6b7280")
    alto = max(6.2, 0.58 * len(data) + 1.3)
    fig, ax = plt.subplots(figsize=(8.1, alto))
    y = np.arange(len(data))
    valores = data[metrica].astype(float).to_numpy()
    ax.barh(y, valores, color=colores)
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas, fontsize=7.4)
    ax.invert_yaxis()
    if metrica == "silhouette_tsne":
        ax.set_xlabel("Silhouette t-SNE (mayor es mejor)")
        ax.set_title("Comparacion V2 por separacion visual")
    else:
        ax.set_xlabel("Davies-Bouldin t-SNE (menor es mejor)")
        ax.set_title("Comparacion V2 por compacidad visual")
    ax.grid(True, axis="x", alpha=0.25)
    for i, valor in enumerate(valores):
        ax.text(valor + max(valores) * 0.015, i, f"{valor:.4f}", va="center", fontsize=7.4)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figura_convergencia_clara(conv: pd.DataFrame, path: Path) -> None:
    data = conv.copy().sort_values("capa")
    xs = np.array([0, 120, 300], dtype=float)
    colores = ["#2f6f9f", "#d66a2d", "#4b8f3a"]
    fig, ax = plt.subplots(figsize=(8.1, 5.1))
    for color, row in zip(colores, data.itertuples(index=False)):
        ys = np.array([row.error_iter_0, row.error_iter_120, row.error_iter_300], dtype=float)
        etiqueta = f"Capa {row.capa} - rango {row.rango}"
        ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=6, color=color, label=etiqueta)
        ax.text(302, ys[-1], f"{ys[-1]:.4f}", va="center", fontsize=8, color=color)
        ax.annotate(
            f"{row.porcentaje_mejora_total_antes_120:.2f}% antes de 120",
            xy=(120, ys[1]),
            xytext=(128, ys[1] * 1.45),
            fontsize=7.8,
            color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
        )
    ax.axvspan(0, 120, color="#dbeafe", alpha=0.28, label="aprendizaje principal")
    ax.axvspan(120, 300, color="#f3f4f6", alpha=0.55, label="ajuste residual")
    ax.axvline(120, color="#c0392b", linestyle="--", linewidth=1.2)
    ax.text(120, ax.get_ylim()[1] * 0.82, "120 iteraciones", ha="center", va="top", fontsize=8.2, color="#8b1e1e")
    ax.set_yscale("log")
    ax.set_xticks([0, 120, 300])
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Error relativo medio (escala log)")
    ax.set_title("Convergencia Deep-ONMF en puntos clave: 0, 120 y 300")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def generar_v1(datos: dict[str, object], figs: dict[str, Path]) -> None:
    f8_nombre = "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8"
    actual = datos["actual_comp"].copy()
    resultado8 = datos["v1_resultado8"].copy()
    total = datos["v1_total"].copy()

    def metricas_metodo(nombre: str) -> pd.DataFrame:
        out = total.loc[total["metodo"] == nombre, [
            "protocolo",
            "muestras",
            "rasgos_originales",
            "silhouette_features",
            "davies_bouldin_features",
            "silhouette_tsne",
            "davies_bouldin_tsne",
        ]].copy()
        out["protocolo"] = out["protocolo"].map(protocolo_corto)
        return out.rename(
            columns={
                "rasgos_originales": "rasgos",
                "silhouette_features": "sil rasgos",
                "davies_bouldin_features": "DB rasgos",
                "silhouette_tsne": "sil t-SNE",
                "davies_bouldin_tsne": "DB t-SNE",
            }
        )

    pdf = PDFDoc(
        RESULTADOS / "Documento explicativo v1.pdf",
        "DOCUMENTO EXPLICATIVO V1\nComparativa completa de CNN, DWT, MFCC, STFT, Deep-ONMF y F8",
        "Implementacion, resultados e interpretacion docente de las representaciones para PCG",
    )
    pdf.heading("Resumen ejecutivo")
    pdf.callout(
        "Este V1 compara CNN, DWT, MFCC, STFT, Deep-ONMF normal, Deep-ONMF mejorado y "
        "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8. El ranking de silhouette t-SNE no se "
        "incluye aqui porque pertenece al documento V2, centrado en inicializaciones y decision final."
    )
    pdf.paragraph(
        "El objetivo de este documento no es mostrar una tabla suelta, sino explicar como profesor que se ha hecho, por "
        "que se ha hecho y como se deben leer los resultados. La comparativa tiene dos niveles. El primero reproduce el "
        "articulo objetivo con la politica estricta de tramas de 2 segundos: los audios menores de 2 segundos se descartan "
        "y quedan 951 audios. El segundo recupera la comparativa ajustada resultado8_deep_onmf_sin_descartar_menores_2s, "
        "donde se mantienen los 1000 audios rellenando con ceros los registros cortos."
    )
    pdf.paragraph(
        "La diferencia entre 951 y 1000 audios no es un error. Es una decision experimental distinta. En el protocolo fiel "
        "al articulo, una trama incompleta no se usa porque no cumple la duracion minima. En resultado8, esos audios cortos "
        "se rellenan hasta 2 segundos para no perder muestras. Por eso el documento no mezcla los numeros como si fueran "
        "exactamente el mismo experimento: primero los separa, despues explica que significa cada diferencia."
    )
    pdf.paragraph(
        "Las metricas usadas son silhouette y Davies-Bouldin, tanto en el espacio de rasgos como en la proyeccion t-SNE. "
        "Silhouette alto indica que cada audio queda mas cerca de audios de su misma clase que de audios de otras clases. "
        "Davies-Bouldin bajo indica grupos compactos y centros alejados. Si un metodo sube silhouette pero empeora "
        "Davies-Bouldin, no es una contradiccion: puede separar vecindades locales y, al mismo tiempo, dejar alguna clase "
        "alargada, fragmentada o demasiado cercana a otra."
    )
    pdf.heading("Glosario de abreviaturas")
    pdf.table(glosario_df(), "Abreviaturas usadas en la memoria")

    pdf.heading("Protocolos comparados")
    pdf.paragraph(
        "La primera tabla resume la decision que mas afecta a la lectura de resultados: como se tratan los audios menores "
        "de 2 segundos. En el protocolo fiel se eliminan 49 audios porque no forman una trama completa. En resultado8 se "
        "rellenan con ceros y se conserva toda la base de 1000 audios. Esta diferencia debe explicarse en la defensa porque "
        "altera la distribucion de muestras y puede cambiar la geometria de t-SNE."
    )
    pdf.table(datos["protocolos"], "Tabla V1. Protocolos y muestras")
    pdf.table(
        datos["auditoria"],
        "Auditoria de audios usados en el protocolo fiel",
        columns=["clase", "audios_totales", "audios_usados", "audios_descartados_por_duracion", "columnas_matriz_x"],
    )
    pdf.paragraph(
        "La auditoria muestra que N y AS conservan 200 audios, mientras MR pierde 16, MS pierde 14 y MVP pierde 19. Por eso "
        "el protocolo fiel al articulo queda en 951 audios. Esta explicacion es importante: no se han perdido muestras por "
        "fallo de codigo, sino por respetar la duracion minima definida para construir tramas de 2 segundos."
    )
    pdf.paragraph(
        "A partir de aqui se distinguen dos lecturas. La lectura metodologica pregunta si la implementacion reproduce el "
        "articulo con sus parametros principales. La lectura empirica pregunta que representacion separa mejor las clases "
        "en esta base local. La version F8 pertenece a esta segunda lectura: no sustituye la reproduccion fiel, sino que "
        "explica la mejora que se encontro durante el barrido de rasgos."
    )

    pdf.heading("Como se leen las metricas")
    pdf.paragraph(
        "El espacio de rasgos es el espacio real que entrega cada metodo. CNN entrega activaciones, MFCC entrega coeficientes "
        "cepstrales, DWT entrega estadisticos por escala, STFT entrega energia tiempo-frecuencia resumida y Deep-ONMF entrega "
        "bases o afinidades. Por eso silhouette_features y davies_bouldin_features miden la separacion antes de dibujar t-SNE."
    )
    pdf.paragraph(
        "t-SNE es una representacion bidimensional para ver vecindades. Se usa porque una figura ayuda a explicar donde se "
        "mezclan las clases, pero no debe venderse como una verdad absoluta. Dos puntos cercanos en t-SNE suelen compartir "
        "vecindad local, pero la distancia global entre islas puede deformarse. Por eso V1 siempre acompana cada figura con "
        "silhouette y Davies-Bouldin."
    )
    pdf.paragraph(
        "La defensa correcta no consiste en elegir una metrica y ocultar las demas. Si silhouette sube y Davies-Bouldin baja, "
        "hay una mejora muy clara. Si solo mejora una, se debe explicar que propiedad ha mejorado: vecindad local, compacidad "
        "interna o separacion entre centros. Esta forma de razonar evita conclusiones simples y permite justificar por que "
        "perfil_softmin_errores_f8 es fuerte: combina silhouette t-SNE 0.2602 con Davies-Bouldin t-SNE 1.3756."
    )

    pdf.heading("Tablas principales de resultados")
    pdf.paragraph(
        "Las tablas se separan para que no se corten y para que se vea que hay dos protocolos. En la primera lectura se "
        "observa la separacion en el espacio original de rasgos; en la segunda, la separacion tras proyectar con t-SNE. "
        "Los valores se redondean a cuatro decimales porque esa precision es suficiente para defender diferencias reales."
    )
    pdf.heading("Protocolo fiel al articulo: 951 audios", level=2)
    pdf.table(metricas_v1_rasgos(actual), "Tabla V1. Metricas de rasgos en el protocolo fiel al articulo")
    pdf.table(metricas_v1_tsne(actual), "Tabla V1. Metricas t-SNE en el protocolo fiel al articulo")
    pdf.image(
        figs["tsne_actual"],
        "Figura V1. t-SNE de CNN, DWT, MFCC, STFT, Deep-ONMF normal y Deep-ONMF mejorado",
        "Esta figura procede de los CSV generados en esta ejecucion. Muestra los seis metodos del protocolo fiel al articulo "
        "con 951 audios. La lectura debe hacerse junto a las tablas: una nube visualmente ordenada puede empeorar Davies-"
        "Bouldin si una clase queda estirada o partida.",
        height=430,
    )
    pdf.heading("Protocolo resultado8_deep_onmf_sin_descartar_menores_2s: 1000 audios", level=2)
    pdf.table(metricas_v1_rasgos(resultado8), "Tabla V1. Metricas de rasgos en resultado8_deep_onmf_sin_descartar_menores_2s")
    pdf.table(metricas_v1_tsne(resultado8), "Tabla V1. Metricas t-SNE en resultado8_deep_onmf_sin_descartar_menores_2s")
    pdf.image(
        figs["resultado8_tsne_1000"],
        "Figura V1. t-SNE de resultado8_deep_onmf_sin_descartar_menores_2s con 1000 audios",
        "Esta figura procede de los CSV extraidos de comparacion_ajustada_completa.zip. Incluye CNN, DWT, MFCC, STFT y "
        "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8. Asi se ve el protocolo de 1000 audios "
        "debajo de sus propias tablas, sin mezclarlo visualmente con el protocolo fiel de 951 audios.",
        height=430,
    )

    pdf.heading("Que cambia al no descartar audios menores de 2 s", level=2)
    pdf.paragraph(
        "El protocolo de 1000 audios no mejora o empeora todos los metodos por igual. La diferencia no es solo numerica: al "
        "rellenar los audios cortos con ceros se conserva una parte de la distribucion que el protocolo fiel elimina. Por eso "
        "CNN obtiene una mejora clara en silhouette t-SNE, mientras que MFCC, DWT y STFT reaccionan de forma distinta. La "
        "comparacion correcta es mirar metodo por metodo y no afirmar que 1000 audios siempre sea mejor."
    )
    pdf.table(mejoras_951_1000_pdf(datos["mejoras_951_1000"]), "Tabla V1. Cambio de metricas t-SNE al pasar de 951 a 1000 audios")
    pdf.image(
        figs["mejora_sil_951_1000"],
        "Figura V1. Mejora de silhouette t-SNE al conservar 1000 audios",
        "CNN pasa de 0.1106 a 0.2259, una mejora de +0.1153 y aproximadamente +104.31%. En DWT, MFCC y STFT la silhouette "
        "t-SNE baja, lo que demuestra que la politica de datos no beneficia de la misma manera a todos los descriptores.",
        height=330,
    )
    pdf.image(
        figs["mejora_db_951_1000"],
        "Figura V1. Cambio de Davies-Bouldin t-SNE al conservar 1000 audios",
        "Davies-Bouldin se lee al reves: menor es mejor. CNN baja de 3.6186 a 2.0143, DWT baja de 16.3527 a 6.5641 y MFCC "
        "baja ligeramente de 6.0001 a 5.5277. STFT empeora porque sube de 3.5703 a 7.8786.",
        height=330,
    )
    pdf.paragraph(
        "Tambien hay que separar las versiones Deep-ONMF. En 951 audios aparecen Deep-ONMF normal y Deep-ONMF mejorado. En "
        "1000 audios aparece resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8. No son filas "
        "identicas una a una: cambian la politica de datos y la forma de leer la salida del modelo. Por eso el documento las "
        "muestra juntas para interpretarlas, pero no las presenta como si fueran el mismo experimento."
    )
    pdf.table(lectura_docente_v1(), "Tabla V1. Lectura docente por metodo")

    pdf.heading("Figura de flujo F8", level=2)
    pdf.image(
        figs["esquema_softmin"],
        "Figura V1. Flujo de perfil_softmin_errores_f8",
        "El esquema resume la mejora esencial: cada audio se compara contra las bases de las cinco clases, se calculan errores "
        "de reconstruccion y esos errores se convierten en afinidades mediante softmin de fuerza 8.",
        height=420,
    )

    pdf.new_page()
    pdf.heading("1. CNN")
    pdf.callout("Implementacion usada: Conv2d(1,32,3x3)+ReLU; Conv2d(32,64,3x3)+ReLU; MaxPool2d(1,6); rasgo final de 64 activaciones.")
    pdf.table(metricas_metodo("CNN"), "Metricas CNN")
    pdf.paragraph(
        "La CNN trabaja sobre una imagen tiempo-frecuencia. En lugar de resumir manualmente la senal, aprende filtros que "
        "recorren el espectrograma y detectan patrones locales. En el articulo se define una red sencilla de tres bloques: "
        "una primera convolucion con 32 filtros de 3x3, una segunda convolucion con 64 filtros de 3x3 y una capa de max-"
        "pooling de tamano 1x6. Esa arquitectura se mantiene para que la comparacion no dependa de una red externa inventada."
    )
    pdf.paragraph(
        "La entrada de la CNN es el espectrograma normalizado. Conv1 aprende bordes y cambios simples de energia; Conv2 "
        "combina esos patrones para formar texturas mas complejas; el pooling 1x6 reduce la dimension temporal o de columnas "
        "sin destruir por completo la informacion frecuencial. Despues se obtiene un vector de 64 activaciones, que es el "
        "rasgo usado para calcular silhouette, Davies-Bouldin y t-SNE."
    )
    pdf.paragraph(
        "En el protocolo fiel, CNN obtiene silhouette t-SNE 0.1106 y Davies-Bouldin t-SNE 3.6186. En resultado8, con 1000 "
        "audios, obtiene silhouette t-SNE 0.2259 y Davies-Bouldin t-SNE 2.0143. Esta diferencia indica que la CNN puede "
        "beneficiarse de la politica de mantener todos los audios y de una distribucion local distinta. Aun asi, no supera "
        "el equilibrio global de F8, que logra 0.2602 y 1.3756."
    )
    pdf.paragraph(
        "La ventaja de CNN es que aprende automaticamente patrones no lineales. La limitacion es que necesita supervision, "
        "normalizacion y entrenamiento; ademas, sus activaciones no explican tan directamente que parte del sonido se ha "
        "reconstruido bien o mal. Por eso CNN es una comparativa potente, pero no ofrece la misma interpretabilidad de bases "
        "por clase que Deep-ONMF."
    )
    pdf.code_block(extraer_lineas_con_patron(CODIGO_GENERACION_FINAL, ["class CNNArticulo", "Conv2d", "MaxPool2d"], contexto=3), "Codigo real: arquitectura CNN")

    pdf.heading("2. DWT")
    pdf.callout("Implementacion usada: wavelet coif5, descomposicion por niveles y estadisticos energia-log, media absoluta y desviacion.")
    pdf.table(metricas_metodo("DWT"), "Metricas DWT")
    pdf.paragraph(
        "DWT analiza la senal por escalas. Mientras STFT usa ventanas y frecuencias, DWT usa una familia de ondas para "
        "separar componentes rapidos y lentos. Esto es util en PCG porque los sonidos cardiacos contienen eventos breves, "
        "cierres de valvulas y posibles soplos. La wavelet coif5 se usa porque el articulo la toma como referencia adecuada "
        "para la senal cardiaca."
    )
    pdf.paragraph(
        "El codigo divide cada audio en tramas, aplica pywt.wavedec con coif5 y despues resume cada bloque de coeficientes. "
        "Los rasgos no son la onda completa, sino estadisticos: energia logaritmica, media absoluta y desviacion. Este "
        "resumen reduce mucho la dimension y hace la comparacion manejable."
    )
    pdf.paragraph(
        "En el protocolo fiel, DWT obtiene silhouette t-SNE 0.1268, pero Davies-Bouldin t-SNE 16.3527. Ese DB tan alto indica "
        "que algun grupo queda muy disperso o demasiado cerca de otros. En resultado8, DWT pasa a silhouette t-SNE 0.1109 y "
        "DB t-SNE 6.5641. Mejora DB respecto al protocolo fiel, pero sigue por debajo de las variantes Deep-ONMF."
    )
    pdf.paragraph(
        "La lectura docente es que DWT detecta informacion util, pero la representacion final no esta optimizada para separar "
        "patologias concretas. Es buena para describir escalas, no necesariamente para producir un perfil de pertenencia por "
        "clase. Por eso funciona como comparativa clasica, pero no como mejor solucion final."
    )
    pdf.code_block(extraer_funcion_codigo(CODIGO_GENERACION_FINAL, "rasgos_dwt", max_lineas=18), "Codigo real: DWT con coif5")

    pdf.heading("3. MFCC")
    pdf.callout("Implementacion usada: 40 filtros Mel, logaritmo, DCT y seleccion de 13 coeficientes finales.")
    pdf.table(metricas_metodo("MFCC"), "Metricas MFCC")
    pdf.paragraph(
        "MFCC procede del analisis de voz, pero tambien se usa en sonidos cardiacos porque resume la envolvente espectral. "
        "La idea es pasar de un espectro lineal a un banco de filtros Mel, aplicar logaritmo para comprimir amplitudes y "
        "despues usar la DCT para obtener coeficientes decorrelacionados. En este trabajo se usan 40 filtros Mel y se guardan "
        "los 13 primeros coeficientes, siguiendo la indicacion del articulo."
    )
    pdf.paragraph(
        "La ventaja de MFCC es la compacidad. Cada audio queda representado por 13 valores medios que resumen la forma "
        "global del espectro. Eso ayuda a evitar sobreajuste y facilita t-SNE, pero tambien puede perder informacion temporal. "
        "Si dos patologias tienen una envolvente parecida aunque difieran en momentos concretos del ciclo cardiaco, MFCC puede "
        "acercarlas demasiado."
    )
    pdf.paragraph(
        "En el protocolo fiel, MFCC obtiene silhouette t-SNE 0.1728 y Davies-Bouldin t-SNE 6.0001. Es mejor que CNN y DWT en "
        "silhouette dentro de 951 audios, pero su DB sigue siendo alto. En resultado8 obtiene silhouette t-SNE 0.1446 y DB "
        "t-SNE 5.5277. La conclusion no es que MFCC sea malo, sino que su compresion no captura tan bien la pertenencia por "
        "clase como el perfil F8."
    )
    pdf.paragraph(
        "MFCC se puede defender como un baseline acustico compacto. Su papel en la memoria es demostrar que una tecnica muy "
        "conocida de audio no basta por si sola para alcanzar el mejor equilibrio. La mejora de Deep-ONMF y F8 aparece porque "
        "no solo miran la envolvente, sino la capacidad de reconstruccion de cada clase."
    )
    pdf.code_block(extraer_funcion_codigo(CODIGO_GENERACION_FINAL, "rasgos_mfcc", max_lineas=16), "Codigo real: MFCC")

    pdf.heading("4. STFT")
    pdf.callout("Implementacion usada: tramas de 2 s, solape de 1 s, ventana Hamming 150, salto 75 y FFT 250.")
    pdf.table(metricas_metodo("STFT"), "Metricas STFT")
    pdf.paragraph(
        "STFT transforma la senal temporal en una matriz tiempo-frecuencia. Esto es fundamental porque un PCG no solo se "
        "define por su amplitud, sino por como cambia su energia a lo largo del ciclo cardiaco. La ventana Hamming de 150 "
        "muestras suaviza bordes, el salto de 75 introduce solape y la FFT de 250 puntos fija la resolucion frecuencial."
    )
    pdf.paragraph(
        "Como metodo comparativo, STFT se resume con medias y desviaciones por bin de frecuencia. Esa representacion conserva "
        "mucha informacion, pero no aprende una base por clase. En otras palabras, STFT describe el audio, mientras Deep-ONMF "
        "aprende diccionarios que intentan reconstruirlo. Por eso STFT es la entrada natural de Deep-ONMF, pero no tiene por "
        "que ser el mejor rasgo final."
    )
    pdf.paragraph(
        "En el protocolo fiel, STFT obtiene silhouette t-SNE 0.1478 y Davies-Bouldin t-SNE 3.5703. En resultado8 obtiene "
        "silhouette t-SNE 0.1319 y DB t-SNE 7.8786. La lectura es clara: STFT aporta la informacion espectral necesaria, "
        "pero usada directamente mezcla clases porque no transforma esa informacion en una decision de pertenencia."
    )
    pdf.paragraph(
        "Por este motivo STFT debe aparecer en la documentacion como comparativa adicional local y como etapa de entrada. "
        "No compite solo contra Deep-ONMF; tambien explica de donde sale la matriz no negativa X que luego se factoriza."
    )
    pdf.code_block(extraer_funcion_codigo(CODIGO_GENERACION_FINAL, "rasgos_stft", max_lineas=14), "Codigo real: resumen STFT")

    pdf.heading("5. Deep-ONMF normal")
    pdf.callout("Implementacion usada: STFT por clase, tres capas ONMF con rangos 9-8-7 y W_final = W1 W2 W3, generando 7 SBV.")
    pdf.table(metricas_metodo("Deep-ONMF normal"), "Metricas Deep-ONMF normal")
    pdf.table(
        resumen_capas(datos["capas"]),
        "Resumen limpio de capas Deep-ONMF",
        columns=["capa", "rango", "error_medio", "ortogonalidad_media", "lectura"],
    )
    pdf.paragraph(
        "Deep-ONMF normal empieza con una matriz X por clase. Las filas representan bins de frecuencia y las columnas tramas "
        "extraidas de los audios de esa clase. Como X es no negativa, se puede aproximar mediante W y H. W contiene bases "
        "espectrales; H indica cuanto se activa cada base en cada trama."
    )
    pdf.paragraph(
        "La parte profunda consiste en aplicar ONMF en tres capas. La primera aprende 9 patrones, la segunda comprime a 8 y "
        "la tercera produce 7 bases espectrales finales. El producto W1 W2 W3 devuelve esas bases al espacio espectral "
        "original. Por eso se habla de 7 SBV finales por clase."
    )
    pdf.paragraph(
        "En el protocolo fiel, Deep-ONMF normal obtiene silhouette t-SNE 0.1907 y Davies-Bouldin t-SNE 1.7910. Frente a CNN, "
        "DWT, MFCC y STFT, el DB es claramente mas bajo, lo que indica grupos mas compactos en la proyeccion. Sin embargo, "
        "la representacion normal todavia lee la estructura interna del modelo, no la comparacion explicita entre clases."
    )
    pdf.paragraph(
        "La limitacion principal es que una activacion alta no siempre significa pertenencia clara a una patologia. Puede "
        "indicar que un patron espectral comun aparece en varias clases. Por eso surge la mejora: usar las bases aprendidas "
        "para preguntar que clase reconstruye mejor cada audio."
    )
    pdf.code_block(extraer_lineas_con_patron(CODIGO_GENERACION_FINAL, ["w_final", "matrices_w[0]", "matrices_w.append"], contexto=3), "Codigo real: producto W1 W2 W3")

    pdf.new_page()
    pdf.heading("Aclaracion docente: W_final, H3 y los rasgos de cada audio", level=2)
    pdf.paragraph(
        "Es importante separar dos objetos que suelen confundirse. W_final y H3 no contienen la misma informacion. W_final "
        "contiene los patrones espectrales aprendidos: es decir, los SBV. H3 contiene las activaciones de esos patrones: cuanto "
        "se usa cada SBV para reconstruir cada columna o trama del espectrograma. Por eso W_final responde a que patrones "
        "existen, mientras que H3 responde a cuanto aparece cada patron en cada parte de los audios."
    )
    pdf.table(
        pd.DataFrame(
            [
                {
                    "matriz": "W_final",
                    "que representa": "7 patrones espectrales o SBV finales",
                    "uso principal": "interpretar que formas de energia en frecuencia aprende cada clase",
                },
                {
                    "matriz": "H3",
                    "que representa": "activacion de los 7 SBV en las tramas de los audios",
                    "uso principal": "obtener 7 caracteristicas comparables por audio",
                },
            ]
        ),
        "Lectura de W_final frente a H3",
    )
    pdf.paragraph(
        "Por tanto, no es completamente preciso decir que al aplicar Deep-ONMF a un fichero se devuelve una matriz de 7 SBV "
        "para ese fichero. La implementacion entrena Deep-ONMF por clase, obtiene una W_final con 7 columnas por clase y "
        "despues usa H3 para representar cada audio. Para un audio concreto se toman las columnas de H3 que corresponden a "
        "sus tramas y se promedia cada fila. Ese promedio produce el vector [SBV_1, ..., SBV_7] que se lleva a las metricas "
        "y al t-SNE."
    )
    pdf.paragraph(
        "La razon de usar H3 para Figura 11 es que cada punto de la figura representa un audio. W_final es muy buena para "
        "interpretar el diccionario espectral de una clase, pero no es por si sola una fila por audio. Si se entrenase una W "
        "independiente para cada fichero, las columnas podrian permutarse o cambiar de escala, y entonces W_audio_1[:, 1] no "
        "tendria por que significar lo mismo que W_audio_2[:, 1]. En cambio, las activaciones H3 son comparables porque se "
        "leen respecto a las bases aprendidas por la clase."
    )
    pdf.paragraph(
        "La frase correcta para defender esta parte es: aunque los 7 SBV estan en W_final, las caracteristicas utilizadas "
        "para representar cada audio se extraen de la matriz de activaciones final H3. Para cada audio se toman las columnas "
        "de H3 asociadas a sus tramas y se calcula la media de cada una de las 7 filas, obteniendo un vector de 7 rasgos. "
        "Estos rasgos indican cuanto participa cada SBV en la reconstruccion del audio."
    )
    pdf.code_block(extraer_funcion_codigo(Path(ROOT / "src" / "tfg_deep_onmf" / "estadistica.py"), "caracteristicas_por_audio", max_lineas=22), "Codigo real: rasgos por audio desde H3")

    pdf.heading("6. Deep-ONMF mejorado")
    pdf.callout("Implementacion usada: mismas bases Deep-ONMF, pero el rasgo final pasa a ser un perfil de afinidad por reconstruccion de 5 valores.")
    pdf.table(metricas_metodo("Deep-ONMF mejorado"), "Metricas Deep-ONMF mejorado")
    pdf.table(diferencias_deep_f8(), "Diferencias entre Deep-ONMF normal, Deep-ONMF mejorado y F8")
    pdf.paragraph(
        "Deep-ONMF mejorado no cambia la idea de aprender bases no negativas por clase. La mejora esta en la lectura del "
        "resultado. En vez de quedarse con activaciones o SBV medios, toma cada audio y lo proyecta contra las bases W de "
        "N, AS, MR, MS y MVP. Para cada proyeccion calcula un error de reconstruccion."
    )
    pdf.paragraph(
        "Esto cambia la pregunta matematica y tambien la pregunta clinica. El Deep-ONMF normal pregunta que patrones se "
        "activan. El Deep-ONMF mejorado pregunta que clase explica mejor el audio. Si un audio de MR se reconstruye mucho "
        "mejor con W_MR que con W_N, W_AS, W_MS o W_MVP, entonces el vector final refleja una evidencia de pertenencia."
    )
    pdf.paragraph(
        "En el protocolo fiel, Deep-ONMF mejorado obtiene silhouette t-SNE 0.2066 y Davies-Bouldin t-SNE 2.0001. La silhouette "
        "sube respecto al Deep-ONMF normal, lo que indica mejor vecindad local, aunque DB empeora ligeramente. Esto se puede "
        "explicar porque algunas clases quedan mas separadas localmente, pero alguna nube puede quedar mas estirada."
    )
    pdf.paragraph(
        "La mejora es importante porque prepara el camino hacia F8. El concepto fuerte no es solo entrenar mas capas, sino "
        "leer las bases como diccionarios de reconstruccion por clase. Esa interpretacion es mas defendible ante tribunal "
        "porque conecta el algoritmo con una decision: que base reconstruye mejor el PCG."
    )
    pdf.code_block(extraer_funcion_codigo(CODIGO_GENERACION_FINAL, "caracteristicas_mejoradas_por_audio", max_lineas=22), "Codigo real: afinidades de reconstruccion")

    pdf.heading("7. perfil_softmin_errores_f8 en resultado8")
    pdf.callout("Implementacion usada: 1000 audios, tramas 2 s, solape 1 s, rangos 9-8-7, 120 iteraciones, penalizacion 0.05 y softmin_errores(fuerza=8).")
    pdf.table(metricas_metodo(f8_nombre), "Metricas resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8")
    pdf.paragraph(
        "F8 es la variante mas importante de la comparativa porque conserva la idea del Deep-ONMF mejorado y la hace mas "
        "estable como rasgo final. Cada audio se reconstruye contra las bases de todas las clases. Despues no se guardan "
        "los errores crudos sin mas, sino que se convierten en afinidades relativas mediante softmin_errores con fuerza 8."
    )
    pdf.paragraph(
        "La palabra fuerza es clave. Si la fuerza es muy baja, las afinidades quedan demasiado suaves y las clases se parecen "
        "mas entre si. Si es excesivamente alta, el perfil puede volverse demasiado duro y perder matices. En el barrido "
        "historico se probaron varias fuerzas y perfil_softmin_errores_f8 ofrecio el mejor equilibrio global."
    )
    pdf.heading("Que es exactamente softmin_errores(fuerza=8)", level=2)
    pdf.paragraph(
        "softmin_errores(fuerza=8) es la transformacion que convierte cinco errores de reconstruccion en cinco afinidades "
        "comparables. Primero se calcula el error de reconstruir el mismo audio con cada base de clase: W_N, W_AS, W_MR, "
        "W_MS y W_MVP. Despues se divide cada error por el menor error de ese audio. Asi el mejor error queda con valor "
        "relativo 1 y los demas se interpretan como veces peor que el mejor."
    )
    pdf.paragraph(
        "La formula usada es: relativo_c = error_c / min(error); logit_c = -8 * (relativo_c - 1); afinidad_c = "
        "exp(logit_c) / sum_j exp(logit_j). El signo negativo es lo que convierte el softmax en softmin: un error pequeno "
        "recibe un peso grande y un error grande recibe un peso pequeno. El numero 8 es la fuerza que separa mas o menos "
        "las afinidades."
    )
    pdf.paragraph(
        "Por ejemplo, si un audio tiene errores [0.10, 0.12, 0.30, 0.40, 0.50], la mejor reconstruccion es 0.10. La clase "
        "con error 0.12 no esta tan lejos, pero las de 0.30, 0.40 y 0.50 son mucho peores. Con fuerza 8, el softmin concentra "
        "casi toda la afinidad en las clases que reconstruyen mejor. Por eso el vector final no es una lista de errores brutos, "
        "sino un perfil de pertenencia por clase."
    )
    pdf.table(ejemplo_softmin_f8(), "Ejemplo numerico de softmin_errores(fuerza=8)")
    pdf.paragraph(
        "Esta tabla es solo un ejemplo pedagogico, no una fila real del experimento. Sirve para entender la idea: si una base "
        "reconstruye claramente mejor, su afinidad sube; si dos bases reconstruyen parecido, las dos conservan peso. Ese "
        "comportamiento es justo lo que interesa para PCG, porque algunas patologias pueden compartir componentes espectrales "
        "y no conviene convertir todo en una decision dura demasiado pronto."
    )
    pdf.paragraph(
        "Los numeros justifican por que esta variante se defiende como mejor resultado: silhouette_features 0.2809, Davies-"
        "Bouldin_features 1.3783, silhouette t-SNE 0.2602 y Davies-Bouldin t-SNE 1.3756. En terminos practicos, eso significa "
        "que el espacio de 5 rasgos ya esta bien separado antes de t-SNE y que la proyeccion tambien conserva una separacion "
        "visual fuerte."
    )
    pdf.paragraph(
        "La comparacion con Deep-ONMF mejorado debe explicarse con cuidado porque no es exactamente el mismo protocolo: F8 "
        "usa 1000 audios y rellena los cortos; el protocolo fiel usa 951 y descarta los menores de 2 segundos. Aun asi, F8 "
        "es defendible como mejor resultado empirico porque fue generado desde codigo propio, con CSV trazables y metricas "
        "coherentes tanto en rasgos como en t-SNE."
    )
    pdf.paragraph(
        "V2 queda reservado para estudiar si NNDSVD, NNDSVDa o NNDSVDar mejoran esta variante o el Deep-ONMF mejorado. V1 "
        "deja ya claro el punto de partida: el resultado fuerte que se debe intentar superar es "
        "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8."
    )
    pdf.code_block(extraer_funcion_codigo(HIST_CODIGO_BARRIDO, "softmin_errores", max_lineas=10), "Codigo real: softmin_errores")
    pdf.code_block(extraer_funcion_codigo(HIST_CODIGO_BARRIDO, "variantes_rasgos", max_lineas=18), "Codigo real: creacion de perfil_softmin_errores_f8")

    pdf.heading("Que se puede afirmar y que no se puede afirmar", level=2)
    pdf.paragraph(
        "F8 no debe presentarse como una garantia matematica de separabilidad universal. No optimiza directamente el t-SNE ni "
        "entrena el modelo para maximizar silhouette. Lo que hace es construir una representacion alternativa: errores de "
        "reconstruccion contra las cinco bases de clase, normalizacion relativa y softmin con fuerza 8. Despues esa "
        "representacion se evalua empiricamente con silhouette y Davies-Bouldin."
    )
    pdf.paragraph(
        "Esto es una diferencia importante para la defensa. No se dice que F8 vaya a separar mejor cualquier base de datos. "
        "Lo que se puede afirmar es que, en este conjunto experimental y frente a las variantes probadas en el barrido, F8 "
        "produce la representacion mas separable segun las metricas calculadas. La afirmacion es fuerte porque mejora tanto "
        "en el espacio de rasgos como en el espacio t-SNE, pero sigue siendo una conclusion empirica, no una garantia teorica."
    )

    pdf.heading("Uso con un clasificador clasico como SVM", level=2)
    pdf.paragraph(
        "Una forma mas solida de defender que las caracteristicas son utiles seria entrenar un clasificador clasico, por "
        "ejemplo un SVM, usando estas caracteristicas como entrada. Asi no se dependeria solo de la visualizacion t-SNE. "
        "Para Deep-ONMF normal, la entrada del SVM seria la matriz X formada por las columnas SBV_1 a SBV_7. Para Deep-ONMF "
        "mejorado, la entrada seria la matriz X formada por afinidad_N, afinidad_AS, afinidad_MR, afinidad_MS y afinidad_MVP."
    )
    pdf.table(
        pd.DataFrame(
            [
                {
                    "variante": "Deep-ONMF normal",
                    "entrada SVM": "SBV_1 ... SBV_7",
                    "dimension": 7,
                    "interpretacion": "activacion media de cada SBV en el audio",
                },
                {
                    "variante": "Deep-ONMF mejorado",
                    "entrada SVM": "afinidad_N ... afinidad_MVP",
                    "dimension": 5,
                    "interpretacion": "afinidad por reconstruccion frente a cada clase",
                },
                {
                    "variante": "perfil_softmin_errores_f8",
                    "entrada SVM": "F_001 ... F_005",
                    "dimension": 5,
                    "interpretacion": "softmin relativo de errores de reconstruccion",
                },
            ]
        ),
        "Entradas posibles para un SVM",
    )
    pdf.paragraph(
        "Para que esa validacion sea correcta hay que evitar fuga de datos. Las bases W_N, W_AS, W_MR, W_MS y W_MVP deben "
        "aprenderse solo con el conjunto de entrenamiento. Despues se proyectan los audios de entrenamiento y test contra esas "
        "mismas bases. El SVM se entrena con X_train y se evalua con X_test. Si se entrenasen las bases con todos los audios "
        "antes de dividir, el test ya habria influido en la representacion y la validacion quedaria contaminada."
    )
    pdf.code_block(
        "X = df[[\"afinidad_N\", \"afinidad_AS\", \"afinidad_MR\", \"afinidad_MS\", \"afinidad_MVP\"]].to_numpy()\n"
        "y = df[\"clase\"].to_numpy()\n"
        "# Despues: StandardScaler + SVC, entrenando solo con el conjunto de entrenamiento.",
        "Entrada conceptual de SVM para Deep-ONMF mejorado",
    )

    pdf.heading("Uso con una senal nueva o con otra base de datos", level=2)
    pdf.paragraph(
        "El Deep-ONMF mejorado debe entenderse como un sistema basado en diccionarios de clase previamente aprendidos. En "
        "entrenamiento se obtienen las matrices W finales de cada clase. En inferencia, una senal nueva no se usa para "
        "reentrenar Deep-ONMF desde cero; se proyecta sobre W_N, W_AS, W_MR, W_MS y W_MVP. Los errores resultantes se "
        "transforman en afinidades y esas afinidades se pueden usar para decidir la clase o como entrada de un SVM."
    )
    pdf.paragraph(
        "Por tanto, si solo se dispone de una senal aislada y no hay W entrenadas previamente, el metodo mejorado no puede "
        "aplicarse de forma completa. Entrenar una W solamente con esa senal no daria una referencia de clase y no resolveria "
        "el problema. En cambio, si ya existen W de clase entrenadas y guardadas, una unica senal nueva si puede evaluarse: "
        "se calcula su error contra cada diccionario y se obtiene su perfil de afinidad."
    )
    pdf.paragraph(
        "Si se trabaja con una base de datos nueva etiquetada, hay dos opciones. La primera es reentrenar W_N, W_AS, W_MR, "
        "W_MS y W_MVP con esa base, que es lo mas correcto si cambian las condiciones de grabacion o distribucion. La segunda "
        "es reutilizar las W antiguas si la nueva base tiene las mismas clases y condiciones similares. En cualquier caso, "
        "la representacion mejorada siempre necesita diccionarios W de clase como referencia."
    )

    pdf.heading("Comparativa final de V1")
    pdf.paragraph(
        "Una vez vistos los metodos por separado, la conclusion no debe formularse como una frase aislada. CNN aprende "
        "texturas; DWT analiza escalas; MFCC resume envolventes; STFT describe energia tiempo-frecuencia; Deep-ONMF normal "
        "aprende bases; Deep-ONMF mejorado transforma esas bases en errores por clase; y F8 convierte esos errores en "
        "afinidades relativas con fuerza 8."
    )
    pdf.paragraph(
        "Si se mira solo el protocolo fiel de 951 audios, Deep-ONMF mejorado mejora la silhouette t-SNE frente a Deep-ONMF "
        "normal: 0.2066 frente a 0.1907. Si se mira el resultado8 de 1000 audios, perfil_softmin_errores_f8 alcanza 0.2602 "
        "de silhouette t-SNE y 1.3756 de Davies-Bouldin t-SNE. Por tanto, el mejor resultado empirico global encontrado es "
        "F8, mientras que el protocolo fiel demuestra que la implementacion del articulo tambien se ha reproducido."
    )
    pdf.paragraph(
        "La razon tecnica de la mejora es que F8 no se limita a describir el audio: lo compara contra las cinco bases de "
        "clase. Un vector final de 5 rasgos puede parecer pequeno, pero cada componente resume una pregunta potente: cuanto "
        "se parece este audio a la clase N, AS, MR, MS o MVP en terminos de reconstruccion. Esa es una representacion mas "
        "cercana al problema de clasificacion que un resumen generico de energia o cepstrum."
    )
    pdf.paragraph(
        "La frase de defensa para V1 seria: se han implementado y comparado CNN, DWT, MFCC, STFT, Deep-ONMF normal, Deep-ONMF "
        "mejorado y resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8; se distinguen correctamente "
        "951 audios del protocolo fiel y 1000 audios del protocolo resultado8; y se demuestra que F8 es el mejor equilibrio "
        "global porque mejora tanto la separacion en rasgos como la separacion visual t-SNE."
    )
    pdf.paragraph(
        "Este documento no usa imagenes del articulo como resultados propios. Las figuras proceden de CSV o PNG generados "
        "en la carpeta del TFG. El documento V2 continuara desde aqui para comparar NNDSVD, NNDSVDa y NNDSVDar sobre F8 y "
        "sobre Deep-ONMF mejorado."
    )
    pdf.heading("Guia de defensa oral", level=2)
    pdf.paragraph(
        "Si el tribunal pregunta por que no basta con mostrar la figura t-SNE, la respuesta es que t-SNE ayuda a visualizar "
        "vecindades, pero la decision se apoya tambien en metricas. Por eso se muestran silhouette y Davies-Bouldin tanto "
        "en rasgos como en t-SNE. La figura explica donde se mezclan las clases; las metricas cuantifican esa mezcla."
    )
    pdf.paragraph(
        "Si pregunta por que hay 951 y 1000 audios, la respuesta es que son dos protocolos distintos y ambos estan escritos. "
        "El protocolo fiel al articulo descarta audios menores de 2 segundos. El protocolo resultado8 no los descarta, sino "
        "que los rellena con ceros para conservar la base completa. Por tanto, no se oculta la diferencia: se usa para separar "
        "fidelidad metodologica y mejor resultado empirico."
    )
    pdf.paragraph(
        "Si pregunta por que F8 mejora, la respuesta es que cambia el significado del vector final. STFT, MFCC y DWT describen "
        "el audio; Deep-ONMF normal aprende bases; Deep-ONMF mejorado y F8 usan esas bases para comparar reconstrucciones por "
        "clase. F8 refuerza esa comparacion con un softmin de fuerza 8 y por eso obtiene el mejor equilibrio numerico."
    )
    defensa = pd.DataFrame(
        [
            {"pregunta": "Por que CNN no gana", "respuesta": "aprende texturas, pero no produce una evidencia de reconstruccion por clase"},
            {"pregunta": "Por que DWT mezcla", "respuesta": "las escalas coif5 capturan transitorios que pueden repetirse en varias patologias"},
            {"pregunta": "Por que MFCC no basta", "respuesta": "los 13 coeficientes comprimen la envolvente y pueden perder detalles temporales"},
            {"pregunta": "Por que defender F8", "respuesta": "combina 0.2809 en rasgos, 1.3783 de DB en rasgos, 0.2602 t-SNE y 1.3756 DB t-SNE"},
        ]
    )
    pdf.table(defensa, "Tabla V1. Respuestas cortas para defensa")
    pdf.save()


def generar_v2(datos: dict[str, object], figs: dict[str, Path]) -> None:
    pdf = PDFDoc(
        RESULTADOS / "Documento explicativo v2.pdf",
        "DOCUMENTO EXPLICATIVO V2\nresultado8, perfil_softmin_errores_f8 e inicializaciones",
        "Comparacion entre F8 exacto, F8 con NNDSVD/NNDSVDa/NNDSVDar y Deep-ONMF mejorado",
    )
    comparacion = datos["comparacion_clave"].copy()
    mejor_sil = comparacion.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_db = comparacion.sort_values("davies_bouldin_tsne", ascending=True).iloc[0]
    f8_nombre = "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8"

    pdf.heading("Resumen ejecutivo")
    pdf.callout(
        "La comparacion de V2 se centra en resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8. "
        "Ese resultado usa 1000 audios, tramas de 2 s, solape de 1 s, rangos 9-8-7, 120 iteraciones y fuerza softmin 8."
    )
    pdf.paragraph(
        "Aqui se comparan tres bloques. El primero es F8 exacto, que ya tenia silhouette t-SNE = 0.2602 y Davies-Bouldin "
        "t-SNE = 1.3756. El segundo aplica las inicializaciones del profesor al mismo flujo F8: NNDSVD, NNDSVDa y NNDSVDar, "
        "manteniendo 1000 audios y el vector final perfil_softmin_errores_f8. El tercero recoge Deep-ONMF mejorado con las "
        "mismas tres inicializaciones dentro del protocolo fiel al articulo, donde se usan 951 audios."
    )
    pdf.table(datos["protocolos"], "Diferencia entre protocolos")

    pdf.heading("Bloque 1. F8 exacto con 1000 audios")
    tabla_f8 = metricas_v2_tabla(
        comparacion.loc[comparacion["bloque"] == "F8 exacto con 1000 audios"]
    ).drop(columns=["bloque"])
    pdf.paragraph(
        "Este bloque es el punto de referencia de V2. Usa exactamente el flujo "
        "resultado8_deep_onmf_sin_descartar_menores_2s y conserva los 1000 audios. Los audios menores de 2 s no se eliminan: "
        "se completan con ceros para poder construir tramas de 2 s. El rasgo final no es una activacion interna cualquiera, "
        "sino un perfil de cinco afinidades obtenido al comparar cada audio contra las bases W_N, W_AS, W_MR, W_MS y W_MVP."
    )
    pdf.table(tabla_f8, "Tabla V2. F8 exacto con 1000 audios")
    pdf.image(
        figs["tsne_f8"],
        "Figura V2. t-SNE de resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8",
        "La figura se coloca justo debajo de su tabla porque se lee conjuntamente: la tabla da las metricas y el t-SNE muestra "
        "donde se separan o se mezclan las clases. Aqui silhouette t-SNE = 0.2602 y Davies-Bouldin t-SNE = 1.3756, por eso "
        "este resultado queda como referencia principal para la defensa.",
        height=350,
    )
    pdf.image(
        figs["esquema_softmin"],
        "Figura V2. Dibujo del flujo F8",
        "El dibujo resume la idea: PCG, STFT, bases Deep-ONMF por clase, errores de reconstruccion y softmin_errores(fuerza=8). "
        "La salida final tiene cinco rasgos y cada rasgo representa afinidad relativa con una clase.",
        height=405,
    )
    pdf.paragraph(
        "La razon por la que F8 funciona mejor que leer directamente los SBV es conceptual. Deep-ONMF normal resume que "
        "patrones aparecen en un audio; F8 pregunta algo mas discriminativo: que base de clase reconstruye mejor ese audio. "
        "Si un audio AS se reconstruye claramente mejor con W_AS que con W_N, W_MR, W_MS o W_MVP, el softmin concentra la "
        "afinidad en AS. Si dos clases reconstruyen parecido, la afinidad se reparte, y eso explica por que las zonas mezcladas "
        "del t-SNE bajan silhouette o suben Davies-Bouldin."
    )

    pdf.new_page()
    pdf.heading("Bloque 2. F8 con NNDSVD, NNDSVDa y NNDSVDar")
    fuente_f8_init = comparacion.loc[
        comparacion["bloque"] == "F8 + NNDSVD/NNDSVDa/NNDSVDar con 1000 audios"
    ].copy()
    fuente_f8_init["metodo"] = fuente_f8_init["metodo"].map(etiqueta_v2_pdf)
    tabla_f8_init = metricas_v2_tabla(fuente_f8_init).drop(columns=["bloque"])
    pdf.paragraph(
        "En este bloque se mantienen los 1000 audios y el vector final perfil_softmin_errores_f8, pero se cambia la forma de "
        "arrancar la factorizacion ONMF. NNDSVD inicializa W y H desde una estructura SVD no negativa; NNDSVDa rellena ceros "
        "con la media; NNDSVDar rellena ceros con ruido pequeno. La pregunta no es si estas variantes son teoricamente mas "
        "bonitas, sino si al aplicarlas al flujo F8 mejoran las metricas reales."
    )
    pdf.table(tabla_f8_init, "Tabla V2. F8 + NNDSVD / NNDSVDa / NNDSVDar con 1000 audios")
    pdf.image(
        figs["tsne_f8_inits"],
        "Figura V2. t-SNE de F8 con las tres inicializaciones",
        "Cada panel usa 1000 audios y vuelve a calcular los errores de reconstruccion contra bases entrenadas con la "
        "inicializacion indicada. Visualmente se comprueba si la inicializacion ordena mejor los grupos o si rompe parte de "
        "la separacion que ya tenia el F8 exacto.",
        height=360,
    )
    pdf.code_block(extraer_funcion_codigo(CODIGO_GENERACION_FINAL, "nndsvd_inicializar", max_lineas=34), "Codigo real: NNDSVD, NNDSVDa y NNDSVDar")
    pdf.paragraph(
        "La lectura de estos numeros es importante para el tribunal. NNDSVD y sus variantes pueden estabilizar el arranque de "
        "ONMF, pero no garantizan superar el mejor resultado empirico. Si la base inicial queda mas ordenada pero luego la "
        "separacion por clase disminuye, silhouette baja. Si los grupos quedan mas alargados o cercanos, Davies-Bouldin sube. "
        "Por eso V2 conserva la comparacion: permite defender con datos si conviene quedarse con F8 exacto o con una variante "
        "inicializada."
    )

    pdf.new_page()
    pdf.heading("Bloque 3. Deep-ONMF mejorado con inicializaciones en 951 audios")
    tabla_actual = metricas_v2_tabla(
        comparacion.loc[
            comparacion["bloque"] == "Deep-ONMF mejorado + NNDSVD/NNDSVDa/NNDSVDar con 951 audios"
        ]
    ).drop(columns=["bloque"])
    pdf.paragraph(
        "Este bloque pertenece al protocolo fiel al articulo. Aqui se usan 951 audios porque se descartan los menores de 2 s: "
        "16 MR, 14 MS y 19 MVP. Por eso no se debe decir que estos valores comparan la misma poblacion que F8 exacto. Lo que "
        "si se puede defender es que, dentro del protocolo estricto, se prueba Deep-ONMF normal, Deep-ONMF mejorado y las tres "
        "inicializaciones propuestas."
    )
    pdf.table(tabla_actual, "Tabla V2. Deep-ONMF mejorado + inicializaciones con 951 audios")
    pdf.image(
        figs["tsne_variantes"],
        "Figura V2. t-SNE de Deep-ONMF mejorado con NNDSVD, NNDSVDa y NNDSVDar",
        "Esta figura sale de los CSV del protocolo fiel al articulo. La lectura correcta es interna a este bloque: se mira si "
        "las inicializaciones mejoran al Deep-ONMF mejorado bajo la politica de descarte de 951 audios.",
        height=350,
    )
    pdf.paragraph(
        "La diferencia frente a F8 es que aqui se mantiene el marco estricto del articulo. Deep-ONMF mejorado tambien usa un "
        "vector de cinco rasgos basado en reconstruccion por clase, pero no coincide completamente con resultado8 porque cambia "
        "la politica de datos. Si una inicializacion mejora el t-SNE de este bloque, se explica como mejora dentro del protocolo "
        "fiel; si no supera a F8, se justifica que F8 sigue siendo el mejor resultado global sobre los 1000 audios."
    )

    pdf.new_page()
    pdf.heading("Comparacion global para decidir que defender")
    pdf.paragraph(
        "Una vez vistos los bloques por separado, la comparacion global sirve para tomar una decision. La tabla siguiente no "
        "oculta que hay dos protocolos: 1000 audios para resultado8 y 951 audios para el protocolo fiel. Se coloca todo junto "
        "para que se vea que opcion domina en silhouette, que opcion domina en Davies-Bouldin y que coste metodologico tiene "
        "cada eleccion."
    )
    tabla_decision_pdf = metricas_v2_decision(comparacion)
    tabla_decision_pdf["metodo"] = tabla_decision_pdf["metodo"].map(etiqueta_v2_pdf)
    tabla_decision_pdf["bloque"] = tabla_decision_pdf["bloque"].map(
        {
            "F8 exacto con 1000 audios": "F8 exacto (1000)",
            "F8 + NNDSVD/NNDSVDa/NNDSVDar con 1000 audios": "F8 + inicializaciones (1000)",
            "Deep-ONMF mejorado + NNDSVD/NNDSVDa/NNDSVDar con 951 audios": "Deep-ONMF mejorado + inicializaciones (951)",
        }
    )
    pdf.table(tabla_decision_pdf, "Tabla V2. Resumen de decision")
    pdf.image(
        figs["v2_comparativa_sil"],
        "Figura V2. Comparacion directa por silhouette t-SNE",
        "Silhouette alto significa vecindarios mas limpios: puntos de la misma clase cerca y puntos de clases distintas lejos. "
        "Si F8 exacto queda arriba, se defiende que la representacion por afinidad de errores separa mejor que las variantes "
        "inicializadas.",
        height=430,
    )
    pdf.image(
        figs["v2_comparativa_db"],
        "Figura V2. Comparacion directa por Davies-Bouldin t-SNE",
        "Davies-Bouldin se interpreta al reves: menor es mejor. Penaliza grupos dispersos y grupos demasiado cercanos. Esta "
        "grafica complementa silhouette porque una separacion visual buena tambien debe tener compacidad.",
        height=430,
    )
    pdf.image(
        figs["ranking_sil"],
        "Figura V2. Ranking de silhouette t-SNE",
        f"El mejor silhouette t-SNE de esta ejecucion queda en {mejor_sil['metodo']} con valor "
        f"{mejor_sil['silhouette_tsne']:.4f}. Si gana una variante F8 con inicializacion, se defenderia esa variante; "
        "si gana el F8 exacto, se mantiene como resultado principal.",
        height=430,
    )
    pdf.image(
        figs["ranking_db"],
        "Figura V2. Ranking de Davies-Bouldin t-SNE",
        f"Davies-Bouldin se interpreta al reves: menor es mejor. El menor valor queda en {mejor_db['metodo']} con "
        f"{mejor_db['davies_bouldin_tsne']:.4f}. Esta metrica penaliza grupos estirados o demasiado cercanos.",
        height=430,
    )

    pdf.new_page()
    pdf.heading("Barrido historico de mejoras")
    ranking_cols = ["variante", "muestras", "rasgos", "silhouette_tsne", "davies_bouldin_tsne"]
    barrido = datos["ranking_barrido"].copy().rename(columns={"rasgos": "rasgos"})
    pdf.table(barrido, "Primeras variantes del barrido ordenadas por silhouette t-SNE", columns=ranking_cols, max_rows=12)
    pdf.image(
        figs["barrido"],
        "Figura V2. Barrido historico",
        "El barrido muestra que no todas las formas de leer Deep-ONMF mejoran. Las variantes que mezclan demasiados resumenes "
        "de H pueden aumentar dimension sin separar mejor. El perfil softmin f8 funciona porque concentra la pregunta correcta: "
        "que base de clase reconstruye mejor este audio.",
        height=405,
    )
    pdf.image(
        figs["mejora_relativa"],
        "Figura V2. Mejora frente a SBV base",
        "La barra de f8 indica cuantas veces mejora el silhouette t-SNE respecto al Deep-ONMF normal basado en SBV medios.",
        height=330,
    )
    if figs.get("comparacion_ajustada_f8") and Path(figs["comparacion_ajustada_f8"]).exists():
        pdf.image(
            figs["comparacion_ajustada_f8"],
            "Figura V2. Comparacion generada en la carpeta resultado8",
            "Esta imagen procede de comparacion_ajustada_completa.zip, no del articulo. Es la comparativa generada con el "
            "resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8.",
            height=285,
        )

    pdf.new_page()
    pdf.heading("Por que perfil_softmin_errores_f8 funciona")
    pdf.paragraph(
        "Deep-ONMF aprende bases W. La lectura SBV directa resume patrones, pero no comprueba si un audio de una clase tambien "
        "puede ser reconstruido por las bases de las otras. perfil_softmin_errores_f8 cambia la pregunta: calcula el error de "
        "reconstruccion contra W_N, W_AS, W_MR, W_MS y W_MVP, divide cada error por el menor error del audio, y aplica un softmin "
        "con fuerza 8. El vector final tiene cinco rasgos y se interpreta como perfil de afinidad por clase."
    )
    pdf.code_block(extraer_funcion_codigo(HIST_CODIGO_BARRIDO, "softmin_errores", max_lineas=10), "Codigo real: softmin_errores")
    pdf.paragraph(
        "La clave esta en la normalizacion relativa. No se usan errores absolutos sin mas: cada audio se compara consigo mismo, "
        "dividiendo por su menor error. Asi el metodo pregunta que base reconstruye mejor ese audio y con cuanta diferencia "
        "respecto a las demas."
    )
    pdf.heading("Definicion explicita de softmin_errores(fuerza=8)", level=2)
    pdf.paragraph(
        "Para cada audio se obtiene un vector de errores e = [e_N, e_AS, e_MR, e_MS, e_MVP]. softmin_errores no usa esos "
        "errores directamente: primero los normaliza por el menor error del propio audio y despues aplica una exponencial "
        "negativa. La formula es relativo_c = e_c / min(e); logit_c = -8 * (relativo_c - 1); afinidad_c = exp(logit_c) / "
        "sum_j exp(logit_j). La salida suma 1 y puede leerse como un perfil de afinidad por clase."
    )
    pdf.table(ejemplo_softmin_f8(), "Ejemplo pedagogico de softmin_errores(fuerza=8)")
    pdf.code_block(extraer_funcion_codigo(HIST_CODIGO_BARRIDO, "proyectar_resultado", max_lineas=30), "Codigo real: proyeccion contra bases W")
    pdf.paragraph(
        "Este bloque recorre los audios y proyecta cada uno contra las cinco bases de clase. Por eso el resultado no es una "
        "foto inventada: cada fila del CSV f8 sale de cinco reconstrucciones reales, una por clase."
    )
    pdf.code_block(extraer_funcion_codigo(HIST_CODIGO_BARRIDO, "variantes_rasgos", max_lineas=24), "Codigo real: creacion de perfil_softmin_errores_f8")
    pdf.paragraph(
        "En variantes_rasgos se ve por que f8 no es un nombre decorativo: es la variante construida con softmin_errores usando "
        "fuerza 8. En el barrido se comparo con f4, f12, f16, f24, SBV, resumenes de H y afinidades por error."
    )

    pdf.heading("Politica de 1000 audios frente a 951 audios")
    pdf.paragraph(
        "resultado8_deep_onmf_sin_descartar_menores_2s usa 1000 audios porque rellena con ceros los registros menores de 2 s. "
        "El protocolo fiel al articulo usa 951 porque descarta esos audios: 16 MR, 14 MS y 19 MVP. Por eso, cuando se comparan "
        "F8 y Deep-ONMF mejorado con inicializaciones, hay que decir explicitamente que tambien se comparan politicas de datos."
    )
    pdf.code_block(
        extraer_lineas_con_patron(HIST_CODIGO_PIPELINE, ["No se descarta", "se rellena con ceros", "Descarte de audios"], contexto=2),
        "Codigo real: politica de audio corto",
    )

    pdf.new_page()
    pdf.heading("Convergencia de 0 a 300 iteraciones")
    pdf.table(
        convergencia_v2_tabla(datos["actual_conv"]),
        "Tabla V2. Convergencia 0-120 frente a 120-300",
    )
    pdf.image(
        figs["conv_error_clara"],
        "Figura V2. Convergencia clara en 0, 120 y 300 iteraciones",
        "Esta version resume la convergencia en los tres puntos que se defienden en el texto. Se usa escala logaritmica para "
        "que las tres capas se puedan comparar sin que el error inicial tape los valores finales. La zona azul marca el "
        "aprendizaje principal hasta 120 y la zona gris el ajuste residual de 120 a 300.",
        height=340,
    )
    pdf.image(
        figs["conv_mejora"],
        "Figura V2. Mejora acumulada antes y despues de 120 iteraciones",
        "La grafica separa la mejora obtenida hasta 120 iteraciones de la mejora residual entre 120 y 300. Sirve para defender "
        "que 120 no es un valor elegido a conveniencia visual, sino un punto donde la mayor parte de la convergencia ya ocurrio.",
        height=340,
    )
    pdf.paragraph(
        "Capa 1, rango 9: el error pasa de 2.6779 a 0.0923 en 120 iteraciones y a 0.0868 en 300. Eso significa que el 99.7882% "
        "de la mejora total observada hasta 300 ya se consigue antes de 120. La primera capa aprende la estructura espectral "
        "mas gruesa y por eso converge muy rapido."
    )
    pdf.paragraph(
        "Capa 2, rango 8: el error baja de 1.2908 a 0.1745 en 120 iteraciones y a 0.1610 en 300. Aqui todavia queda algo mas "
        "de margen que en la capa 1, pero el 98.8007% de la mejora total ya esta antes de 120. Seguir hasta 300 afina, pero "
        "no cambia de forma proporcional la separacion de clases."
    )
    pdf.paragraph(
        "Capa 3, rango 7: el error baja de 1.3651 a 0.2494 y en 300 queda en 0.2489. El 99.9589% de la mejora ya aparece antes "
        "de 120. Esta es la capa que produce los 7 SBV finales, asi que el resultado muestra que la representacion final queda "
        "practicamente estable antes de llegar a 300."
    )
    pdf.paragraph(
        "La conclusion de convergencia es que 120 iteraciones es un compromiso razonable. Aumentar hasta 300 reduce algo el "
        "error, pero la ganancia adicional es pequena frente al coste y no garantiza que silhouette o Davies-Bouldin mejoren. "
        "En Deep-ONMF no basta con minimizar reconstruccion: tambien hay que conservar una representacion que separe clases."
    )
    pdf.heading("Version defendible ante tribunal", level=2)
    if f8_nombre in str(mejor_sil["metodo"]):
        conclusion = (
            f"Segun silhouette t-SNE, la opcion a defender es {mejor_sil['metodo']} con "
            f"{mejor_sil['silhouette_tsne']:.4f}. Esta conclusion favorece el bloque de 1000 audios y el perfil f8."
        )
    else:
        conclusion = (
            f"Segun silhouette t-SNE, la opcion que mas separa visualmente es {mejor_sil['metodo']} con "
            f"{mejor_sil['silhouette_tsne']:.4f}. En defensa hay que explicar que pertenece al protocolo de 951 audios."
        )
    pdf.paragraph(conclusion)
    pdf.paragraph(
        f"Segun Davies-Bouldin t-SNE, la mejor compacidad corresponde a {mejor_db['metodo']} con "
        f"{mejor_db['davies_bouldin_tsne']:.4f}. Si silhouette y Davies-Bouldin no eligen exactamente la misma fila, la defensa "
        "debe decirlo con transparencia y justificar la eleccion por equilibrio entre separacion visual, compacidad y protocolo."
    )
    pdf.paragraph(
        "La frase corta para la defensa seria: se reprodujo el articulo con sus parametros, se corrigio la comparativa de CNN, "
        "se estudio resultado8_deep_onmf_sin_descartar_menores_2s__perfil_softmin_errores_f8, se aplicaron NNDSVD, NNDSVDa y "
        "NNDSVDar tanto al flujo f8 como al Deep-ONMF mejorado, y se eligio la fila con mejor equilibrio metrico sin ocultar "
        "la diferencia entre 1000 y 951 audios."
    )
    pdf.save()


def generar_historico(datos: dict[str, object], figs: dict[str, Path]) -> None:
    pdf = PDFDoc(
        RESULTADOS / "Documento explicativo comparativa historica y mejoras.pdf",
        "DOCUMENTO EXPLICATIVO\nComparativa historica y mejoras posibles",
        "Evolucion de pruebas desde comparacion final hasta la documentacion actual",
    )
    pdf.heading("Resumen ejecutivo")
    pdf.paragraph(
        "Este documento reconstruye la historia de las pruebas. Primero se partio de Deep-ONMF normal, usando siete SBV "
        "medios por audio. Despues se observo que esa representacion no siempre ganaba frente a CNN. La mejora importante "
        "fue dejar de mirar solo activaciones internas y empezar a comparar cada audio contra las bases de todas las clases. "
        "Ese cambio dio lugar al perfil_softmin_errores_f8, que es el mejor resultado global encontrado."
    )
    pdf.heading("Linea temporal de mejoras")
    timeline = pd.DataFrame(
        [
            {"fase": "Deep-ONMF base", "cambio": "usar SBV medios", "resultado": "representacion interpretable pero menos separable"},
            {"fase": "Barrido de rasgos", "cambio": "probar resumenes de H y errores", "resultado": "se descubre que los errores por base discriminan mejor"},
            {"fase": "perfil_softmin_errores_f8", "cambio": "convertir errores en afinidades", "resultado": "mejor silhouette y mejor DB t-SNE historicos"},
            {"fase": "Protocolo fiel", "cambio": "descartar menores de 2 s y corregir CNN", "resultado": "comparacion metodologicamente estricta con 951 audios"},
            {"fase": "Inicializaciones", "cambio": "NNDSVD, NNDSVDa, NNDSVDar", "resultado": "NNDSVD mejora dentro del protocolo actual"},
        ]
    )
    pdf.table(timeline, "Evolucion de las decisiones experimentales")
    pdf.heading("Que mejora realmente los resultados")
    mejoras = pd.DataFrame(
        [
            {"mejora": "Softmin de errores", "por que ayuda": "convierte reconstruccion por clase en afinidades comparables", "riesgo": "depende de bases W estables"},
            {"mejora": "Conservar 1000 audios", "por que ayuda": "mantiene toda la distribucion local usada en la comparativa historica", "riesgo": "se aleja del descarte estricto del paper"},
            {"mejora": "NNDSVD", "por que ayuda": "arranca ONMF desde estructura SVD no negativa", "riesgo": "no corrige por si solo una mala representacion final"},
            {"mejora": "Mas rasgos de H", "por que ayuda": "anade informacion estadistica", "riesgo": "puede meter ruido y empeorar t-SNE"},
            {"mejora": "Mas iteraciones", "por que ayuda": "reduce algo el error", "riesgo": "ganancia pequena tras 120 iteraciones"},
        ]
    )
    pdf.table(mejoras, "Mejoras posibles y lectura critica")
    pdf.heading("Resultado numerico clave")
    pdf.table(metricas_compactas(datos["comparacion_clave"]), "Comparacion consolidada")
    pdf.image(
        figs["esquema_softmin"],
        "Figura H1. Mecanismo del perfil f8",
        "La mejora se entiende mejor como un cambio de pregunta: de cuanto se activa una base a que base reconstruye mejor.",
        height=420,
    )
    pdf.image(
        figs["barrido"],
        "Figura H2. Evidencia del barrido",
        "El barrido prueba empiricamente que f8 no fue elegido a mano despues de mirar una unica imagen, sino dentro de una "
        "familia de variantes comparadas con las mismas metricas.",
        height=405,
    )
    pdf.heading("Conclusion defendible")
    pdf.paragraph(
        "La conclusion que se debe defender es doble. Desde el punto de vista metodologico, se ha reproducido el articulo "
        "con sus parametros principales y se ha corregido la CNN para que coincida con la descripcion publicada. Desde el "
        "punto de vista de rendimiento, el mejor resultado global sigue siendo perfil_softmin_errores_f8. No hay contradiccion: "
        "un protocolo explica fidelidad al paper y el otro explica la mejor mejora empirica obtenida sobre la base local."
    )
    pdf.save()


def limpiar_raiz_resultados() -> None:
    permitidos = {
        "fotos datos y graficas",
        "Documento explicativo v1.pdf",
        "Documento explicativo v2.pdf",
    }
    for item in RESULTADOS.iterdir():
        if item.name not in permitidos:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def verificar() -> pd.DataFrame:
    filas = []
    objetivos = [
        RESULTADOS / "Documento explicativo v1.pdf",
        RESULTADOS / "Documento explicativo v2.pdf",
    ]
    palabras = [
        "resultado8_deep_onmf_sin_descartar_menores_2s",
        "perfil_softmin_errores_f8",
        "0.2602",
        "1.3756",
        "1000",
        "951",
        "NNDSVD",
        "NNDSVDa",
        "NNDSVDar",
        "Normal actual",
        "Mejorado actual",
        "Historico f8",
    ]
    for pdf in objetivos:
        doc = fitz.open(pdf)
        text = "\n".join(page.get_text() for page in doc)
        verticales = all(page.rect.width < page.rect.height for page in doc)
        vacias = sum(1 for page in doc if not page.get_text().strip() and not page.get_images())
        pobres = sum(1 for page in doc if len(page.get_text().strip()) < 80 and not page.get_images())
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(1.05, 1.05), alpha=False)
            safe = pdf.stem.replace(" ", "_")
            pix.save(VERIF / f"preview_{safe}_pagina_{i + 1}.png")
        filas.append(
            {
                "pdf": pdf.name,
                "paginas": len(doc),
                "v1_minimo_18_paginas": len(doc) >= 18 if "v1" in pdf.name else "",
                "verticales": verticales,
                "paginas_vacias": vacias,
                "paginas_pobres": pobres,
                "imagenes": sum(len(page.get_images(full=True)) for page in doc),
                "sin_ranking_silhouette_en_v1": ("Figura V1. Ranking de silhouette t-SNE" not in text) if "v1" in pdf.name else "",
                "v1_contiene_mejora_cnn_951_1000": all(s in text for s in ["0.1106", "0.2259", "+104.31%", "2.0143", "3.6186"]) if "v1" in pdf.name else "",
                **{f"contiene_{p}": (p in text) for p in palabras},
            }
        )
        doc.close()
    df = pd.DataFrame(filas)
    guardar_csv(df, VERIF / "verificacion_documentos_estilo_referencia.csv")
    return df


def main() -> None:
    preparar_salida()
    datos = cargar_datos()
    guardar_tablas(datos)
    figs = generar_figuras(datos)
    limpiar_raiz_resultados()
    generar_v1(datos, figs)
    generar_v2(datos, figs)
    df_verif = verificar()
    print(df_verif.to_string(index=False))


if __name__ == "__main__":
    main()
