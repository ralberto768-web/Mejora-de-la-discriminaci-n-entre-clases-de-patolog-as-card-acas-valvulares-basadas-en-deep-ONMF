from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .audio import DatosClase, construir_matriz_clase, descubrir_audios
from .configuracion import Configuracion
from .estadistica import caracteristicas_por_audio, resumen_auditoria
from .onmf import ResultadoONMF, deep_onmf


COLORES = {
    "N": "#1b9e77",
    "AS": "#d95f02",
    "MR": "#7570b3",
    "MS": "#e7298a",
    "MVP": "#66a61e",
}


CONFIGURACIONES_ONMF = [
    {"nombre": "onmf_iter120_semilla42", "iteraciones": 120, "semilla": 42},
    {"nombre": "onmf_iter120_semilla7", "iteraciones": 120, "semilla": 7},
    {"nombre": "onmf_iter120_semilla99", "iteraciones": 120, "semilla": 99},
    {"nombre": "onmf_iter300_semilla42", "iteraciones": 300, "semilla": 42},
]


def _siguiente_carpeta(base: Path, etiqueta: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    indice = 1
    while True:
        carpeta = base / f"resultado{indice}-{etiqueta}"
        if not carpeta.exists():
            carpeta.mkdir(parents=True)
            return carpeta
        indice += 1


def _preparar_x(df: pd.DataFrame, columnas: list[str], escalado: str) -> np.ndarray:
    x = df[columnas].to_numpy(dtype=float)
    if escalado == "standard":
        return StandardScaler().fit_transform(x)
    if escalado == "minmax":
        return MinMaxScaler().fit_transform(x)
    if escalado == "ninguno":
        return x
    raise ValueError(f"Escalado no reconocido: {escalado}")


def _calcular_metricas(x: np.ndarray, etiquetas: np.ndarray) -> tuple[float, float]:
    clases_unicas = np.unique(etiquetas)
    if len(clases_unicas) < 2 or len(x) <= len(clases_unicas):
        return float("nan"), float("nan")
    return float(silhouette_score(x, etiquetas)), float(davies_bouldin_score(x, etiquetas))


def _figura_tsne(
    df: pd.DataFrame,
    columnas: list[str],
    clases: tuple[str, ...],
    escalado: str,
    titulo: str,
    ruta_png: Path,
    ruta_csv: Path,
) -> dict[str, object]:
    x = _preparar_x(df, columnas, escalado)
    etiquetas = df["clase"].to_numpy()
    perplexity = min(30, max(5, (len(df) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
        max_iter=1000,
    )
    coords = tsne.fit_transform(x)
    salida = df[[c for c in ["clase", "archivo", "origen"] if c in df.columns]].copy()
    salida["tSNE_1"] = coords[:, 0]
    salida["tSNE_2"] = coords[:, 1]
    salida.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    fig, eje = plt.subplots(figsize=(9, 7))
    for clase in clases:
        mascara = salida["clase"] == clase
        eje.scatter(
            salida.loc[mascara, "tSNE_1"],
            salida.loc[mascara, "tSNE_2"],
            s=18,
            alpha=0.78,
            color=COLORES.get(clase, "#666666"),
            label=clase,
            edgecolors="none",
        )
    eje.set_title(titulo)
    eje.set_xlabel("t-SNE 1")
    eje.set_ylabel("t-SNE 2")
    eje.grid(True, alpha=0.2)
    eje.legend(title="Clase")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=220)
    plt.close(fig)

    sil, db = _calcular_metricas(x, etiquetas)
    return {
        "n_puntos": len(df),
        "n_features": len(columnas),
        "escalado": escalado,
        "perplexity": perplexity,
        "silhouette_features": sil,
        "davies_bouldin_features": db,
    }


def _features_w_filas(w_por_clase: dict[str, np.ndarray], clases: tuple[str, ...]) -> pd.DataFrame:
    filas = []
    for clase in clases:
        w = w_por_clase[clase]
        for fila in range(w.shape[0]):
            registro = {"clase": clase, "archivo": f"frecuencia_bin_{fila:03d}", "origen": "filas_W_final"}
            for indice in range(w.shape[1]):
                registro[f"F_{indice + 1}"] = float(w[fila, indice])
            filas.append(registro)
    return pd.DataFrame(filas)


def _features_h_columnas(
    h_por_clase: dict[str, np.ndarray],
    clases: tuple[str, ...],
    max_columnas_por_clase: int,
    semilla: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    filas = []
    for clase in clases:
        h = h_por_clase[clase]
        n = h.shape[1]
        seleccion = np.arange(n)
        if n > max_columnas_por_clase:
            seleccion = rng.choice(seleccion, size=max_columnas_por_clase, replace=False)
        for columna in sorted(seleccion.tolist()):
            registro = {"clase": clase, "archivo": f"columna_H_{columna:06d}", "origen": "columnas_H_final"}
            for indice in range(h.shape[0]):
                registro[f"F_{indice + 1}"] = float(h[indice, columna])
            filas.append(registro)
    return pd.DataFrame(filas)


def _ejecutar_onmf(
    configuracion: Configuracion,
    iteraciones: int,
    semilla: int,
) -> tuple[dict[str, DatosClase], dict[str, ResultadoONMF], dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    cfg = replace(configuracion, iteraciones_onmf=iteraciones, semilla=semilla)
    registros = descubrir_audios(cfg.carpeta_base_datos, cfg.clases)
    datos_por_clase: dict[str, DatosClase] = {}
    resultados: dict[str, ResultadoONMF] = {}
    w_por_clase: dict[str, np.ndarray] = {}
    h_por_clase: dict[str, np.ndarray] = {}

    for posicion, clase in enumerate(cfg.clases, start=1):
        print(f"  Deep ONMF {clase} | iter={iteraciones} | semilla={semilla}")
        datos = construir_matriz_clase(clase, registros, cfg)
        resultado = deep_onmf(
            datos.matriz,
            rangos=cfg.rangos_onmf,
            iteraciones=cfg.iteraciones_onmf,
            penalizacion_ortogonal=cfg.penalizacion_ortogonal,
            semilla=cfg.semilla + posicion * 100,
        )
        datos_por_clase[clase] = datos
        resultados[clase] = resultado
        w_por_clase[clase] = resultado.w_final
        h_por_clase[clase] = resultado.h_final

    auditoria = resumen_auditoria(registros, datos_por_clase, cfg.clases)
    return datos_por_clase, resultados, w_por_clase, h_por_clase, auditoria


def _informe(
    carpeta: Path,
    resumen: pd.DataFrame,
    auditoria: pd.DataFrame,
    rellenar_audios_cortos: bool,
) -> str:
    mejor_sil = resumen.sort_values("silhouette_features", ascending=False).iloc[0]
    mejor_db = resumen.sort_values("davies_bouldin_features", ascending=True).iloc[0]
    if rellenar_audios_cortos:
        modo_audios = (
            "En estos testeos no se descarta ningun audio menor de 2 segundos. "
            "Esos audios se rellenan con ceros hasta 2 segundos para usar toda la base disponible."
        )
    else:
        modo_audios = (
            "En estos testeos se aplica el criterio literal del articulo: "
            "los audios menores de 2 segundos no entran en las matrices X."
        )
    lineas = [
        "INFORME DE TESTEOS DEEP-ONMF PARA FIGURA 11D",
        "",
        "Objetivo:",
        "Comprobar por qué la Figura 11D generada no se parece a la del artículo.",
        "",
        "Criterio usado para audios cortos:",
        modo_audios,
        "",
        "Punto crítico detectado en el texto del artículo:",
        "En la sección 6.7 los autores dicen que, para comparar métodos en la Figura 11,",
        "pasan de analizar SBV individuales a usar la matriz W final completa como conjunto",
        "de características. Por eso se prueban varias formas de construir la entrada de t-SNE.",
        "",
        "Carpeta de resultados:",
        str(carpeta),
        "",
        "Auditoria del criterio de datos usado:",
        auditoria.to_string(index=False),
        "",
        "Resumen de variantes:",
        resumen.to_string(index=False),
        "",
        "Mejor variante por silhouette en el espacio de características:",
        f"{mejor_sil['configuracion_onmf']} / {mejor_sil['variante']} / escalado={mejor_sil['escalado']} "
        f"con silhouette={mejor_sil['silhouette_features']:.4f}",
        "",
        "Mejor variante por Davies-Bouldin en el espacio de características:",
        f"{mejor_db['configuracion_onmf']} / {mejor_db['variante']} / escalado={mejor_db['escalado']} "
        f"con DB={mejor_db['davies_bouldin_features']:.4f}",
        "",
        "Lectura:",
        "Si la variante basada en filas de W se parece más al documento, la diferencia venía de",
        "usar antes características por audio derivadas de H. Si ninguna variante reproduce la Figura 11D,",
        "entonces faltan detalles no publicados: normalización exacta, inicialización, optimizador,",
        "selección de muestras o parámetros internos de t-SNE.",
    ]
    return "\n".join(lineas)


def ejecutar_testeos_figura11d(configuracion: Configuracion) -> Path:
    etiqueta = (
        "testeos_deep_onmf_figura11D_sin_descartar"
        if configuracion.rellenar_audios_cortos
        else "testeos_deep_onmf_figura11D"
    )
    carpeta = _siguiente_carpeta(configuracion.raiz / "resultados_deep_onmf", etiqueta)
    carpeta_figuras = carpeta / "figuras_11D"
    carpeta_datos = carpeta / "datos"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)
    carpeta_datos.mkdir(parents=True, exist_ok=True)

    resumen_filas = []
    auditoria_referencia = None

    for cfg_onmf in CONFIGURACIONES_ONMF:
        nombre_cfg = cfg_onmf["nombre"]
        print(f"Ejecutando {nombre_cfg}")
        datos_por_clase, resultados, w_por_clase, h_por_clase, auditoria = _ejecutar_onmf(
            configuracion,
            iteraciones=cfg_onmf["iteraciones"],
            semilla=cfg_onmf["semilla"],
        )
        if auditoria_referencia is None:
            auditoria_referencia = auditoria
            auditoria.to_csv(carpeta_datos / "auditoria_deep_onmf.csv", index=False, encoding="utf-8-sig")

        features_h_audio = caracteristicas_por_audio(datos_por_clase, h_por_clase)
        features_h_audio = features_h_audio.rename(columns={f"SBV_{i}": f"F_{i}" for i in range(1, 8)})
        features_h_audio["origen"] = "media_H_por_audio"
        features_w_filas = _features_w_filas(w_por_clase, configuracion.clases)
        features_h_columnas = _features_h_columnas(
            h_por_clase,
            configuracion.clases,
            max_columnas_por_clase=220,
            semilla=cfg_onmf["semilla"],
        )

        variantes = [
            ("H_media_por_audio", features_h_audio, ["standard", "ninguno"]),
            ("W_filas_matriz_final", features_w_filas, ["standard", "minmax", "ninguno"]),
            ("H_columnas_muestreadas", features_h_columnas, ["standard"]),
        ]

        for variante, df, escalados in variantes:
            columnas = [c for c in df.columns if c.startswith("F_")]
            df.to_csv(carpeta_datos / f"{nombre_cfg}__{variante}__features.csv", index=False, encoding="utf-8-sig")
            for escalado in escalados:
                base_nombre = f"{nombre_cfg}__{variante}__{escalado}"
                print(f"  t-SNE {base_nombre}")
                metricas = _figura_tsne(
                    df=df,
                    columnas=columnas,
                    clases=configuracion.clases,
                    escalado=escalado,
                    titulo=f"Figura 11D test - {variante} - {nombre_cfg} - {escalado}",
                    ruta_png=carpeta_figuras / f"{base_nombre}.png",
                    ruta_csv=carpeta_datos / f"{base_nombre}__coordenadas_tsne.csv",
                )
                resumen_filas.append(
                    {
                        "configuracion_onmf": nombre_cfg,
                        "variante": variante,
                        **metricas,
                    }
                )

        errores = [
            {
                "clase": clase,
                "error_relativo_final": resultado.error_relativo_final,
                "iteraciones": cfg_onmf["iteraciones"],
                "semilla": cfg_onmf["semilla"],
            }
            for clase, resultado in resultados.items()
        ]
        pd.DataFrame(errores).to_csv(carpeta_datos / f"{nombre_cfg}__errores_onmf.csv", index=False, encoding="utf-8-sig")

    resumen = pd.DataFrame(resumen_filas)
    resumen.to_csv(carpeta / "resumen_testeos_figura11D.csv", index=False, encoding="utf-8-sig")
    (carpeta / "00_INFORME_TESTEOS_FIGURA11D.txt").write_text(
        _informe(
            carpeta,
            resumen,
            auditoria_referencia if auditoria_referencia is not None else pd.DataFrame(),
            configuracion.rellenar_audios_cortos,
        ),
        encoding="utf-8-sig",
    )

    (carpeta / "parametros_testeos.json").write_text(
        json.dumps(CONFIGURACIONES_ONMF, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return carpeta
