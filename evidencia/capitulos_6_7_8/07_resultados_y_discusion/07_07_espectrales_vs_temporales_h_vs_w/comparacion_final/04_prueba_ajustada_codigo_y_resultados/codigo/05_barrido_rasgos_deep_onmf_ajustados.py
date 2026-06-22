from __future__ import annotations

"""Prueba rasgos Deep ONMF ajustados para la comparacion de Figura 11."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys

RAIZ_CARPETA = Path(__file__).resolve().parents[1]
RAIZ_COMPARACION = RAIZ_CARPETA.parent
RAIZ_OBJETIVO = RAIZ_COMPARACION.parent
SRC_OBJETIVO = RAIZ_OBJETIVO / "src"
MPL_CACHE = RAIZ_CARPETA / "resultados" / ".cache_matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

if str(SRC_OBJETIVO) not in sys.path:
    sys.path.insert(0, str(SRC_OBJETIVO))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from tfg_deep_onmf.audio import construir_matriz_audio, descubrir_audios
from tfg_deep_onmf.configuracion import Configuracion
from tfg_deep_onmf.onmf import proyectar_sobre_w


CLASES = ("N", "AS", "MR", "MS", "MVP")


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barre representaciones Deep ONMF para Figura 11.")
    parser.add_argument(
        "--patron",
        default="resultados/resultado*-deep_onmf*/documentacion_tecnica/caracteristicas_sbv_por_audio.csv",
        help="Patron relativo a Programacion objetivo para localizar resultados Deep ONMF.",
    )
    parser.add_argument("--max-resultados", type=int, default=0, help="0 usa todos los resultados localizados.")
    parser.add_argument("--semilla-tsne", type=int, default=42, help="Semilla comun para PCA y t-SNE.")
    return parser.parse_args()


def crear_salida() -> dict[str, Path]:
    base = RAIZ_CARPETA / "resultados"
    carpetas = {
        "base": base,
        "rasgos": base / "01_rasgos_candidatos",
        "coordenadas": base / "02_coordenadas_barrido",
    }
    for carpeta in carpetas.values():
        carpeta.mkdir(parents=True, exist_ok=True)
    return carpetas


def leer_parametros(ruta_csv: Path) -> dict[str, object]:
    ruta_json = ruta_csv.parent / "parametros_configuracion.json"
    return json.loads(ruta_json.read_text(encoding="utf-8-sig"))


def configuracion_desde_parametros(parametros: dict[str, object]) -> Configuracion:
    return Configuracion(
        raiz=RAIZ_OBJETIVO,
        frecuencia_esperada_hz=int(parametros["frecuencia_esperada_hz"]),
        duracion_trama_s=float(parametros["duracion_trama_s"]),
        solape_trama_s=float(parametros["solape_trama_s"]),
        longitud_ventana=int(parametros["longitud_ventana"]),
        salto_ventana=int(parametros["salto_ventana"]),
        puntos_fft=int(parametros["puntos_fft"]),
        rangos_onmf=tuple(int(valor) for valor in parametros["rangos_onmf"]),
        iteraciones_onmf=int(parametros["iteraciones_onmf"]),
        penalizacion_ortogonal=float(parametros["penalizacion_ortogonal"]),
        semilla=int(parametros["semilla"]),
        rellenar_audios_cortos=bool(parametros["rellenar_audios_cortos"]),
    )


def preparar_tsne(x: np.ndarray, etiquetas: np.ndarray, semilla: int) -> tuple[dict[str, float], np.ndarray]:
    x_escalada = StandardScaler().fit_transform(x)
    componentes = min(50, x_escalada.shape[1], max(1, x_escalada.shape[0] - 1))
    entrada = (
        PCA(n_components=componentes, random_state=semilla).fit_transform(x_escalada)
        if x_escalada.shape[1] > componentes
        else x_escalada
    )
    coordenadas = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (len(x) - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=semilla,
        max_iter=1000,
    ).fit_transform(entrada)
    return (
        {
            "muestras": int(len(x)),
            "rasgos": int(x.shape[1]),
            "rasgos_entrada_tsne": int(entrada.shape[1]),
            "silhouette_features": float(silhouette_score(x_escalada, etiquetas)),
            "davies_bouldin_features": float(davies_bouldin_score(x_escalada, etiquetas)),
            "silhouette_tsne": float(silhouette_score(coordenadas, etiquetas)),
            "davies_bouldin_tsne": float(davies_bouldin_score(coordenadas, etiquetas)),
        },
        coordenadas,
    )


def softmin_errores(errores: np.ndarray, fuerza: float) -> np.ndarray:
    relativos = errores / np.maximum(errores.min(axis=1, keepdims=True), 1e-12)
    logits = -fuerza * (relativos - 1.0)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def diferencias_columnas(x: np.ndarray) -> np.ndarray:
    columnas = [x[:, a] - x[:, b] for a, b in combinations(range(x.shape[1]), 2)]
    return np.stack(columnas, axis=1)


def normalizar_nombre(texto: str) -> str:
    return "".join(caracter if caracter.isalnum() else "_" for caracter in texto).strip("_").lower()


def csv_rasgos(df_base: pd.DataFrame, x: np.ndarray, variante: str, ruta: Path) -> None:
    salida = df_base[["clase", "archivo", "ruta"]].copy()
    salida.insert(3, "variante_deep_onmf", variante)
    for indice in range(x.shape[1]):
        salida[f"F_{indice + 1:03d}"] = x[:, indice]
    salida.to_csv(ruta, index=False, encoding="utf-8-sig")


def cargar_w(ruta_csv: Path) -> dict[str, np.ndarray]:
    ruta_npz = ruta_csv.parent / "matrices_w_finales_por_clase.npz"
    if not ruta_npz.exists():
        raise FileNotFoundError(f"Falta {ruta_npz}")
    datos = np.load(ruta_npz)
    return {clase: datos[f"W_{clase}"] for clase in CLASES}


def proyectar_resultado(
    df: pd.DataFrame,
    configuracion: Configuracion,
    w_por_clase: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    mapa_registros = {str(registro.ruta): registro for registro in descubrir_audios(configuracion.carpeta_base_datos, CLASES)}
    errores_filas: list[list[float]] = []
    resumenes_h: list[np.ndarray] = []
    for posicion, fila in enumerate(df.itertuples(index=False), start=1):
        registro = mapa_registros[str(Path(fila.ruta))]
        matriz = construir_matriz_audio(registro, configuracion)
        errores_audio: list[float] = []
        h_real = None
        for clase in CLASES:
            h, error = proyectar_sobre_w(matriz, w_por_clase[clase])
            errores_audio.append(error)
            if clase == fila.clase:
                h_real = h
        if h_real is None:
            raise RuntimeError(f"No se ha calculado H real para {fila.archivo}")
        resumenes_h.append(
            np.concatenate(
                [
                    np.log1p(h_real.mean(axis=1)),
                    np.log1p(h_real.std(axis=1)),
                    np.log1p(np.quantile(h_real, 0.90, axis=1)),
                    np.log1p(h_real.max(axis=1)),
                ]
            )
        )
        errores_filas.append(errores_audio)
        if posicion == 1 or posicion % 100 == 0 or posicion == len(df):
            print(f"  Proyecciones {posicion}/{len(df)}")
    return np.asarray(errores_filas, dtype=float), np.asarray(resumenes_h, dtype=float)


def variantes_rasgos(df: pd.DataFrame, errores: np.ndarray, resumen_h: np.ndarray) -> dict[str, np.ndarray]:
    columnas_sbv = [columna for columna in df.columns if columna.startswith("SBV_")]
    sbv = df[columnas_sbv].to_numpy(dtype=float)
    afinidad = -np.log(np.maximum(errores, 1e-12))
    diferencias_afinidad = diferencias_columnas(afinidad)
    soft_4 = softmin_errores(errores, fuerza=4.0)
    soft_8 = softmin_errores(errores, fuerza=8.0)
    soft_12 = softmin_errores(errores, fuerza=12.0)
    soft_16 = softmin_errores(errores, fuerza=16.0)
    soft_24 = softmin_errores(errores, fuerza=24.0)
    return {
        "sbv_media_base": sbv,
        "sbv_media_log": np.log1p(np.maximum(sbv, 0.0)),
        "resumen_h_media_std_q90_max": resumen_h,
        "afinidad_errores_por_base": afinidad,
        "afinidad_y_diferencias_errores": np.concatenate([afinidad, diferencias_afinidad], axis=1),
        "perfil_softmin_errores_f4": soft_4,
        "perfil_softmin_errores_f8": soft_8,
        "perfil_softmin_errores_f12": soft_12,
        "perfil_softmin_errores_f16": soft_16,
        "perfil_softmin_errores_f24": soft_24,
        "afinidad_y_resumen_h": np.concatenate([afinidad, resumen_h], axis=1),
        "softmin_f8_y_resumen_h": np.concatenate([soft_8, resumen_h], axis=1),
    }


def localizar_resultados(patron: str, max_resultados: int) -> list[Path]:
    rutas = sorted(RAIZ_OBJETIVO.glob(patron), key=lambda ruta: ruta.stat().st_mtime)
    rutas = [ruta for ruta in rutas if (ruta.parent / "matrices_w_finales_por_clase.npz").exists()]
    return rutas[-max_resultados:] if max_resultados > 0 else rutas


def informe_barrido(resumen: pd.DataFrame, mejor: pd.Series) -> str:
    columnas = [
        "resultado_deep_onmf",
        "variante",
        "duracion_trama_s",
        "rangos_onmf",
        "iteraciones_onmf",
        "semilla_onmf",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    tabla = resumen[columnas].sort_values(
        ["silhouette_features", "silhouette_tsne", "davies_bouldin_features"],
        ascending=[False, False, True],
    )
    tabla_formateada = tabla.copy()
    for columna in [
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]:
        tabla_formateada[columna] = tabla_formateada[columna].map(lambda valor: f"{valor:.4f}")
    lineas_tabla = [
        "| " + " | ".join(tabla_formateada.columns) + " |",
        "| " + " | ".join("---" for _ in tabla_formateada.columns) + " |",
    ]
    lineas_tabla.extend("| " + " | ".join(str(valor) for valor in fila) + " |" for fila in tabla_formateada.itertuples(index=False, name=None))
    return "\n".join(
        [
            "# Barrido de rasgos Deep ONMF ajustados",
            "",
            "## Criterio",
            "",
            "Se comparan rasgos que salen de Deep ONMF sin escribir resultados a mano:",
            "",
            "- SBV medios de la primera prueba.",
            "- Estadisticos de las activaciones H de la base real de cada audio.",
            "- Errores de reconstruccion de cada audio frente a las cinco bases W de clase.",
            "- Perfiles de afinidad construidos desde esos errores.",
            "",
            "El candidato final se elige por `silhouette_features`: mide la separacion antes de t-SNE.",
            "Se conservan tambien Davies-Bouldin y las metricas t-SNE para revisar la foto.",
            "",
            "## Mejor candidato",
            "",
            f"- Resultado Deep ONMF: `{mejor['resultado_deep_onmf']}`.",
            f"- Variante de rasgos: `{mejor['variante']}`.",
            f"- Silhouette en rasgos: `{mejor['silhouette_features']:.4f}`.",
            f"- Davies-Bouldin en rasgos: `{mejor['davies_bouldin_features']:.4f}`.",
            f"- Silhouette t-SNE: `{mejor['silhouette_tsne']:.4f}`.",
            f"- Davies-Bouldin t-SNE: `{mejor['davies_bouldin_tsne']:.4f}`.",
            "",
            "## Tabla completa",
            "",
            *lineas_tabla,
            "",
            "## Nota metodologica",
            "",
            "Los perfiles por error usan las bases W aprendidas por clase. Esto sigue siendo Deep ONMF,",
            "pero no es el mismo vector reducido de 7 SBV de la primera prueba. En la carpeta ajustada",
            "se deja el codigo y el CSV de rasgos para que esa diferencia quede visible.",
        ]
    )


def main() -> int:
    args = parsear_argumentos()
    carpetas = crear_salida()
    rutas = localizar_resultados(args.patron, args.max_resultados)
    if not rutas:
        raise FileNotFoundError(f"No se han localizado resultados Deep ONMF con patron {args.patron!r}")

    filas_metricas: list[dict[str, object]] = []
    rutas_candidatos: dict[tuple[str, str], Path] = {}
    for ruta_csv in rutas:
        resultado = ruta_csv.parents[1].name
        parametros = leer_parametros(ruta_csv)
        configuracion = configuracion_desde_parametros(parametros)
        df = pd.read_csv(ruta_csv).sort_values(["clase", "archivo"]).reset_index(drop=True)
        if len(df) != 1000:
            print(f"Saltando {resultado}: tiene {len(df)} audios y se esperan 1000.")
            continue
        etiquetas = df["clase"].to_numpy()
        print(f"Resultado {resultado}: proyectando contra bases W")
        errores, resumen_h = proyectar_resultado(df, configuracion, cargar_w(ruta_csv))

        for variante, x in variantes_rasgos(df, errores, resumen_h).items():
            nombre_archivo = f"{normalizar_nombre(resultado)}__{normalizar_nombre(variante)}.csv"
            ruta_rasgos = carpetas["rasgos"] / nombre_archivo
            csv_rasgos(df, x, variante, ruta_rasgos)
            metricas, coordenadas = preparar_tsne(x, etiquetas, args.semilla_tsne)
            ruta_coords = carpetas["coordenadas"] / nombre_archivo.replace(".csv", "__tsne.csv")
            pd.DataFrame(
                {
                    "clase": df["clase"],
                    "archivo": df["archivo"],
                    "tSNE_1": coordenadas[:, 0],
                    "tSNE_2": coordenadas[:, 1],
                }
            ).to_csv(ruta_coords, index=False, encoding="utf-8-sig")
            rutas_candidatos[(resultado, variante)] = ruta_rasgos
            filas_metricas.append(
                {
                    "resultado_deep_onmf": resultado,
                    "variante": variante,
                    "duracion_trama_s": configuracion.duracion_trama_s,
                    "solape_trama_s": configuracion.solape_trama_s,
                    "rangos_onmf": "-".join(str(valor) for valor in configuracion.rangos_onmf),
                    "iteraciones_onmf": configuracion.iteraciones_onmf,
                    "penalizacion_ortogonal": configuracion.penalizacion_ortogonal,
                    "semilla_onmf": configuracion.semilla,
                    "ruta_csv_sbv": str(ruta_csv),
                    "ruta_rasgos": str(ruta_rasgos),
                    **metricas,
                }
            )
            print(
                f"  {variante}: sil_features={metricas['silhouette_features']:.4f} "
                f"DB_features={metricas['davies_bouldin_features']:.4f} "
                f"sil_tsne={metricas['silhouette_tsne']:.4f}"
            )

    resumen = pd.DataFrame(filas_metricas)
    if resumen.empty:
        raise RuntimeError("No se han podido calcular candidatos.")
    resumen = resumen.sort_values(
        ["silhouette_features", "davies_bouldin_features", "silhouette_tsne"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    mejor = resumen.iloc[0]
    ruta_mejor = Path(str(mejor["ruta_rasgos"]))
    (RAIZ_CARPETA / "resultados" / "mejores_rasgos_deep_onmf.csv").write_bytes(ruta_mejor.read_bytes())
    resumen.to_csv(RAIZ_CARPETA / "resultados" / "metricas_barrido_deep_onmf.csv", index=False, encoding="utf-8-sig")
    (RAIZ_CARPETA / "resultados" / "00_INFORME_BARRIDO_DEEP_ONMF.md").write_text(
        informe_barrido(resumen, mejor),
        encoding="utf-8",
    )
    (RAIZ_CARPETA / "resultados" / "mejor_candidato_deep_onmf.json").write_text(
        json.dumps(mejor.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Mejor candidato guardado: {ruta_mejor}")
    print(f"CSV final: {RAIZ_CARPETA / 'resultados' / 'mejores_rasgos_deep_onmf.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
