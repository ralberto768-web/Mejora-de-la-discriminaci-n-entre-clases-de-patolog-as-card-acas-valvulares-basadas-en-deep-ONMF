from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd


CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA / "codigo"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CARPETA / ".cache_matplotlib"))

from tfg_deep_onmf.audio import construir_matriz_audio, construir_matriz_clase, descubrir_audios
from tfg_deep_onmf.configuracion import Configuracion
from tfg_deep_onmf.onmf import proyectar_sobre_w


EPS = 1e-12
VARIANTES = ("nndsvd", "nndsvda", "nndsvdar")


@dataclass
class Capa:
    clase: str
    variante: str
    indice: int
    rango: int
    forma_entrada: tuple[int, int]
    forma_w: tuple[int, int]
    forma_h: tuple[int, int]
    error_relativo: float
    ortogonalidad_media: float
    segundos: float


@dataclass
class Resultado:
    w_final: np.ndarray
    h_final: np.ndarray
    capas: list[Capa]
    error_relativo_final: float


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-ONMF H mejorado con NNDSVD, NNDSVDa y NNDSVDar.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--iteraciones", type=int, default=120)
    parser.add_argument("--limite-por-clase", type=int, default=0)
    parser.add_argument("--rapido", action="store_true")
    return parser.parse_args()


def resolver_datos(ruta: Path | None) -> Path:
    if ruta is not None:
        datos = ruta.expanduser().resolve()
        if datos.exists():
            return datos
        raise FileNotFoundError(f"No existe la carpeta de datos indicada: {datos}")
    for base in [Path.cwd(), CARPETA, *CARPETA.parents]:
        datos = base / "Bases de Datos"
        if datos.exists():
            return datos.resolve()
    raise FileNotFoundError("No se encontro una carpeta 'Bases de Datos'. Usa --datos RUTA.")


def crear_configuracion(args: argparse.Namespace, datos: Path, salida: Path) -> Configuracion:
    config = Configuracion(
        raiz=CARPETA,
        semilla=args.semilla,
        iteraciones_onmf=5 if args.rapido else args.iteraciones,
        rellenar_audios_cortos=True,
    )
    object.__setattr__(config, "carpeta_base_datos", datos)
    object.__setattr__(config, "carpeta_resultados", salida)
    return config


def limitar_registros(registros: list, clases: tuple[str, ...], limite: int) -> list:
    if limite <= 0:
        return registros
    seleccionados = []
    conteos = {clase: 0 for clase in clases}
    for registro in sorted(registros, key=lambda r: (clases.index(r.clase), r.ruta.name)):
        if conteos[registro.clase] < limite:
            seleccionados.append(registro)
            conteos[registro.clase] += 1
    return seleccionados


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
    componentes = min(rango, min(x.shape))
    media = float(np.mean(x))
    u, s, vt = randomized_svd(x, n_components=componentes, random_state=semilla)
    w = np.zeros((x.shape[0], rango), dtype=np.float64)
    h = np.zeros((rango, x.shape[1]), dtype=np.float64)

    w[:, 0] = math.sqrt(max(s[0], EPS)) * np.abs(u[:, 0])
    h[0, :] = math.sqrt(max(s[0], EPS)) * np.abs(vt[0, :])
    for j in range(1, componentes):
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
        escala = math.sqrt(max(s[j] * sigma, EPS))
        w[:, j] = escala * uu
        h[j, :] = escala * vv

    if rango > componentes:
        rng = np.random.default_rng(semilla)
        w[:, componentes:] = media * rng.random((x.shape[0], rango - componentes)) / 100.0
        h[componentes:, :] = media * rng.random((rango - componentes, x.shape[1])) / 100.0

    if variante == "nndsvda":
        w[w <= EPS] = media
        h[h <= EPS] = media
    elif variante == "nndsvdar":
        rng = np.random.default_rng(semilla)
        w[w <= EPS] = media * rng.random(np.count_nonzero(w <= EPS)) / 100.0
        h[h <= EPS] = media * rng.random(np.count_nonzero(h <= EPS)) / 100.0
    elif variante != "nndsvd":
        raise ValueError(f"Variante no soportada: {variante}")

    return normalizar_columnas_w(np.maximum(w, EPS), np.maximum(h, EPS))


def factorizar_onmf(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
    variante: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.maximum(matriz, EPS).astype(np.float64, copy=False)
    w, h = nndsvd_inicializar(x, rango, variante, semilla)
    for _ in range(iteraciones):
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        w = np.maximum(w, EPS)
        h *= (w.T @ x + penalizacion_ortogonal * h) / (
            (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + EPS
        )
        h = np.maximum(h, EPS)
        w, h = normalizar_columnas_w(w, h)
    return w, h, error_relativo(x, w, h), ortogonalidad_media(h)


def deep_onmf_mejorado(matriz: np.ndarray, clase: str, variante: str, config: Configuracion) -> Resultado:
    entrada = np.maximum(matriz, EPS)
    matrices_w: list[np.ndarray] = []
    capas: list[Capa] = []
    for indice, rango in enumerate(config.rangos_onmf, start=1):
        inicio = time.perf_counter()
        w, h, error, ort = factorizar_onmf(
            entrada,
            rango=rango,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + indice * 1000,
            variante=variante,
        )
        capas.append(
            Capa(
                clase=clase,
                variante=variante,
                indice=indice,
                rango=rango,
                forma_entrada=entrada.shape,
                forma_w=w.shape,
                forma_h=h.shape,
                error_relativo=error,
                ortogonalidad_media=ort,
                segundos=time.perf_counter() - inicio,
            )
        )
        matrices_w.append(w)
        entrada = h

    w_final = matrices_w[0] @ matrices_w[1] @ matrices_w[2]
    escala = np.maximum(np.linalg.norm(w_final, axis=0), EPS)
    w_final = w_final / escala[None, :]
    h_final = entrada * escala[:, None]
    return Resultado(w_final=w_final, h_final=h_final, capas=capas, error_relativo_final=error_relativo(matriz, w_final, h_final))


def softmin_errores(errores: np.ndarray, fuerza: float = 8.0) -> np.ndarray:
    errores = np.asarray(errores, dtype=np.float64)
    escala = np.exp(-fuerza * (errores - np.min(errores)))
    return escala / np.maximum(np.sum(escala), EPS)


def rasgos_por_audio(registros: list, w_por_clase: dict[str, np.ndarray], config: Configuracion) -> pd.DataFrame:
    filas = []
    for registro in registros:
        matriz = construir_matriz_audio(registro, config)
        errores = []
        fila = {"clase": registro.clase, "archivo": registro.ruta.name, "ruta": str(registro.ruta)}
        for clase in config.clases:
            _h, error = proyectar_sobre_w(matriz, w_por_clase[clase], iteraciones=60)
            errores.append(error)
            fila[f"error_{clase}"] = error
        pesos = softmin_errores(np.array(errores), fuerza=8.0)
        for clase, peso in zip(config.clases, pesos):
            fila[f"F8_{clase}"] = float(peso)
        filas.append(fila)
    return pd.DataFrame(filas)


def guardar_capas(resultados: dict[str, Resultado], ruta: Path) -> None:
    filas = []
    for resultado in resultados.values():
        for capa in resultado.capas:
            filas.append(
                {
                    "clase": capa.clase,
                    "variante": capa.variante,
                    "capa": capa.indice,
                    "rango": capa.rango,
                    "entrada": f"{capa.forma_entrada[0]}x{capa.forma_entrada[1]}",
                    "W": f"{capa.forma_w[0]}x{capa.forma_w[1]}",
                    "H": f"{capa.forma_h[0]}x{capa.forma_h[1]}",
                    "error_relativo": capa.error_relativo,
                    "ortogonalidad_media": capa.ortogonalidad_media,
                    "segundos": capa.segundos,
                }
            )
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parsear_argumentos()
    datos = resolver_datos(args.datos)
    salida = (args.salida or (Path.cwd() / "resultados")).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)
    config = crear_configuracion(args, datos, salida)
    registros = descubrir_audios(config.carpeta_base_datos, config.clases)
    registros = limitar_registros(registros, config.clases, args.limite_por_clase or (2 if args.rapido else 0))
    if not registros:
        raise RuntimeError("No se encontraron WAV validos.")

    metricas = []
    for variante in VARIANTES:
        resultados: dict[str, Resultado] = {}
        w_por_clase = {}
        h_por_clase = {}
        for posicion, clase in enumerate(config.clases, start=1):
            datos_clase = construir_matriz_clase(clase, registros, config)
            resultado = deep_onmf_mejorado(datos_clase.matriz, clase, variante, config)
            resultados[clase] = resultado
            w_por_clase[clase] = resultado.w_final
            h_por_clase[clase] = resultado.h_final
            metricas.append(
                {
                    "variante": variante,
                    "clase": clase,
                    "audios_usados": len(datos_clase.audios_usados),
                    "forma_W": f"{resultado.w_final.shape[0]}x{resultado.w_final.shape[1]}",
                    "forma_H": f"{resultado.h_final.shape[0]}x{resultado.h_final.shape[1]}",
                    "error_relativo_final": resultado.error_relativo_final,
                }
            )

        np.savez(salida / f"matrices_W_H_{variante}.npz", **{f"W_{c}": w_por_clase[c] for c in config.clases}, **{f"H_{c}": h_por_clase[c] for c in config.clases})
        guardar_capas(resultados, salida / f"capas_{variante}.csv")
        rasgos = rasgos_por_audio(registros, w_por_clase, config)
        rasgos.to_csv(salida / f"rasgos_F8_softmin_{variante}.csv", index=False, encoding="utf-8-sig")

    metricas_df = pd.DataFrame(metricas)
    metricas_df.to_csv(salida / "metricas_variantes.csv", index=False, encoding="utf-8-sig")
    validacion = pd.DataFrame(
        [
            {"comprobacion": "variantes", "ok": set(metricas_df["variante"]) == set(VARIANTES)},
            {"comprobacion": "sin_nan", "ok": not metricas_df.isna().any().any()},
            {"comprobacion": "clases_por_variante", "ok": all(len(metricas_df[metricas_df["variante"] == v]) == len(config.clases) for v in VARIANTES)},
        ]
    )
    validacion.to_csv(salida / "validacion.csv", index=False, encoding="utf-8-sig")
    (salida / "parametros.json").write_text(json.dumps(config.como_diccionario(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ejecucion completada. Resultados guardados en: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
