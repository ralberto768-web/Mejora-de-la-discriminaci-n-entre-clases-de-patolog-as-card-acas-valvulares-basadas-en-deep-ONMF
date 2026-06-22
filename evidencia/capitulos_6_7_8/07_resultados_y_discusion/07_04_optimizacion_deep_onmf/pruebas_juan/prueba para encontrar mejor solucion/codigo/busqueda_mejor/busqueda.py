from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .configuracion_busqueda import (
    ARQUITECTURAS_HISTORICAS,
    ARQUITECTURAS_INICIALES,
    CLASIFICADORES,
    ConfiguracionBusqueda,
    arquitectura_valida,
    clave_carpeta,
    etiqueta,
)


NOMBRES_METRICAS = (
    "metricas_binarias_por_fold.csv",
    "metricas_multiclase_por_fold.csv",
    "resumen_metricas_binarias.csv",
    "resumen_metricas_multiclase.csv",
)


def generar_vecinos(
    rangos: tuple[int, ...],
    config: ConfiguracionBusqueda,
    paso: int,
) -> set[tuple[int, ...]]:
    """Genera cambios locales de bases, profundidad y compresion."""

    candidatos: set[tuple[int, ...]] = set()
    for indice in range(len(rangos)):
        for delta in (-paso, paso):
            nuevo = list(rangos)
            nuevo[indice] += delta
            candidato = tuple(nuevo)
            if arquitectura_valida(candidato, config):
                candidatos.add(candidato)

    for delta in (-paso, paso):
        candidato = tuple(valor + delta for valor in rangos)
        if arquitectura_valida(candidato, config):
            candidatos.add(candidato)

    if len(rangos) < config.capas_maximas:
        for indice, (izquierda, derecha) in enumerate(
            zip(rangos, rangos[1:]),
            start=1,
        ):
            if izquierda - derecha > 1:
                centro = (izquierda + derecha) // 2
                candidato = rangos[:indice] + (centro,) + rangos[indice:]
                if arquitectura_valida(candidato, config):
                    candidatos.add(candidato)
        if rangos[0] + paso <= config.base_maxima:
            candidato = (rangos[0] + paso,) + rangos
            if arquitectura_valida(candidato, config):
                candidatos.add(candidato)
        if rangos[-1] - paso >= config.base_minima:
            candidato = rangos + (rangos[-1] - paso,)
            if arquitectura_valida(candidato, config):
                candidatos.add(candidato)

    if len(rangos) > config.capas_minimas:
        for indice in range(len(rangos)):
            candidato = rangos[:indice] + rangos[indice + 1 :]
            if arquitectura_valida(candidato, config):
                candidatos.add(candidato)

    candidatos.discard(rangos)
    return candidatos


def _normalizar_historico(
    tabla: pd.DataFrame,
    rangos: tuple[int, ...],
) -> pd.DataFrame:
    resultado = tabla.copy()
    if "distribucion" not in resultado:
        resultado.insert(0, "distribucion", etiqueta(rangos))
    else:
        resultado["distribucion"] = etiqueta(rangos)
    if "numero_capas" not in resultado:
        resultado.insert(1, "numero_capas", len(rangos))
    return resultado


def importar_historicos(
    resultados_originales: Path,
    resultados_tres_pruebas: Path,
    carpeta_destino: Path,
) -> None:
    """Copia metricas historicas sin modificar sus archivos de origen."""

    fuentes = {
        (9, 8, 7): resultados_originales / "metricas",
        (15, 10, 5): (
            resultados_tres_pruebas
            / "distribucion_15_10_5"
            / "metricas"
        ),
        (10, 6, 4): (
            resultados_tres_pruebas
            / "distribucion_10_6_4"
            / "metricas"
        ),
        (8, 5, 3): (
            resultados_tres_pruebas / "distribucion_8_5_3" / "metricas"
        ),
    }
    for rangos, fuente in fuentes.items():
        destino = carpeta_destino / clave_carpeta(rangos) / "metricas"
        destino.mkdir(parents=True, exist_ok=True)
        for nombre in NOMBRES_METRICAS:
            ruta = fuente / nombre
            if not ruta.exists():
                raise FileNotFoundError(f"Falta la metrica historica {ruta}")
            tabla = _normalizar_historico(
                pd.read_csv(ruta, encoding="utf-8-sig"),
                rangos,
            )
            # El bloque historico solo necesita W y H3.
            tabla = tabla[
                tabla["representacion"].isin(("DeepONMF_W", "DeepONMF_H3"))
            ]
            tabla.to_csv(
                destino / nombre,
                index=False,
                encoding="utf-8-sig",
            )
        (destino.parent / "origen.txt").write_text(
            str(fuente.resolve()) + "\n",
            encoding="utf-8",
        )
        if rangos == (9, 8, 7):
            auditoria = (
                resultados_originales
                / "representaciones"
                / "auditoria_deep_onmf.csv"
            )
        else:
            clave = clave_carpeta(rangos)
            auditoria = (
                resultados_tres_pruebas
                / f"distribucion_{clave}"
                / "representaciones"
                / "auditoria_deep_onmf.csv"
            )
        if auditoria.exists():
            destino_auditoria = destino.parent / "representaciones"
            destino_auditoria.mkdir(parents=True, exist_ok=True)
            pd.read_csv(auditoria, encoding="utf-8-sig").to_csv(
                destino_auditoria / "auditoria_deep_onmf.csv",
                index=False,
                encoding="utf-8-sig",
            )


def _carpetas_configuraciones(
    carpeta_resultados: Path,
) -> Iterable[tuple[tuple[int, ...], Path, str]]:
    historicos = carpeta_resultados / "historicos_importados"
    nuevas = carpeta_resultados / "configuraciones"
    for origen, tipo in ((historicos, "historica"), (nuevas, "nueva")):
        if not origen.exists():
            continue
        for carpeta in sorted(origen.glob("*")):
            if not carpeta.is_dir():
                continue
            try:
                rangos = tuple(int(valor) for valor in carpeta.name.split("_"))
            except ValueError:
                continue
            yield rangos, carpeta, tipo


def consolidar_metricas(
    carpeta_resultados: Path,
) -> dict[str, pd.DataFrame]:
    acumuladas: dict[str, list[pd.DataFrame]] = {
        nombre: [] for nombre in NOMBRES_METRICAS
    }
    for _, carpeta, _ in _carpetas_configuraciones(carpeta_resultados):
        for nombre in NOMBRES_METRICAS:
            ruta = carpeta / "metricas" / nombre
            if ruta.exists():
                acumuladas[nombre].append(
                    pd.read_csv(ruta, encoding="utf-8-sig")
                )

    tablas: dict[str, pd.DataFrame] = {}
    carpeta_tablas = carpeta_resultados / "tablas_csv"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)
    for nombre, partes in acumuladas.items():
        if partes:
            tabla = pd.concat(partes, ignore_index=True)
            tabla = tabla.drop_duplicates(
                ["distribucion", "clasificador", "representacion", "fold"]
                if "por_fold" in nombre
                else ["distribucion", "clasificador", "representacion"],
                keep="last",
            )
        else:
            tabla = pd.DataFrame()
        clave = nombre.removesuffix(".csv")
        tablas[clave] = tabla
        tabla.to_csv(
            carpeta_tablas / f"todas_{nombre}",
            index=False,
            encoding="utf-8-sig",
        )
    return tablas


def tabla_compacta_busqueda(
    resumen_binario: pd.DataFrame,
    numero_finalistas: int,
) -> pd.DataFrame:
    temporal = resumen_binario[
        resumen_binario["representacion"].str.startswith("DeepONMF_H")
    ].copy()
    filas: list[dict[str, object]] = []
    for distribucion, grupo in temporal.groupby("distribucion"):
        fila: dict[str, object] = {
            "distribucion": distribucion,
            "numero_capas": int(grupo["numero_capas"].iloc[0]),
        }
        for clasificador in CLASIFICADORES:
            datos = grupo[grupo["clasificador"] == clasificador]
            if len(datos) != 1:
                raise AssertionError(
                    f"{distribucion}: falta el resumen H de {clasificador}"
                )
            registro = datos.iloc[0]
            fila[f"Accuracy_{clasificador}"] = float(
                registro["Accuracy_mean"]
            )
            fila[f"Score_{clasificador}"] = float(registro["Score_mean"])
            fila[f"Accuracy_std_{clasificador}"] = float(
                registro["Accuracy_std"]
            )
        fila["Accuracy_media"] = float(
            np.mean([fila[f"Accuracy_{c}"] for c in CLASIFICADORES])
        )
        fila["Score_medio"] = float(
            np.mean([fila[f"Score_{c}"] for c in CLASIFICADORES])
        )
        fila["estabilidad_std"] = float(
            np.mean([fila[f"Accuracy_std_{c}"] for c in CLASIFICADORES])
        )
        filas.append(fila)

    tabla = pd.DataFrame(filas).sort_values(
        ["Accuracy_media", "Score_medio", "estabilidad_std", "distribucion"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    tabla.insert(0, "posicion", range(1, len(tabla) + 1))
    tabla["finalista"] = tabla["posicion"] <= numero_finalistas
    return tabla.reset_index(drop=True)


def obtener_finalistas(
    carpeta_resultados: Path,
    numero_finalistas: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    tablas = consolidar_metricas(carpeta_resultados)
    compacta = tabla_compacta_busqueda(
        tablas["resumen_metricas_binarias"],
        numero_finalistas,
    )
    finalistas = set(
        compacta.loc[compacta["finalista"], "distribucion"].astype(str)
    )
    tablas["tabla_compacta_busqueda"] = compacta
    tablas["finalistas_binarias_resumen"] = tablas[
        "resumen_metricas_binarias"
    ][
        tablas["resumen_metricas_binarias"]["distribucion"].isin(finalistas)
    ].copy()
    tablas["finalistas_multiclase_resumen"] = tablas[
        "resumen_metricas_multiclase"
    ][
        tablas["resumen_metricas_multiclase"]["distribucion"].isin(
            finalistas
        )
    ].copy()
    tablas["finalistas_binarias_por_fold"] = tablas[
        "metricas_binarias_por_fold"
    ][
        tablas["metricas_binarias_por_fold"]["distribucion"].isin(finalistas)
    ].copy()
    tablas["finalistas_multiclase_por_fold"] = tablas[
        "metricas_multiclase_por_fold"
    ][
        tablas["metricas_multiclase_por_fold"]["distribucion"].isin(
            finalistas
        )
    ].copy()
    carpeta_tablas = carpeta_resultados / "tablas_csv"
    for nombre, tabla in tablas.items():
        tabla.to_csv(
            carpeta_tablas / f"{nombre}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return compacta, tablas


def cargar_estado(ruta: Path) -> dict[str, object]:
    if not ruta.exists():
        return {
            "ronda": 0,
            "rondas_sin_mejora": 0,
            "mejor_accuracy": 0.0,
            "mejor_distribucion": "",
            "finalizada": False,
        }
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_estado(ruta: Path, estado: dict[str, object]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(
        json.dumps(estado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporal.replace(ruta)


def arquitecturas_evaluadas(carpeta_resultados: Path) -> set[tuple[int, ...]]:
    completas: set[tuple[int, ...]] = set()
    for rangos, carpeta, tipo in _carpetas_configuraciones(carpeta_resultados):
        if tipo == "historica":
            completas.add(rangos)
            continue
        if (carpeta / "CONFIGURACION_COMPLETADA.json").exists():
            completas.add(rangos)
            continue
        ruta_bin = carpeta / "metricas" / "metricas_binarias_por_fold.csv"
        ruta_multi = carpeta / "metricas" / "metricas_multiclase_por_fold.csv"
        if not ruta_bin.exists() or not ruta_multi.exists():
            continue
        try:
            binaria = pd.read_csv(ruta_bin, encoding="utf-8-sig")
            multi = pd.read_csv(ruta_multi, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            continue
        if len(binaria) == 30 and len(multi) == 30:
            completas.add(rangos)
    return completas


def ejecutar_busqueda_adaptativa(
    carpeta_resultados: Path,
    config: ConfiguracionBusqueda,
    evaluar: Callable[[tuple[int, ...], int], None],
    rapido: bool = False,
) -> pd.DataFrame:
    ruta_estado = carpeta_resultados / "estado_busqueda.json"
    estado = cargar_estado(ruta_estado)
    if bool(estado.get("finalizada")):
        compacta, _ = obtener_finalistas(
            carpeta_resultados,
            config.numero_finalistas,
        )
        return compacta

    evaluadas = arquitecturas_evaluadas(carpeta_resultados)
    iniciales = [
        arquitectura
        for arquitectura in ARQUITECTURAS_INICIALES
        if arquitectura not in evaluadas
    ]
    if rapido:
        iniciales = [(10, 9, 7, 5)]

    pendientes = iniciales
    while True:
        ronda = int(estado["ronda"]) + 1
        if not pendientes:
            compacta, _ = obtener_finalistas(
                carpeta_resultados,
                config.numero_finalistas,
            )
            padres = [
                tuple(int(v) for v in valor.split("-"))
                for valor in compacta.head(config.padres_por_ronda)[
                    "distribucion"
                ]
            ]
            candidatas: set[tuple[int, ...]] = set()
            for padre in padres:
                candidatas.update(
                    generar_vecinos(padre, config, paso=max(1, ronda - 1))
                )
            evaluadas = arquitecturas_evaluadas(carpeta_resultados)
            pendientes = sorted(candidatas - evaluadas)

        if not pendientes:
            estado["finalizada"] = True
            estado["motivo_finalizacion"] = (
                "No quedan vecinas nuevas alrededor de las mejores "
                "configuraciones."
            )
            guardar_estado(ruta_estado, estado)
            break

        print(
            f"[busqueda] Ronda {ronda}: {len(pendientes)} "
            "configuraciones completas"
        )
        for indice, arquitectura in enumerate(pendientes, start=1):
            print(
                f"[busqueda] Ronda {ronda}, {indice}/{len(pendientes)}: "
                f"{etiqueta(arquitectura)}"
            )
            evaluar(arquitectura, ronda)

        compacta, _ = obtener_finalistas(
            carpeta_resultados,
            config.numero_finalistas,
        )
        mejor = compacta.iloc[0]
        nueva_accuracy = float(mejor["Accuracy_media"])
        mejora = nueva_accuracy - float(estado["mejor_accuracy"])
        if mejora > config.mejora_minima:
            estado["rondas_sin_mejora"] = 0
        else:
            estado["rondas_sin_mejora"] = (
                int(estado["rondas_sin_mejora"]) + 1
            )
        estado.update(
            {
                "ronda": ronda,
                "mejor_accuracy": nueva_accuracy,
                "mejor_score": float(mejor["Score_medio"]),
                "mejor_distribucion": str(mejor["distribucion"]),
                "ultima_mejora": mejora,
                "configuraciones_evaluadas": len(
                    arquitecturas_evaluadas(carpeta_resultados)
                ),
            }
        )
        guardar_estado(ruta_estado, estado)
        if rapido or int(estado["rondas_sin_mejora"]) >= config.rondas_sin_mejora:
            estado["finalizada"] = True
            estado["motivo_finalizacion"] = (
                "Tres rondas consecutivas sin mejorar Accuracy."
                if not rapido
                else "Prueba rapida completada."
            )
            guardar_estado(ruta_estado, estado)
            break

        evaluadas = arquitecturas_evaluadas(carpeta_resultados)
        padres = [
            tuple(int(v) for v in valor.split("-"))
            for valor in compacta.head(config.padres_por_ronda)[
                "distribucion"
            ]
        ]
        candidatas = set()
        for padre in padres:
            candidatas.update(
                generar_vecinos(padre, config, paso=max(1, ronda))
            )
        pendientes = sorted(candidatas - evaluadas)

    compacta, _ = obtener_finalistas(
        carpeta_resultados,
        config.numero_finalistas,
    )
    return compacta
