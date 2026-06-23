from __future__ import annotations

import argparse
from pathlib import Path
import time

import fitz
import pandas as pd

from codigo.configuracion import cargar_configuracion
from codigo.datos import descubrir_audios, limitar_por_clase
from pruebas_juan.auditoria import auditar_base, crear_y_auditar_folds

from .configuraciones import (
    CLASIFICADORES,
    REFERENCIA,
    clave,
    construir_ranking,
    convertir_etiqueta,
    etiqueta,
    es_creciente,
    es_decreciente,
    guardar_plan_arquitecturas,
    invertir,
    seleccionar_arquitecturas_profundas,
    seleccionar_diez,
)
from .evaluacion import ejecutar_configuracion
from .informes import generar_pdf_comparacion, generar_pdf_completo
from .matrices import generar_matrices_confusion


RAIZ_ULTIMA = Path(__file__).resolve().parents[2]
RAIZ_BUSQUEDA = RAIZ_ULTIMA.parent
RAIZ_PRUEBAS = RAIZ_BUSQUEDA.parent
RAIZ_IMPLEMENTACION = RAIZ_PRUEBAS.parent
RESULTADOS_ANTERIORES = RAIZ_BUSQUEDA / "resultados"
RESULTADOS_PUNTO3 = RAIZ_IMPLEMENTACION / "resultados_punto3_validacion"
RESULTADOS_TRES_PRUEBAS = RAIZ_PRUEBAS / "resultados"
CONFIGURACION_BASE = RAIZ_IMPLEMENTACION / "configuracion_experimento.json"


def _leer_resumen_anterior() -> pd.DataFrame:
    ruta = (
        RESULTADOS_ANTERIORES
        / "tablas_csv"
        / "todas_resumen_metricas_multiclase.csv"
    )
    tabla = pd.read_csv(ruta, encoding="utf-8-sig")
    if tabla["distribucion"].nunique() != 149 or len(tabla) != 894:
        raise AssertionError(
            "Los resultados anteriores no contienen 149 arquitecturas completas"
        )
    return tabla


def _leer_por_fold_anterior() -> pd.DataFrame:
    ruta = (
        RESULTADOS_ANTERIORES
        / "tablas_csv"
        / "todas_metricas_multiclase_por_fold.csv"
    )
    return pd.read_csv(ruta, encoding="utf-8-sig")


def _resumenes_nuevos(carpeta: Path) -> list[pd.DataFrame]:
    tablas = []
    if not carpeta.exists():
        return tablas
    for ruta in carpeta.glob("*/metricas/resumen_metricas_multiclase.csv"):
        tablas.append(pd.read_csv(ruta, encoding="utf-8-sig"))
    return tablas


def _por_fold_nuevos(carpeta: Path) -> list[pd.DataFrame]:
    tablas = []
    if not carpeta.exists():
        return tablas
    for ruta in carpeta.glob("*/metricas/metricas_multiclase_por_fold.csv"):
        tablas.append(pd.read_csv(ruta, encoding="utf-8-sig"))
    return tablas


def _consolidar_decrecientes(
    carpeta_resultados: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    anterior = _leer_resumen_anterior()
    por_fold_anterior = _leer_por_fold_anterior()
    carpeta_nuevas = carpeta_resultados / "dec"
    partes = [anterior, *_resumenes_nuevos(carpeta_nuevas)]
    partes_fold = [por_fold_anterior, *_por_fold_nuevos(carpeta_nuevas)]
    resumen = pd.concat(partes, ignore_index=True).drop_duplicates(
        ["distribucion", "clasificador", "representacion"],
        keep="last",
    )
    por_fold = pd.concat(partes_fold, ignore_index=True).drop_duplicates(
        ["distribucion", "clasificador", "representacion", "fold"],
        keep="last",
    )
    origenes = {
        valor: "anterior"
        for valor in anterior["distribucion"].astype(str).unique()
    }
    for distribucion, capas in resumen[
        ~resumen["distribucion"].astype(str).isin(origenes)
    ][["distribucion", "numero_capas"]].drop_duplicates().itertuples(
        index=False
    ):
        origenes[str(distribucion)] = f"nueva_{int(capas)}_capas"
    resumen["origen"] = resumen["distribucion"].map(origenes)
    por_fold["origen"] = por_fold["distribucion"].map(origenes)
    carpeta_tablas = carpeta_resultados / "tablas_csv"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(
        carpeta_tablas / "resumen_multiclase_todas_decrecientes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    por_fold.to_csv(
        carpeta_tablas / "metricas_multiclase_por_fold_decrecientes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return resumen, por_fold, origenes


def _consolidar_crecientes(carpeta_resultados: Path) -> pd.DataFrame:
    partes = _resumenes_nuevos(
        carpeta_resultados / "inc"
    )
    if not partes:
        return pd.DataFrame()
    tabla = pd.concat(partes, ignore_index=True).drop_duplicates(
        ["distribucion", "clasificador", "representacion"],
        keep="last",
    )
    tabla["origen"] = "invertida_creciente"
    tabla.to_csv(
        carpeta_resultados
        / "tablas_csv"
        / "resumen_multiclase_crecientes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    partes_fold = _por_fold_nuevos(
        carpeta_resultados / "inc"
    )
    pd.concat(partes_fold, ignore_index=True).drop_duplicates(
        ["distribucion", "clasificador", "representacion", "fold"],
        keep="last",
    ).to_csv(
        carpeta_resultados
        / "tablas_csv"
        / "metricas_multiclase_por_fold_crecientes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return tabla


def _metadatos_maestros() -> pd.DataFrame:
    ruta = (
        RESULTADOS_PUNTO3
        / "representaciones"
        / "DeepONMF_W"
        / "metadatos.csv"
    )
    return pd.read_csv(ruta, encoding="utf-8-sig")


def _resolver_predicciones(
    carpeta_resultados: Path,
    distribucion: str,
    clasificador: str,
    representacion: str,
) -> Path:
    rangos = convertir_etiqueta(distribucion)
    rutas = [
        carpeta_resultados
        / "dec"
        / clave(rangos)
        / "pred"
        / clasificador
        / ("W" if representacion == "DeepONMF_W" else representacion[9:]),
        carpeta_resultados
        / "inc"
        / clave(rangos)
        / "pred"
        / clasificador
        / ("W" if representacion == "DeepONMF_W" else representacion[9:]),
        RESULTADOS_ANTERIORES
        / "configuraciones"
        / clave(rangos)
        / "clasificadores"
        / clasificador
        / "multiclase"
        / representacion,
    ]
    if rangos == REFERENCIA:
        rutas.extend(
            [
                RESULTADOS_PUNTO3
                / "clasificadores"
                / clasificador
                / "multiclase"
                / representacion,
                RESULTADOS_PUNTO3
                / "clasificadores"
                / clasificador
                / representacion,
            ]
        )
    if rangos in ((15, 10, 5), (10, 6, 4), (8, 5, 3)):
        base_historica = (
            RESULTADOS_TRES_PRUEBAS
            / f"distribucion_{clave(rangos)}"
            / "clasificadores"
            / clasificador
        )
        rutas.extend(
            [
                base_historica / "multiclase" / representacion,
                base_historica / representacion,
            ]
        )
    for ruta in rutas:
        if (
            (ruta / "fold_1_predicciones.csv").exists()
            or (ruta / "f1_pred.csv").exists()
        ):
            return ruta
    raise FileNotFoundError(
        f"No se encuentran predicciones para {distribucion}, "
        f"{clasificador}, {representacion}"
    )


def _detalle_seleccionado(
    seleccion: pd.DataFrame,
    resumen_decreciente: pd.DataFrame,
    resumen_creciente: pd.DataFrame,
) -> pd.DataFrame:
    filas = []
    for _, pareja in seleccion.iterrows():
        for sentido, distribucion, fuente in (
            ("decreciente", pareja["distribucion"], resumen_decreciente),
            (
                "creciente",
                pareja["distribucion_invertida"],
                resumen_creciente,
            ),
        ):
            bloque = fuente[
                fuente["distribucion"].astype(str).eq(str(distribucion))
            ].copy()
            bloque["pareja"] = int(pareja["posicion_seleccion"])
            bloque["sentido"] = sentido
            bloque["tipo_matriz"] = bloque["representacion"].map(
                lambda valor: "W" if str(valor) == "DeepONMF_W" else "H"
            )
            filas.append(bloque)
    detalle = pd.concat(filas, ignore_index=True)
    if len(detalle) != 120:
        raise AssertionError(
            f"Se esperaban 120 resultados seleccionados, hay {len(detalle)}"
        )
    return detalle.sort_values(
        ["pareja", "clasificador", "tipo_matriz", "sentido"]
    ).reset_index(drop=True)


def _calcular_diferencias(detalle: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (pareja, clasificador, tipo), grupo in detalle.groupby(
        ["pareja", "clasificador", "tipo_matriz"]
    ):
        decreciente = grupo[grupo["sentido"].eq("decreciente")].iloc[0]
        creciente = grupo[grupo["sentido"].eq("creciente")].iloc[0]
        fila = {
            "pareja": int(pareja),
            "clasificador": clasificador,
            "tipo_matriz": tipo,
            "decreciente": decreciente["distribucion"],
            "creciente": creciente["distribucion"],
        }
        for metrica in (
            "Score",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
        ):
            fila[f"delta_{metrica}"] = float(
                creciente[f"{metrica}_mean"]
                - decreciente[f"{metrica}_mean"]
            )
        filas.append(fila)
    tabla = pd.DataFrame(filas).sort_values(
        ["pareja", "clasificador", "tipo_matriz"]
    )
    if len(tabla) != 60:
        raise AssertionError(f"Se esperaban 60 diferencias, hay {len(tabla)}")
    return tabla


def _conclusiones(diferencias: pd.DataFrame) -> list[str]:
    h = diferencias[diferencias["tipo_matriz"].eq("H")]
    lineas = []
    for clasificador in CLASIFICADORES:
        bloque = h[h["clasificador"].eq(clasificador)]
        media = float(bloque["delta_Accuracy"].mean())
        mejores = int(bloque["delta_Accuracy"].gt(0).sum())
        peores = int(bloque["delta_Accuracy"].lt(0).sum())
        lineas.append(
            f"{clasificador}: la arquitectura creciente mejora la Accuracy "
            f"en {mejores} parejas y empeora en {peores}; diferencia media "
            f"creciente menos decreciente = {media:+.4f}."
        )
    media_global = float(h["delta_Accuracy"].mean())
    sentido = "creciente" if media_global > 0 else "decreciente"
    lineas.append(
        f"En el conjunto de las treinta comparaciones de H final, el sentido "
        f"con mejor promedio es el {sentido}; diferencia global = "
        f"{media_global:+.4f}. Esta conclusion se limita a las arquitecturas, "
        "clasificadores y protocolo evaluados."
    )
    return lineas


def _auditar_pdf(
    ruta_completa: Path,
    ruta_comparacion: Path,
    ranking: pd.DataFrame,
    matrices: pd.DataFrame,
    carpeta_resultados: Path,
) -> None:
    completo = fitz.open(ruta_completa)
    comparacion = fitz.open(ruta_comparacion)
    texto_completo = "\n".join(pagina.get_text() for pagina in completo)
    texto_comparacion = "\n".join(
        pagina.get_text() for pagina in comparacion
    )
    columnas_orden = [
        "Accuracy",
        "Score",
        "Sensitivity",
        "Specificity",
        "Precision",
        "clasificador",
        "distribucion",
    ]
    ranking_ordenado = ranking.sort_values(
        columnas_orden,
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    auditoria = {
        "pdf_completo_existe": ruta_completa.exists(),
        "pdf_comparacion_existe": ruta_comparacion.exists(),
        "pdf_completo_horizontal": all(
            pagina.rect.width > pagina.rect.height for pagina in completo
        ),
        "pdf_comparacion_horizontal": all(
            pagina.rect.width > pagina.rect.height for pagina in comparacion
        ),
        "pdf_comparacion_once_paginas": len(comparacion) == 11,
        "pdf_completo_sin_matrices": "Mayor error" not in texto_completo,
        "pdf_completo_contiene_ranking": (
            "Ranking completo" in texto_completo
        ),
        "pdf_comparacion_contiene_veinte_matrices": sum(
            len(pagina.get_images(full=True)) for pagina in comparacion
        )
        == 20,
        "pdf_comparacion_solo_ujanet_h": (
            "SVM" not in texto_comparacion
            and "KNN" not in texto_comparacion
            and "DeepONMF_W" not in texto_comparacion
            and "UjaNet" in texto_comparacion
        ),
        "ranking_incluye_9_8_7": "9-8-7" in set(
            ranking["distribucion"].astype(str)
        ),
        "ranking_ordenado_por_accuracy_y_score": ranking.reset_index(
            drop=True
        ).equals(ranking_ordenado),
        "ranking_sin_medias_entre_clasificadores": not {
            "Accuracy_media",
            "Score_medio",
            "posicion_global",
        }.intersection(ranking.columns),
        "pdf_sin_medias_entre_clasificadores": (
            "Accuracy media" not in texto_completo
            and "Score medio" not in texto_completo
            and "promedio de los tres" not in texto_completo.lower()
        ),
        "veinte_matrices_ujanet": (
            len(matrices) == 20
            and set(matrices["clasificador"].astype(str)) == {"UjaNet"}
        ),
        "matrices_con_1000_predicciones": bool(
            matrices["predicciones"].eq(1000).all()
        ),
    }
    completo.close()
    comparacion.close()
    pd.DataFrame(
        [{"comprobacion": clave, "correcto": valor} for clave, valor in auditoria.items()]
    ).to_csv(
        carpeta_resultados / "auditoria_final.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not all(auditoria.values()):
        raise AssertionError(f"Auditoria final incorrecta: {auditoria}")


def _ejecutar_rapido(
    registros,
    maestros: pd.DataFrame,
    folds,
    config_base,
    carpeta_resultados: Path,
) -> None:
    for rangos, sentido in (
        ((15, 10, 5, 2), "decrecientes"),
        ((2, 5, 10, 15), "crecientes"),
    ):
        ejecutar_configuracion(
            rangos,
            registros,
            maestros,
            folds,
            config_base,
            carpeta_resultados / ("dec" if sentido == "decrecientes" else "inc"),
            rapido=True,
        )
    (carpeta_resultados / "NO_ENTREGABLE_PRUEBA_RAPIDA.txt").write_text(
        "Prueba reducida: no representa la validacion sobre 1000 audios.\n",
        encoding="utf-8",
    )


def ejecutar(
    carpeta_datos: Path,
    carpeta_resultados: Path,
    rapido: bool,
    limite_por_clase: int,
    solo_informe: bool,
) -> tuple[Path, Path] | None:
    inicio = time.perf_counter()
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    config_base = cargar_configuracion(CONFIGURACION_BASE)

    if not solo_informe:
        registros = descubrir_audios(carpeta_datos.resolve())
        if rapido:
            registros = limitar_por_clase(
                registros,
                limite_por_clase or 2,
            )
        auditar_base(
            registros,
            config_base,
            carpeta_resultados,
            exigir_base_completa=not rapido,
        )
        maestros = _metadatos_maestros()
        if rapido:
            claves = {
                registro.clase + "/" + registro.archivo
                for registro in registros
            }
            maestros = maestros[
                (
                    maestros["clase"].astype(str)
                    + "/"
                    + maestros["archivo"].astype(str)
                ).isin(claves)
            ].reset_index(drop=True)
        folds = crear_y_auditar_folds(
            maestros,
            config_base,
            RESULTADOS_PUNTO3 / "particiones_5fold.csv",
            carpeta_resultados,
            exigir_protocolo_completo=not rapido,
        )
        if rapido:
            _ejecutar_rapido(
                registros,
                maestros,
                folds,
                config_base,
                carpeta_resultados,
            )
            return None

        anterior = _leer_resumen_anterior()
        anteriores = {
            convertir_etiqueta(valor)
            for valor in anterior["distribucion"].astype(str).unique()
        }
        objetivos = seleccionar_arquitecturas_profundas(anterior, 50)
        plan = guardar_plan_arquitecturas(
            objetivos,
            anteriores,
            carpeta_resultados / "plan_arquitecturas_profundas.csv",
        )
        for fila in plan.itertuples(index=False):
            rangos = convertir_etiqueta(fila.distribucion)
            if rangos in anteriores:
                continue
            ejecutar_configuracion(
                rangos,
                registros,
                maestros,
                folds,
                config_base,
                carpeta_resultados / "dec",
                rapido=False,
            )

        resumen_decreciente, _, origenes = _consolidar_decrecientes(
            carpeta_resultados
        )
        ranking = construir_ranking(resumen_decreciente, origenes)
        seleccion = seleccionar_diez(ranking)
        seleccion.to_csv(
            carpeta_resultados / "tablas_csv" / "diez_principales.csv",
            index=False,
            encoding="utf-8-sig",
        )
        for valor in seleccion["distribucion"].astype(str):
            rangos = invertir(convertir_etiqueta(valor))
            if not es_creciente(rangos):
                raise AssertionError(f"Inversion no creciente: {rangos}")
            ejecutar_configuracion(
                rangos,
                registros,
                maestros,
                folds,
                config_base,
                carpeta_resultados / "inc",
                rapido=False,
            )

    resumen_decreciente, _, origenes = _consolidar_decrecientes(
        carpeta_resultados
    )
    ranking = construir_ranking(resumen_decreciente, origenes)
    seleccion = seleccionar_diez(ranking)
    seleccionadas = set(seleccion["distribucion"].astype(str))
    ranking["seleccionada"] = (
        ranking["distribucion"].astype(str).isin(seleccionadas)
        & ranking["mejor_resultado_arquitectura"]
    )
    ranking.to_csv(
        carpeta_resultados / "tablas_csv" / "ranking_completo.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seleccion.to_csv(
        carpeta_resultados / "tablas_csv" / "diez_principales.csv",
        index=False,
        encoding="utf-8-sig",
    )
    conteos = ranking[
        ranking["mejor_resultado_arquitectura"]
    ].groupby("numero_capas").agg(
        configuraciones=("distribucion", "nunique"),
        accuracy_maxima=("Accuracy", "max"),
        score_maximo=("Score", "max"),
    ).reset_index()
    conteos.to_csv(
        carpeta_resultados / "tablas_csv" / "resumen_por_profundidad.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detalle_completo = resumen_decreciente.copy()
    detalle_completo["origen"] = detalle_completo["distribucion"].map(
        origenes
    )
    detalle_completo = detalle_completo.sort_values(
        ["Accuracy_mean", "Score_mean"],
        ascending=False,
        kind="mergesort",
    )

    resumen_creciente = _consolidar_crecientes(carpeta_resultados)
    detalle = _detalle_seleccionado(
        seleccion,
        resumen_decreciente,
        resumen_creciente,
    )
    diferencias = _calcular_diferencias(detalle)
    detalle.to_csv(
        carpeta_resultados
        / "tablas_csv"
        / "resultados_principales_e_invertidas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    orden_sentido = pd.CategoricalDtype(
        ["decreciente", "creciente"],
        ordered=True,
    )
    tabla_ujanet_h = detalle[
        detalle["clasificador"].eq("UjaNet")
        & detalle["tipo_matriz"].eq("H")
    ].copy()
    tabla_ujanet_h["sentido"] = tabla_ujanet_h["sentido"].astype(
        orden_sentido
    )
    tabla_ujanet_h = tabla_ujanet_h.sort_values(
        ["pareja", "sentido"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(tabla_ujanet_h) != 20:
        raise AssertionError(
            f"Se esperaban 20 filas UjaNet-H, hay {len(tabla_ujanet_h)}"
        )
    tabla_ujanet_h.to_csv(
        carpeta_resultados
        / "tablas_csv"
        / "tabla_global_ujanet_h_20_filas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diferencias.to_csv(
        carpeta_resultados
        / "tablas_csv"
        / "diferencias_creciente_menos_decreciente.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matrices, _ = generar_matrices_confusion(
        seleccion,
        lambda distribucion, clasificador, representacion: _resolver_predicciones(
            carpeta_resultados,
            distribucion,
            clasificador,
            representacion,
        ),
        carpeta_resultados / "matrices_confusion_ujanet",
        clasificadores=("UjaNet",),
    )
    pdf_completo = generar_pdf_completo(
        carpeta_resultados
        / "RESULTADOS_COMPLETOS_MULTICLASE_ULTIMA_PRUEBA_JUAN.pdf",
        ranking,
        detalle_completo,
        conteos,
        seleccion,
    )
    pdf_comparacion = generar_pdf_comparacion(
        carpeta_resultados
        / "PRINCIPALES_INVERTIDAS_Y_MATRICES_CONFUSION.pdf",
        seleccion,
        detalle,
        diferencias,
        matrices,
        _conclusiones(diferencias),
    )
    _auditar_pdf(
        pdf_completo,
        pdf_comparacion,
        ranking,
        matrices,
        carpeta_resultados,
    )
    nombre_tiempo = (
        "resumen_regeneracion_documentos.txt"
        if solo_informe
        else "resumen_tiempo.txt"
    )
    (carpeta_resultados / nombre_tiempo).write_text(
        (
            "Tiempo de regeneracion de documentos: "
            if solo_informe
            else "Tiempo total: "
        )
        + f"{(time.perf_counter() - inicio) / 60.0:.2f} minutos\n",
        encoding="utf-8",
    )
    return pdf_completo, pdf_comparacion


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ampliacion multiclase e inversion de Deep-ONMF."
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=RAIZ_IMPLEMENTACION.parent / "Bases de Datos",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ_ULTIMA / "resultados",
    )
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--limite-por-clase", type=int, default=0)
    parser.add_argument("--solo-informe", action="store_true")
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    salida = args.salida
    if args.rapido and salida == RAIZ_ULTIMA / "resultados":
        salida = RAIZ_ULTIMA / "resultados_verificacion"
    resultado = ejecutar(
        args.datos,
        salida,
        args.rapido,
        args.limite_por_clase,
        args.solo_informe,
    )
    if resultado:
        print(f"[fin] PDF completo: {resultado[0]}")
        print(f"[fin] PDF comparacion: {resultado[1]}")
