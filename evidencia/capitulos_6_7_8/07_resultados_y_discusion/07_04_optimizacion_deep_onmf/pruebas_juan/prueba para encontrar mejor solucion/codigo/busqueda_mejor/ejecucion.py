from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from codigo.configuracion import cargar_configuracion
from codigo.datos import descubrir_audios, limitar_por_clase
from codigo.metricas import comprobar_formulas_metricas
from pruebas_juan.auditoria import auditar_base, crear_y_auditar_folds

from .busqueda import (
    ejecutar_busqueda_adaptativa,
    importar_historicos,
    obtener_finalistas,
)
from .configuracion_busqueda import (
    ConfiguracionBusqueda,
    clave_carpeta,
    configurar_experimento,
    etiqueta,
    guardar_configuracion,
)
from .evaluacion import evaluar_configuracion
from .extraccion import comprobar_formas, extraer_representaciones
from .informe import generar_pdf
from .tabla_inicial_finalistas import generar_pdf_tabla_inicial_mas_cinco


RAIZ_NUEVA = Path(__file__).resolve().parents[2]
RAIZ_PRUEBAS = RAIZ_NUEVA.parent
RAIZ_IMPLEMENTACION = RAIZ_PRUEBAS.parent
CONFIGURACION_BASE = RAIZ_IMPLEMENTACION / "configuracion_experimento.json"
RESULTADOS_ORIGINALES = RAIZ_IMPLEMENTACION / "resultados_punto3_validacion"
RESULTADOS_TRES_PRUEBAS = RAIZ_PRUEBAS / "resultados"
PDF_ANTERIOR = (
    RESULTADOS_TRES_PRUEBAS
    / "RESULTADOS_3_PRUEBAS_JUAN_TABLA_COMPLETA.pdf"
)


def _metadatos_maestros() -> pd.DataFrame:
    ruta = (
        RESULTADOS_ORIGINALES
        / "representaciones"
        / "DeepONMF_W"
        / "metadatos.csv"
    )
    if not ruta.exists():
        raise FileNotFoundError(f"No se encuentran los metadatos originales: {ruta}")
    return pd.read_csv(ruta, encoding="utf-8-sig")


def _comprobar_orden(
    metadatos: pd.DataFrame,
    maestros: pd.DataFrame,
) -> None:
    claves = (
        metadatos["clase"].astype(str)
        + "/"
        + metadatos["archivo"].astype(str)
    ).tolist()
    claves_maestras = (
        maestros["clase"].astype(str)
        + "/"
        + maestros["archivo"].astype(str)
    ).tolist()
    if claves != claves_maestras:
        raise AssertionError(
            "El orden de los audios no coincide con el experimento original"
        )


def _actualizar_registro(
    ruta: Path,
    datos: dict[str, object],
) -> None:
    if ruta.exists():
        tabla = pd.read_csv(ruta, encoding="utf-8-sig")
        tabla = tabla[
            tabla["distribucion"].astype(str) != str(datos["distribucion"])
        ]
    else:
        tabla = pd.DataFrame()
    tabla = pd.concat([tabla, pd.DataFrame([datos])], ignore_index=True)
    tabla.sort_values(["ronda", "distribucion"]).to_csv(
        ruta,
        index=False,
        encoding="utf-8-sig",
    )


def _limpiar_matrices_no_finalistas(
    carpeta_resultados: Path,
    compacta: pd.DataFrame,
) -> None:
    finalistas = set(
        compacta.loc[compacta["finalista"], "distribucion"].astype(str)
    )
    for carpeta in (carpeta_resultados / "configuraciones").glob("*"):
        if not carpeta.is_dir():
            continue
        distribucion = carpeta.name.replace("_", "-")
        if distribucion in finalistas:
            continue
        ruta = (
            carpeta
            / "representaciones"
            / "representaciones_deep_onmf.npz"
        )
        ruta.unlink(missing_ok=True)


def _comprobar_cinco_finalistas(compacta: pd.DataFrame) -> None:
    if len(compacta) < 5:
        raise AssertionError("No hay suficientes configuraciones para cinco finalistas")
    if int(compacta["finalista"].sum()) != 5:
        raise AssertionError("La seleccion no contiene exactamente cinco finalistas")


def ejecutar(
    carpeta_datos: Path,
    carpeta_resultados: Path,
    rapido: bool = False,
    limite_por_clase: int = 0,
    solo_informe: bool = False,
) -> Path | None:
    inicio_total = time.perf_counter()
    comprobar_formulas_metricas()
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    config_busqueda = ConfiguracionBusqueda()

    if solo_informe:
        compacta, tablas = obtener_finalistas(
            carpeta_resultados,
            config_busqueda.numero_finalistas,
        )
        _comprobar_cinco_finalistas(compacta)
        ruta_principal = generar_pdf(
            PDF_ANTERIOR,
            carpeta_resultados
            / "RESULTADOS_BUSQUEDA_MEJOR_CONFIGURACION_DEEP_ONMF.pdf",
            compacta,
            tablas,
            carpeta_resultados,
        )
        generar_pdf_tabla_inicial_mas_cinco(
            carpeta_resultados
            / "RESULTADOS_TABLA_INICIAL_MAS_5_MEJORES.pdf",
            RESULTADOS_TRES_PRUEBAS / "tablas_csv",
            carpeta_resultados / "tablas_csv",
        )
        return ruta_principal

    config_base = cargar_configuracion(CONFIGURACION_BASE)
    registros = descubrir_audios(carpeta_datos.resolve())
    if rapido:
        limite_por_clase = limite_por_clase or 2
        registros = limitar_por_clase(registros, limite_por_clase)
    elif limite_por_clase:
        raise ValueError("--limite-por-clase solo puede usarse con --rapido")

    auditar_base(
        registros,
        config_base,
        carpeta_resultados,
        exigir_base_completa=not rapido,
    )
    maestros = _metadatos_maestros()
    if rapido:
        claves = set(
            registro.clase + "/" + registro.archivo for registro in registros
        )
        maestros = maestros[
            (
                maestros["clase"].astype(str)
                + "/"
                + maestros["archivo"].astype(str)
            ).isin(claves)
        ].reset_index(drop=True)
    if len(maestros) != len(registros):
        raise AssertionError("Los metadatos maestros no coinciden con los audios")

    folds = crear_y_auditar_folds(
        maestros,
        config_base,
        RESULTADOS_ORIGINALES / "particiones_5fold.csv",
        carpeta_resultados,
        exigir_protocolo_completo=not rapido,
    )

    if not rapido:
        importar_historicos(
            RESULTADOS_ORIGINALES,
            RESULTADOS_TRES_PRUEBAS,
            carpeta_resultados / "historicos_importados",
        )

    ruta_registro = carpeta_resultados / "registro_configuraciones.csv"

    def evaluar(rangos: tuple[int, ...], ronda: int) -> None:
        inicio = time.perf_counter()
        distribucion = etiqueta(rangos)
        carpeta = (
            carpeta_resultados
            / "configuraciones"
            / clave_carpeta(rangos)
        )
        config = configurar_experimento(config_base, rangos, rapido=rapido)
        guardar_configuracion(
            config,
            config_busqueda,
            carpeta / "configuracion_utilizada.json",
            rapido,
        )
        extraccion = extraer_representaciones(
            registros,
            config,
            carpeta / "representaciones",
            reutilizar=True,
        )
        _comprobar_orden(extraccion.metadatos, maestros)
        comprobar_formas(
            extraccion.matrices,
            numero_audios=len(registros),
            numero_capas=len(rangos),
            rango_final=rangos[-1],
            bins_frecuencia=config.bins_frecuencia,
            tramas_tiempo=config.tramas_stft_por_segmento,
        )
        evaluar_configuracion(
            extraccion.matrices,
            extraccion.metadatos,
            folds,
            config,
            carpeta,
            reutilizar=True,
        )
        segundos = time.perf_counter() - inicio
        marcador = {
            "distribucion": distribucion,
            "rangos": list(rangos),
            "numero_capas": len(rangos),
            "ronda": ronda,
            "segundos": segundos,
            "audios": len(registros),
            "folds": len(folds),
            "evaluaciones_esperadas": 2 * 3 * 2 * len(folds),
        }
        (carpeta / "CONFIGURACION_COMPLETADA.json").write_text(
            json.dumps(marcador, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _actualizar_registro(ruta_registro, marcador)

    compacta = ejecutar_busqueda_adaptativa(
        carpeta_resultados,
        config_busqueda,
        evaluar,
        rapido=rapido,
    )

    if rapido:
        (carpeta_resultados / "NO_ENTREGABLE_PRUEBA_RAPIDA.txt").write_text(
            "Prueba tecnica reducida. No representa los 1000 audios.\n",
            encoding="utf-8",
        )
        return None

    _comprobar_cinco_finalistas(compacta)
    compacta, tablas = obtener_finalistas(
        carpeta_resultados,
        config_busqueda.numero_finalistas,
    )
    _limpiar_matrices_no_finalistas(carpeta_resultados, compacta)
    ruta_pdf = generar_pdf(
        PDF_ANTERIOR,
        carpeta_resultados
        / "RESULTADOS_BUSQUEDA_MEJOR_CONFIGURACION_DEEP_ONMF.pdf",
        compacta,
        tablas,
        carpeta_resultados,
    )
    generar_pdf_tabla_inicial_mas_cinco(
        carpeta_resultados / "RESULTADOS_TABLA_INICIAL_MAS_5_MEJORES.pdf",
        RESULTADOS_TRES_PRUEBAS / "tablas_csv",
        carpeta_resultados / "tablas_csv",
    )
    (carpeta_resultados / "resumen_tiempo.txt").write_text(
        f"Tiempo total: {(time.perf_counter() - inicio_total) / 60.0:.2f} minutos\n",
        encoding="utf-8",
    )
    return ruta_pdf


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Busca la mejor arquitectura Deep-ONMF de dos a cinco capas."
        )
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=RAIZ_IMPLEMENTACION.parent / "Bases de Datos",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ_NUEVA / "resultados",
    )
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--limite-por-clase", type=int, default=0)
    parser.add_argument("--solo-informe", action="store_true")
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    salida = args.salida
    if args.rapido and salida == RAIZ_NUEVA / "resultados":
        salida = RAIZ_NUEVA / "resultados_verificacion"
    ruta = ejecutar(
        args.datos,
        salida,
        rapido=args.rapido,
        limite_por_clase=args.limite_por_clase,
        solo_informe=args.solo_informe,
    )
    if ruta is not None:
        print(f"[fin] PDF generado en {ruta}")
