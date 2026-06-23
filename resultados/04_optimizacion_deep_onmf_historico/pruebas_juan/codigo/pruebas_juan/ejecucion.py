from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from codigo.configuracion import cargar_configuracion
from codigo.datos import descubrir_audios, limitar_por_clase
from codigo.metricas import comprobar_formulas_metricas

from .auditoria import auditar_base, crear_y_auditar_folds
from .configuracion_pruebas import (
    DISTRIBUCIONES,
    configurar_rangos,
    guardar_configuracion_prueba,
)
from .evaluacion import evaluar_distribucion
from .extraccion_deep import comprobar_formas, extraer_w_h3
from .graficas import generar_graficas
from .informe_pdf import auditar_pdf, generar_pdf
from .tablas import consolidar_tablas


RAIZ_PRUEBAS = Path(__file__).resolve().parents[2]
RAIZ_IMPLEMENTACION = RAIZ_PRUEBAS.parent
RESULTADOS_ORIGINALES = RAIZ_IMPLEMENTACION / "resultados_punto3_validacion"
CONFIGURACION_BASE = RAIZ_IMPLEMENTACION / "configuracion_experimento.json"


def ejecutar_distribuciones(
    claves: list[str],
    carpeta_datos: Path,
    carpeta_resultados: Path,
    rapido: bool = False,
    limite_por_clase: int = 0,
    reutilizar: bool = True,
) -> None:
    inicio_total = time.perf_counter()
    comprobar_formulas_metricas()
    if rapido and limite_por_clase <= 0:
        limite_por_clase = 2
    if not rapido and limite_por_clase > 0:
        raise ValueError("El límite por clase solo puede usarse con --rapido")

    carpeta_datos = carpeta_datos.resolve()
    carpeta_resultados = carpeta_resultados.resolve()
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    config_base = cargar_configuracion(CONFIGURACION_BASE)
    registros = descubrir_audios(carpeta_datos)
    if limite_por_clase > 0:
        registros = limitar_por_clase(registros, limite_por_clase)
    auditar_base(
        registros,
        config_base,
        carpeta_resultados,
        exigir_base_completa=not rapido,
    )
    print(f"[datos] {len(registros)} audios detectados en {carpeta_datos}")

    for clave in claves:
        if clave not in DISTRIBUCIONES:
            raise ValueError(f"Distribución desconocida: {clave}")
        rangos = DISTRIBUCIONES[clave]
        config = configurar_rangos(config_base, rangos, rapido=rapido)
        carpeta_distribucion = carpeta_resultados / f"distribucion_{clave}"
        guardar_configuracion_prueba(
            config,
            carpeta_distribucion / "configuracion_utilizada.json",
            modo_rapido=rapido,
        )
        resultado = extraer_w_h3(
            registros,
            config,
            carpeta_distribucion / "representaciones",
            reutilizar=reutilizar,
        )
        comprobar_formas(
            resultado.matrices,
            numero_audios=len(registros),
            rango_final=rangos[-1],
            bins_frecuencia=config.bins_frecuencia,
            tramas_tiempo=config.tramas_stft_por_segmento,
        )
        folds = crear_y_auditar_folds(
            resultado.metadatos,
            config,
            RESULTADOS_ORIGINALES / "particiones_5fold.csv",
            carpeta_resultados,
            exigir_protocolo_completo=not rapido,
        )
        tabla_bin, tabla_multi, _ = evaluar_distribucion(
            resultado.matrices,
            resultado.metadatos,
            folds,
            config,
            carpeta_distribucion,
            reutilizar=reutilizar,
        )
        esperadas = 2 * 3 * len(folds)
        if len(tabla_bin) != esperadas or len(tabla_multi) != esperadas:
            raise AssertionError(
                f"{clave}: se esperaban {esperadas} filas por fold y se obtuvieron "
                f"{len(tabla_bin)} binarias y {len(tabla_multi)} multiclase"
            )
        print(f"[distribución {clave}] extracción y clasificación completadas")

    if not rapido and _estan_las_tres_distribuciones(carpeta_resultados):
        generar_entrega_completa(carpeta_resultados)
    elif rapido:
        (carpeta_resultados / "NO_ENTREGABLE_PRUEBA_RAPIDA.txt").write_text(
            "Esta carpeta contiene una verificación técnica reducida. "
            "No representa los resultados sobre 1000 audios.\n",
            encoding="utf-8",
        )
    else:
        print("[informe] Se generará el PDF cuando estén completas las tres distribuciones.")

    segundos = time.perf_counter() - inicio_total
    print(f"[fin] Tiempo total: {segundos / 60.0:.2f} minutos")


def _estan_las_tres_distribuciones(carpeta_resultados: Path) -> bool:
    for clave in DISTRIBUCIONES:
        carpeta_metricas = carpeta_resultados / f"distribucion_{clave}" / "metricas"
        for nombre in (
            "resumen_metricas_binarias.csv",
            "resumen_metricas_multiclase.csv",
            "metricas_binarias_por_fold.csv",
            "metricas_multiclase_por_fold.csv",
        ):
            if not (carpeta_metricas / nombre).exists():
                return False
    return True


def generar_entrega_completa(carpeta_resultados: Path) -> Path:
    tablas = consolidar_tablas(carpeta_resultados, RESULTADOS_ORIGINALES)
    comprobar_tablas_finales(tablas)
    graficas_binarias = generar_graficas(
        tablas["comparacion_distribuciones_binaria"],
        carpeta_resultados / "figuras",
        "binaria",
    )
    graficas_multiclase = generar_graficas(
        tablas["comparacion_distribuciones_multiclase"],
        carpeta_resultados / "figuras",
        "multiclase",
    )
    auditoria_base = pd.read_csv(
        carpeta_resultados / "auditoria_base_datos.csv",
        encoding="utf-8-sig",
    )
    protocolo = pd.read_csv(
        carpeta_resultados / "validacion_protocolo.csv",
        encoding="utf-8-sig",
    )
    ruta_pdf = carpeta_resultados / "RESULTADOS_3_PRUEBAS_JUAN_TABLA_COMPLETA.pdf"
    auditoria_filas = generar_pdf(
        ruta_pdf,
        tablas,
        graficas_binarias + graficas_multiclase,
        auditoria_base,
        protocolo,
        carpeta_resultados / "tablas_csv",
    )
    auditar_pdf(ruta_pdf, auditoria_filas, carpeta_resultados / "tablas_csv")
    print(f"[informe] PDF completo generado en {ruta_pdf}")
    return ruta_pdf


def comprobar_tablas_finales(tablas: dict[str, pd.DataFrame]) -> None:
    if len(tablas["tabla_global_binaria_completa_36_filas"]) != 36:
        raise AssertionError("La tabla binaria global no contiene las 36 filas esperadas")
    if len(tablas["tabla_global_multiclase_completa_36_filas"]) != 36:
        raise AssertionError("La tabla multiclase global no contiene las 36 filas esperadas")
    if len(tablas["tp_tn_fp_fn_por_fold_deeponmf"]) != 120:
        raise AssertionError("La tabla Deep-ONMF por fold no contiene las 120 filas esperadas")
    # La comprobación principal usa la tabla consolidada, por lo que funciona
    # también cuando el usuario elige una carpeta de salida distinta.
    if len(tablas["comparacion_distribuciones_binaria"]) != 24:
        raise AssertionError("La comparación binaria no contiene 24 combinaciones Deep-ONMF")


def copiar_configuraciones_estaticas() -> None:
    """Mantiene copias legibles de las tres configuraciones junto al código."""

    config_base = cargar_configuracion(CONFIGURACION_BASE)
    carpeta = RAIZ_PRUEBAS / "configuraciones"
    for clave, rangos in DISTRIBUCIONES.items():
        guardar_configuracion_prueba(
            configurar_rangos(config_base, rangos, rapido=False),
            carpeta / f"configuracion_{clave}.json",
            modo_rapido=False,
        )


def construir_parser(descripcion: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument(
        "--datos",
        type=Path,
        default=RAIZ_IMPLEMENTACION.parent / "Bases de Datos",
        help="Carpeta que contiene los 1000 WAV de Yaseen.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ_PRUEBAS / "resultados",
        help="Carpeta de salida.",
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Ejecuta una comprobación reducida no entregable.",
    )
    parser.add_argument(
        "--limite-por-clase",
        type=int,
        default=0,
        help="Número de audios por clase, solo permitido con --rapido.",
    )
    parser.add_argument(
        "--recalcular",
        action="store_true",
        help="Ignora cachés y resultados parciales existentes.",
    )
    return parser


def main_conjunto() -> None:
    copiar_configuraciones_estaticas()
    parser = construir_parser("Ejecuta las tres distribuciones solicitadas por Juan.")
    args = parser.parse_args()
    salida = args.salida
    if args.rapido and salida == RAIZ_PRUEBAS / "resultados":
        salida = RAIZ_PRUEBAS / "resultados_verificacion"
    ejecutar_distribuciones(
        list(DISTRIBUCIONES),
        args.datos,
        salida,
        rapido=args.rapido,
        limite_por_clase=args.limite_por_clase,
        reutilizar=not args.recalcular,
    )


def main_distribucion(clave: str) -> None:
    copiar_configuraciones_estaticas()
    parser = construir_parser(f"Ejecuta únicamente la distribución {clave.replace('_', '-')}.")
    args = parser.parse_args()
    salida = args.salida
    if args.rapido and salida == RAIZ_PRUEBAS / "resultados":
        salida = RAIZ_PRUEBAS / "resultados_verificacion"
    ejecutar_distribuciones(
        [clave],
        args.datos,
        salida,
        rapido=args.rapido,
        limite_por_clase=args.limite_por_clase,
        reutilizar=not args.recalcular,
    )
