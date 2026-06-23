from __future__ import annotations

"""PASO 5 - Resumen final.

Este script concatena con los pasos 3 y 4:
1. Reune los archivos de metricas generados por CNN y LSTM.
2. Crea un resumen final en texto en la carpeta resultados.
3. Muestra en terminal el mismo contenido para que la ejecucion sea clara.

Ejecucion manual:
    python 5-evaluar_modelos.py --duracion 2.0
"""

import argparse
import re
from datetime import datetime

import numpy as np

from codigo_comun.configuracion import crear_configuracion
from codigo_comun.consola import crear_registrador, titulo


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 5: crear resumen final.")
    parser.add_argument("--duracion", type=float, default=None)
    parser.add_argument("--duraciones", type=float, nargs="+", default=None)
    parser.add_argument("--proporciones-entrenamiento", type=float, nargs="+", default=None)
    parser.add_argument("--semilla", type=int, default=42)
    return parser.parse_args()


def leer_si_existe(ruta) -> str:
    if ruta.exists():
        return ruta.read_text(encoding="utf-8")
    return "No se ha generado todavia. Revisa si el entrenamiento se omitio por dependencias.\n"


def extraer_exactitud(texto_metricas: str) -> float | None:
    coincidencia = re.search(r"Exactitud global:\s*([0-9]+(?:\.[0-9]+)?)", texto_metricas)
    if not coincidencia:
        return None
    return float(coincidencia.group(1))


def etiqueta_particion(proporcion_entrenamiento: float) -> str:
    entrenamiento = int(round(proporcion_entrenamiento * 100))
    prueba = 100 - entrenamiento
    return f"entrenamiento_{entrenamiento}_prueba_{prueba}"


def etiqueta_particion_visible(proporcion_entrenamiento: float) -> str:
    entrenamiento = int(round(proporcion_entrenamiento * 100))
    prueba = 100 - entrenamiento
    return f"{entrenamiento}/{prueba}"


def proporciones_a_evaluar(args: argparse.Namespace) -> list[float]:
    if args.proporciones_entrenamiento:
        return args.proporciones_entrenamiento
    return [0.70, 0.75]


def duraciones_a_evaluar(args: argparse.Namespace) -> list[float]:
    if args.duraciones:
        return args.duraciones
    if args.duracion is not None:
        return [args.duracion]
    return [2.0, 1.5, 1.0]


def leer_metricas_modelo(cfg, modelo: str, proporcion_entrenamiento: float) -> str:
    etiqueta = etiqueta_particion(proporcion_entrenamiento)
    prefijo = "3-metricas_cnn" if modelo == "CNN" else "4-metricas_lstm"
    ruta = cfg.carpeta_resultados / f"{prefijo}_{cfg.etiqueta_duracion}s_{etiqueta}.txt"
    if ruta.exists():
        return ruta.read_text(encoding="utf-8")

    # Compatibilidad con resultados antiguos de 70/30 generados antes de incluir la etiqueta.
    if abs(proporcion_entrenamiento - 0.70) < 1e-9:
        ruta_antigua = cfg.carpeta_resultados / f"{prefijo}_{cfg.etiqueta_duracion}s.txt"
        if ruta_antigua.exists():
            return ruta_antigua.read_text(encoding="utf-8")
    return "No se ha generado todavia. Revisa si el entrenamiento se omitio por dependencias.\n"


def extraer_matriz_y_metricas(texto_metricas: str) -> tuple[list[str], list[list[int]], dict[str, tuple[float, float, float]]]:
    clases: list[str] = []
    matriz: list[list[int]] = []
    metricas: dict[str, tuple[float, float, float]] = {}
    lineas = texto_metricas.splitlines()

    for indice, linea in enumerate(lineas):
        if linea.strip().startswith("AS") and "MR" in linea and "MVP" in linea:
            clases = linea.split()
            for fila in lineas[indice + 1 : indice + 1 + len(clases)]:
                partes = fila.replace(":", " ").split()
                if len(partes) >= len(clases) + 1:
                    matriz.append([int(valor) for valor in partes[1 : 1 + len(clases)]])
            break

    patron = re.compile(
        r"-\s+(\w+):\s+precision=([0-9.]+),\s+sensibilidad=([0-9.]+),\s+f1=([0-9.]+)"
    )
    for linea in lineas:
        coincidencia = patron.search(linea)
        if coincidencia:
            clase = coincidencia.group(1)
            metricas[clase] = (
                float(coincidencia.group(2)),
                float(coincidencia.group(3)),
                float(coincidencia.group(4)),
            )
    return clases, matriz, metricas


def tabla_modelo(nombre: str, texto_metricas: str) -> list[str]:
    if "No se ha generado" in texto_metricas:
        return [
            f"MODELO {nombre}",
            "No se ha generado todavia.",
            "Motivo habitual: no se entreno este modelo o se ejecuto con --modelo solo-datos.",
        ]

    exactitud = extraer_exactitud(texto_metricas)
    clases, matriz, metricas = extraer_matriz_y_metricas(texto_metricas)
    if not clases or not matriz:
        return [f"MODELO {nombre}", texto_metricas.strip()]

    lineas = [
        f"MODELO {nombre}",
        f"Exactitud global: {exactitud:.4f}" if exactitud is not None else "Exactitud global: no disponible",
        "La parte izquierda es la matriz de confusion; a la derecha estan las metricas de esa misma clase.",
        "",
        "Clase real | " + " ".join(f"{clase:>5}" for clase in clases) + " || precision sensibilidad     f1",
        "-" * 74,
    ]
    for clase, fila in zip(clases, matriz):
        precision, sensibilidad, f1 = metricas.get(clase, (0.0, 0.0, 0.0))
        lineas.append(
            f"{clase:>10} | "
            + " ".join(f"{valor:5d}" for valor in fila)
            + f" || {precision:9.4f} {sensibilidad:12.4f} {f1:6.4f}"
        )

    if exactitud is not None:
        if exactitud >= 0.90:
            lineas.append("Lectura: resultado alto; el modelo acierta al menos el 90% de la prueba.")
        elif exactitud >= 0.75:
            lineas.append("Lectura: resultado aceptable; aprende patrones claros, pero aun hay errores relevantes.")
        else:
            lineas.append("Lectura: resultado bajo o medio; conviene revisar epocas, arquitectura o errores por clase.")
    return lineas


def describir_caracteristicas(cfg) -> list[str]:
    if not cfg.archivo_caracteristicas.exists():
        return [
            f"No existe el archivo de caracteristicas para {cfg.duracion_segmento} s.",
            "Ejecuta primero los pasos 1 y 2 para esta duracion.",
        ]

    datos = np.load(cfg.archivo_caracteristicas, allow_pickle=True)
    x = datos["x"]
    y = datos["y"]
    clases = datos["clases"].tolist()
    lineas = [
        f"Archivo de caracteristicas: {cfg.archivo_caracteristicas}",
        f"Forma de datos: {x.shape[0]} muestras x {x.shape[1]} bandas mel x {x.shape[2]} tramas temporales.",
        "Interpretacion: cada muestra es un audio cardiaco convertido a log-mel; las bandas mel resumen frecuencias y las tramas resumen el tiempo.",
        "Recuento por clase:",
    ]
    for indice, clase in enumerate(clases):
        lineas.append(f"- {clase}: {int(np.sum(y == indice))} muestras")
    return lineas


def explicar_modelo(nombre: str, texto_metricas: str) -> list[str]:
    if "No se ha generado" in texto_metricas:
        return [
            texto_metricas.strip(),
            "Interpretacion: no hay exactitud ni matriz de confusion porque el entrenamiento no se ejecuto.",
            "Causa habitual: falta instalar PyTorch o se ejecuto con --modelo solo-datos.",
        ]
    exactitud = extraer_exactitud(texto_metricas)
    explicacion_exactitud = "No se pudo extraer la exactitud numerica."
    if exactitud is not None:
        if exactitud >= 0.90:
            explicacion_exactitud = "Resultado alto: el modelo acierta al menos el 90% de las muestras de prueba."
        elif exactitud >= 0.75:
            explicacion_exactitud = "Resultado aceptable: el modelo aprende patrones claros, aunque todavia comete errores relevantes."
        else:
            explicacion_exactitud = "Resultado bajo o medio: conviene revisar entrenamiento, epocas, arquitectura o balance de errores por clase."
    return [
        texto_metricas.strip(),
        explicacion_exactitud,
        "Interpretacion: la exactitud global indica el porcentaje de audios de prueba clasificados correctamente.",
        "La matriz de confusion se lee por filas: cada fila es la clase real y cada columna es la clase predicha.",
        f"Si {nombre} tiene valores altos en la diagonal de la matriz, esta separando bien las clases.",
    ]


def main() -> int:
    args = parsear_argumentos()
    duraciones = duraciones_a_evaluar(args)
    proporciones = proporciones_a_evaluar(args)
    cfg = crear_configuracion(duracion_segmento=duraciones[0], semilla=args.semilla)
    cfg.carpeta_resultados.mkdir(parents=True, exist_ok=True)
    fecha_ejecucion = datetime.now()
    resultados_comparacion: list[tuple[str, float, float, float]] = []
    particiones_visibles = ", ".join(etiqueta_particion_visible(proporcion) for proporcion in proporciones)

    with crear_registrador(cfg.carpeta_resultados, "5-evaluar_modelos.txt") as registro:
        titulo(registro, "Paso 5 - Resumen final de la ejecucion")
        partes = [
            "RESUMEN FINAL DE CLASIFICACION DE SONIDOS CARDIACOS",
            f"Fecha: {fecha_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "QUE HACE ESTE EXPERIMENTO",
            "Se prueban tres duraciones de segmento del articulo: 2.0 s, 1.5 s y 1.0 s.",
            "Para cada duracion se preparan audios, se extraen espectrogramas log-mel y se entrenan CNN y LSTM.",
            f"Ademas se comparan estas particiones entrenamiento/prueba: {particiones_visibles}.",
            "Ejemplo: en 1000 audios, 75/25 equivale a 750 audios de entrenamiento y 250 de prueba.",
            "",
            "CLASES DEL PROBLEMA",
            "N = sonido normal.",
            "AS = estenosis aortica.",
            "MR = regurgitacion mitral.",
            "MS = estenosis mitral.",
            "MVP = prolapso de la valvula mitral.",
            "",
            "COMO LEER LOS RESULTADOS",
            "Exactitud global: proporcion de muestras de prueba acertadas.",
            "Precision: de las veces que el modelo predice una clase, cuantas son correctas.",
            "Sensibilidad: de las muestras reales de una clase, cuantas detecta el modelo.",
            "F1: equilibrio entre precision y sensibilidad.",
            "Matriz de confusion: filas = clase real, columnas = clase predicha.",
            "",
        ]

        for proporcion in proporciones:
            partes.extend(
                [
                    "",
                    "#" * 90,
                    f"PRUEBA CON PARTICION {etiqueta_particion_visible(proporcion)}",
                    "#" * 90,
                    "Interpretacion de la particion: el primer numero es el porcentaje usado para entrenar y el segundo para probar.",
                ]
            )

            for duracion in duraciones:
                cfg_duracion = crear_configuracion(duracion_segmento=duracion, semilla=args.semilla)
                metricas_cnn = leer_metricas_modelo(cfg_duracion, "CNN", proporcion)
                metricas_lstm = leer_metricas_modelo(cfg_duracion, "LSTM", proporcion)
                exactitud_cnn = extraer_exactitud(metricas_cnn)
                exactitud_lstm = extraer_exactitud(metricas_lstm)
                if exactitud_cnn is not None:
                    resultados_comparacion.append(("CNN", duracion, proporcion, exactitud_cnn))
                if exactitud_lstm is not None:
                    resultados_comparacion.append(("LSTM", duracion, proporcion, exactitud_lstm))
                partes.extend(
                    [
                        "",
                        "=" * 90,
                        f"SEGMENTOS DE {duracion} SEGUNDOS",
                        "=" * 90,
                        "",
                        "CARACTERISTICAS LOG-MEL",
                        *describir_caracteristicas(cfg_duracion),
                        "",
                        *tabla_modelo("CNN", metricas_cnn),
                        "",
                        *tabla_modelo("LSTM", metricas_lstm),
                    ]
                )

        partes.extend(["", "COMPARACION GLOBAL ENTRE DURACIONES Y MODELOS"])
        if resultados_comparacion:
            for modelo, duracion, proporcion, exactitud in sorted(resultados_comparacion, key=lambda item: (item[2], item[0], item[1])):
                partes.append(
                    f"- {modelo} | {duracion} s | particion {etiqueta_particion_visible(proporcion)}: exactitud global = {exactitud:.4f}"
                )
            mejor_modelo, mejor_duracion, mejor_proporcion, mejor_exactitud = max(resultados_comparacion, key=lambda item: item[3])
            partes.extend(
                [
                    "",
                    f"Mejor resultado de esta ejecucion: {mejor_modelo} con segmentos de {mejor_duracion} s y particion {etiqueta_particion_visible(mejor_proporcion)}, exactitud {mejor_exactitud:.4f}.",
                    "Interpretacion: esta combinacion es la que mejor separa las cinco clases en el conjunto de prueba generado.",
                    "Para comparar de forma justa con el articulo, revisa especialmente la CNN, porque en el documento original es el modelo que obtiene mayor rendimiento.",
                ]
            )
            mejores_por_modelo = {}
            for modelo, duracion, proporcion, exactitud in resultados_comparacion:
                if modelo not in mejores_por_modelo or exactitud > mejores_por_modelo[modelo][2]:
                    mejores_por_modelo[modelo] = (duracion, proporcion, exactitud)
            for modelo, (duracion, proporcion, exactitud) in sorted(mejores_por_modelo.items()):
                partes.append(
                    f"Mejor {modelo}: {duracion} s con particion {etiqueta_particion_visible(proporcion)} y exactitud {exactitud:.4f}."
                )
        else:
            partes.extend(
                [
                    "No hay metricas numericas disponibles para comparar.",
                    "Esto ocurre si todavia no se han entrenado CNN/LSTM o si se ejecuto el programa en modo solo-datos.",
                ]
            )

        partes.extend(
            [
                "",
                "ORDEN DE EJECUCION CONCATENADO",
                "0-INICIO.py -> 1-preparar_datos.py -> 2-extraer_espectrogramas.py -> 3-entrenar_cnn.py -> 4-entrenar_lstm.py -> 5-evaluar_modelos.py",
                "",
                "CONCLUSION AUTOMATICA",
                "El informe queda guardado con copia historica para no perder resultados entre ejecuciones.",
                "Si aparecen metricas CNN/LSTM, las dependencias de entrenamiento estan instaladas y se han podido comparar los modelos.",
                "Si en alguna ejecucion no aparecen metricas, revisa la instalacion de PyTorch o evita usar --modelo solo-datos.",
            ]
        )
        carpeta_historico = cfg.carpeta_resultados / "resultado final"
        carpeta_historico.mkdir(parents=True, exist_ok=True)

        resumen_completo = "\n".join(partes)
        marca_pruebas = "\n" + "#" * 90 + "\nPRUEBA CON PARTICION"
        inicio_pruebas = resumen_completo.find(marca_pruebas)
        inicio_comparacion = resumen_completo.find("\nCOMPARACION GLOBAL ENTRE DURACIONES Y MODELOS")
        cabecera = resumen_completo[:inicio_pruebas].rstrip()
        comparacion_global = resumen_completo[inicio_comparacion:].strip()
        archivos_generados = []

        for indice, proporcion in enumerate(proporciones):
            visible = etiqueta_particion_visible(proporcion)
            etiqueta = etiqueta_particion(proporcion)
            marcador_actual = f"\n{'#' * 90}\nPRUEBA CON PARTICION {visible}\n{'#' * 90}"
            inicio_actual = resumen_completo.find(marcador_actual)
            if inicio_actual < 0:
                continue
            if indice + 1 < len(proporciones):
                siguiente_visible = etiqueta_particion_visible(proporciones[indice + 1])
                marcador_siguiente = f"\n{'#' * 90}\nPRUEBA CON PARTICION {siguiente_visible}\n{'#' * 90}"
                fin_actual = resumen_completo.find(marcador_siguiente, inicio_actual + 1)
            else:
                fin_actual = inicio_comparacion
            seccion_prueba = resumen_completo[inicio_actual:fin_actual].strip()
            resumen_prueba = "\n\n".join([cabecera, seccion_prueba, comparacion_global])
            archivo_prueba = (
                carpeta_historico
                / f"resultado_final-{fecha_ejecucion.strftime('%Y-%m-%d_%H-%M-%S')}-{etiqueta}.txt"
            )
            archivo_prueba.write_text(resumen_prueba + "\n", encoding="utf-8")
            archivos_generados.append(archivo_prueba)

        registro.escribir(resumen_completo)
        registro.escribir("")
        registro.escribir("Resultados finales guardados en documentos separados:")
        for archivo in archivos_generados:
            registro.escribir(f"- {archivo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
