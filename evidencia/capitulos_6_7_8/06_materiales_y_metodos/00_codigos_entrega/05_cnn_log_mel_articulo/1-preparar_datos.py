from __future__ import annotations

"""PASO 1 - Preparar datos.

Este script es el primer eslabon real de la cadena:
1. Lee los WAV originales desde la carpeta "Bases de Datos".
2. Detecta su clase usando el nombre de la carpeta o del archivo.
3. Convierte cada audio a mono, 8000 Hz y una duracion fija.
4. Guarda los segmentos preparados para que el paso 2 pueda extraer log-mel.

Ejecucion manual:
    python 1-preparar_datos.py --duracion 2.0
"""

import argparse
import shutil
from collections import Counter

from codigo_comun.configuracion import crear_configuracion
from codigo_comun.consola import crear_registrador, titulo
from codigo_comun.datos import (
    ajustar_a_duracion,
    descubrir_audios,
    guardar_metadatos_csv,
    guardar_wav_mono,
    leer_wav_mono,
    remuestrear_lineal,
)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 1: preparar audios cardiacos.")
    parser.add_argument("--duracion", type=float, default=2.0, help="Duracion fija de cada segmento en segundos.")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla registrada para trazabilidad.")
    parser.add_argument("--limite-por-clase", type=int, default=0, help="Limite opcional para pruebas rapidas; 0 usa todo.")
    return parser.parse_args()


def main() -> int:
    args = parsear_argumentos()
    cfg = crear_configuracion(duracion_segmento=args.duracion, semilla=args.semilla)
    cfg.carpeta_resultados.mkdir(parents=True, exist_ok=True)

    with crear_registrador(cfg.carpeta_resultados, f"1-preparar_datos_{cfg.etiqueta_duracion}s.txt") as registro:
        titulo(registro, "Paso 1 - Preparacion de audios")
        registro.escribir(f"Carpeta origen: {cfg.carpeta_bases_datos}")
        registro.escribir(f"Carpeta destino: {cfg.carpeta_datos_preparados}")
        registro.escribir(f"Frecuencia objetivo: {cfg.fs_objetivo} Hz")
        registro.escribir(f"Duracion por segmento: {cfg.duracion_segmento} s")

        registros = descubrir_audios(cfg.carpeta_bases_datos, cfg.clases)
        if args.limite_por_clase > 0:
            contador = Counter()
            filtrados = []
            for registro_audio in registros:
                if contador[registro_audio.clase] < args.limite_por_clase:
                    filtrados.append(registro_audio)
                    contador[registro_audio.clase] += 1
            registros = filtrados

        if not registros:
            registro.escribir("No se han encontrado audios WAV validos. Revisa la carpeta Bases de Datos.")
            return 1

        if cfg.carpeta_datos_preparados.exists():
            registro.escribir("Limpiando datos preparados anteriores para evitar duplicados.")
            shutil.rmtree(cfg.carpeta_datos_preparados)

        filas_metadatos: list[dict[str, str]] = []
        conteo = Counter()
        for indice, registro_audio in enumerate(registros, start=1):
            senal, fs_original = leer_wav_mono(registro_audio.ruta)
            senal = remuestrear_lineal(senal, fs_original, cfg.fs_objetivo)
            senal = ajustar_a_duracion(senal, cfg.fs_objetivo, cfg.duracion_segmento)

            nombre_salida = f"{registro_audio.clase}-{indice:04d}.wav"
            ruta_salida = cfg.carpeta_datos_preparados / registro_audio.clase / nombre_salida
            guardar_wav_mono(ruta_salida, senal, cfg.fs_objetivo)

            filas_metadatos.append(
                {
                    "clase": registro_audio.clase,
                    "archivo_origen": str(registro_audio.ruta),
                    "archivo_preparado": str(ruta_salida),
                }
            )
            conteo[registro_audio.clase] += 1
            if indice % 50 == 0 or indice == len(registros):
                registro.escribir(f"Preparados {indice}/{len(registros)} audios...")

        guardar_metadatos_csv(cfg.carpeta_resultados / f"1-metadatos_datos_preparados_{cfg.etiqueta_duracion}s.csv", filas_metadatos)
        registro.escribir("")
        registro.escribir("Resumen por clase:")
        for clase in cfg.clases:
            registro.escribir(f"- {clase}: {conteo[clase]} archivos preparados")
        registro.escribir("")
        registro.escribir("Paso siguiente: ejecutar 2-extraer_espectrogramas.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
