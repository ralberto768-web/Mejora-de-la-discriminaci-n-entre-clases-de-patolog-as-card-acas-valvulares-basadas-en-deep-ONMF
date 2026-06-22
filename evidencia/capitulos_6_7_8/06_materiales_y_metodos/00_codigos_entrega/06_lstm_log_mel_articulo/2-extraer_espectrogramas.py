from __future__ import annotations

"""PASO 2 - Extraer caracteristicas log-mel.

Este script concatena con el paso 1:
1. Lee los WAV ya preparados por "1-preparar_datos.py".
2. Calcula el espectrograma log-mel con 40 bandas, tramas de 25 ms y salto de 10 ms.
3. Guarda un archivo NPZ que usaran los pasos 3 y 4 para entrenar CNN/LSTM.

Ejecucion manual:
    python 2-extraer_espectrogramas.py --duracion 2.0
"""

import argparse
from collections import Counter

import numpy as np

from codigo_comun.caracteristicas import espectrograma_log_mel
from codigo_comun.configuracion import crear_configuracion
from codigo_comun.consola import crear_registrador, titulo
from codigo_comun.datos import descubrir_audios, leer_wav_mono


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 2: extraer espectrogramas log-mel.")
    parser.add_argument("--duracion", type=float, default=2.0, help="Duracion usada en el paso 1.")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla registrada para trazabilidad.")
    return parser.parse_args()


def main() -> int:
    args = parsear_argumentos()
    cfg = crear_configuracion(duracion_segmento=args.duracion, semilla=args.semilla)
    cfg.carpeta_resultados.mkdir(parents=True, exist_ok=True)

    with crear_registrador(cfg.carpeta_resultados, f"2-extraer_espectrogramas_{cfg.etiqueta_duracion}s.txt") as registro:
        titulo(registro, "Paso 2 - Extraccion de espectrogramas log-mel")
        registro.escribir(f"Carpeta de audios preparados: {cfg.carpeta_datos_preparados}")
        registros = descubrir_audios(cfg.carpeta_datos_preparados, cfg.clases)
        if not registros:
            registro.escribir("No hay audios preparados. Ejecuta primero 1-preparar_datos.py.")
            return 1

        x = []
        y = []
        rutas = []
        mapa_clases = {clase: indice for indice, clase in enumerate(cfg.clases)}
        for indice, registro_audio in enumerate(registros, start=1):
            senal, fs = leer_wav_mono(registro_audio.ruta)
            if fs != cfg.fs_objetivo:
                registro.escribir(f"Advertencia: {registro_audio.ruta.name} tiene {fs} Hz; se esperaba {cfg.fs_objetivo} Hz.")
            x.append(espectrograma_log_mel(senal, cfg))
            y.append(mapa_clases[registro_audio.clase])
            rutas.append(str(registro_audio.ruta))
            if indice % 50 == 0 or indice == len(registros):
                registro.escribir(f"Extraidos {indice}/{len(registros)} espectrogramas...")

        x_np = np.stack(x).astype(np.float32)
        y_np = np.array(y, dtype=np.int64)
        np.savez_compressed(
            cfg.archivo_caracteristicas,
            x=x_np,
            y=y_np,
            clases=np.array(cfg.clases),
            rutas=np.array(rutas),
            fs_objetivo=cfg.fs_objetivo,
            duracion_segmento=cfg.duracion_segmento,
            duracion_trama=cfg.duracion_trama,
            salto_trama=cfg.salto_trama,
            bandas_mel=cfg.bandas_mel,
            longitud_fft=cfg.longitud_fft,
        )

        conteo = Counter(y_np.tolist())
        registro.escribir("")
        registro.escribir(f"Archivo de caracteristicas creado: {cfg.archivo_caracteristicas}")
        registro.escribir(f"Forma de X: {x_np.shape} = audios x bandas_mel x tramas")
        registro.escribir("Resumen por clase:")
        for clase, indice in mapa_clases.items():
            registro.escribir(f"- {clase}: {conteo[indice]} espectrogramas")
        registro.escribir("")
        registro.escribir("Paso siguiente: ejecutar 3-entrenar_cnn.py y/o 4-entrenar_lstm.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
