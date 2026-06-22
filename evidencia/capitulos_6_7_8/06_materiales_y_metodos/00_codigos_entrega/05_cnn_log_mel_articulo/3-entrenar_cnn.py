from __future__ import annotations

"""PASO 3 - Entrenar CNN.

Este script concatena con el paso 2:
1. Lee el archivo NPZ de caracteristicas log-mel.
2. Divide los datos en entrenamiento/prueba por clase.
3. Entrena una red convolucional inspirada en MyNet.m.
4. Guarda el modelo y las metricas en texto.

Ejecucion manual:
    python 3-entrenar_cnn.py --duracion 2.0 --epocas 30
"""

import argparse

import numpy as np

from codigo_comun.configuracion import crear_configuracion
from codigo_comun.consola import crear_registrador, titulo
from codigo_comun.datos import dividir_estratificado
from codigo_comun.metricas import resumen_metricas
from codigo_comun.modelos import entrenar_modelo


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 3: entrenar modelo CNN.")
    parser.add_argument("--duracion", type=float, default=2.0)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--tamano-lote", type=int, default=64)
    parser.add_argument("--tasa-aprendizaje", type=float, default=5e-5)
    parser.add_argument("--proporcion-entrenamiento", type=float, default=0.70)
    return parser.parse_args()


def etiqueta_particion(proporcion_entrenamiento: float) -> str:
    entrenamiento = int(round(proporcion_entrenamiento * 100))
    prueba = 100 - entrenamiento
    return f"entrenamiento_{entrenamiento}_prueba_{prueba}"


def main() -> int:
    args = parsear_argumentos()
    cfg = crear_configuracion(duracion_segmento=args.duracion, semilla=args.semilla)
    etiqueta = etiqueta_particion(args.proporcion_entrenamiento)
    with crear_registrador(cfg.carpeta_resultados, f"3-entrenar_cnn_{cfg.etiqueta_duracion}s_{etiqueta}.txt") as registro:
        titulo(registro, "Paso 3 - Entrenamiento CNN")
        if not cfg.archivo_caracteristicas.exists():
            registro.escribir("No existe el archivo de caracteristicas. Ejecuta primero 2-extraer_espectrogramas.py.")
            return 1

        datos = np.load(cfg.archivo_caracteristicas, allow_pickle=True)
        x = datos["x"]
        y = datos["y"]
        clases = datos["clases"].tolist()
        ind_ent, ind_pru = dividir_estratificado(y, args.proporcion_entrenamiento, cfg.semilla)
        registro.escribir(f"Datos cargados: {x.shape[0]} muestras")
        registro.escribir(f"Entrenamiento: {len(ind_ent)} | Prueba: {len(ind_pru)}")
        registro.escribir(f"Particion usada: {etiqueta.replace('_', ' ')}")

        try:
            resultado = entrenar_modelo(
                "cnn",
                x,
                y,
                ind_ent,
                ind_pru,
                clases,
                epocas=args.epocas,
                tamano_lote=args.tamano_lote,
                tasa_aprendizaje=args.tasa_aprendizaje,
                semilla=cfg.semilla,
                ruta_modelo=cfg.carpeta_resultados / f"modelo_cnn_{cfg.etiqueta_duracion}s_{etiqueta}.pt",
            )
        except ModuleNotFoundError as exc:
            registro.escribir(str(exc))
            registro.escribir("Entrenamiento CNN omitido hasta instalar dependencias.")
            return 2

        resumen = resumen_metricas(resultado.reales, resultado.predicciones, clases)
        registro.escribir("")
        registro.escribir(resumen)
        registro.escribir(f"Perdida final de entrenamiento: {resultado.perdida_final:.6f}")
        archivo_metricas = cfg.carpeta_resultados / f"3-metricas_cnn_{cfg.etiqueta_duracion}s_{etiqueta}.txt"
        archivo_metricas.write_text(resumen + "\n", encoding="utf-8")
        registro.escribir(f"Metricas guardadas en: {archivo_metricas}")
        registro.escribir(f"Modelo guardado en: {cfg.carpeta_resultados / f'modelo_cnn_{cfg.etiqueta_duracion}s_{etiqueta}.pt'}")
        registro.escribir("")
        registro.escribir("Paso siguiente: ejecutar 4-entrenar_lstm.py o 5-evaluar_modelos.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
