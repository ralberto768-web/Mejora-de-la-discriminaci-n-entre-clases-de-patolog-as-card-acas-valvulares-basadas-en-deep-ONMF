from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12
FS_POR_DEFECTO = 8000.0
NOMBRES_CARACTERISTICAS = {
    "energia_rms": "energía RMS",
    "cruces_cero": "cruces por cero",
    "centroide_hz": "centroide espectral (Hz)",
    "relacion_soplo": "relación de soplo",
    "entropia_espectral": "entropía espectral",
    "irregularidad_temporal": "irregularidad temporal",
}
NOMBRES_CLASES = {
    "sana": "señal sana",
    "estenosis_aortica": "estenosis aórtica",
    "regurgitacion_mitral": "regurgitación mitral",
    "estenosis_mitral": "estenosis mitral",
    "prolapso_mitral": "prolapso mitral",
}


def leer_csv(ruta: Path, fs_por_defecto: float) -> tuple[np.ndarray, float]:
    """Lee un CSV con columnas tiempo_s,amplitud o una sola columna de amplitud."""
    with ruta.open("r", encoding="utf-8", newline="") as f:
        filas = [fila for fila in csv.reader(f) if fila and any(c.strip() for c in fila)]

    if not filas:
        raise ValueError(f"El fichero está vacío: {ruta}")

    primera = [c.strip().lower() for c in filas[0]]
    tiene_cabecera = any(not _es_numero(c) for c in primera)
    datos = filas[1:] if tiene_cabecera else filas

    indice_tiempo: int | None = None
    indice_amplitud = 0
    if tiene_cabecera:
        nombres_amplitud = {"amplitud", "amplitude", "senal", "señal", "signal", "valor", "value"}
        nombres_tiempo = {"tiempo", "tiempo_s", "time", "time_s", "t"}
        for i, nombre in enumerate(primera):
            if nombre in nombres_amplitud:
                indice_amplitud = i
            if nombre in nombres_tiempo:
                indice_tiempo = i
    elif len(datos[0]) >= 2:
        indice_tiempo = 0
        indice_amplitud = 1

    tiempos: list[float] = []
    amplitudes: list[float] = []
    for fila in datos:
        if len(fila) <= indice_amplitud:
            continue
        amplitudes.append(float(fila[indice_amplitud]))
        if indice_tiempo is not None and len(fila) > indice_tiempo:
            tiempos.append(float(fila[indice_tiempo]))

    senal = np.asarray(amplitudes, dtype=np.float64)
    if senal.size < 8:
        raise ValueError("La señal debe contener al menos 8 muestras.")

    fs = fs_por_defecto
    if len(tiempos) == len(amplitudes):
        diferencias = np.diff(np.asarray(tiempos, dtype=np.float64))
        diferencias = diferencias[diferencias > 0]
        if diferencias.size:
            fs = float(1.0 / np.median(diferencias))

    return senal, fs


def leer_wav(ruta: Path) -> tuple[np.ndarray, float]:
    """Lee ficheros WAV PCM mono o estéreo usando solo la librería estándar."""
    with wave.open(str(ruta), "rb") as wav:
        canales = wav.getnchannels()
        fs = float(wav.getframerate())
        bytes_muestra = wav.getsampwidth()
        bruto = wav.readframes(wav.getnframes())

    if bytes_muestra == 1:
        datos = np.frombuffer(bruto, dtype=np.uint8).astype(np.float64)
        datos = (datos - 128.0) / 128.0
    elif bytes_muestra == 2:
        datos = np.frombuffer(bruto, dtype=np.int16).astype(np.float64) / 32768.0
    elif bytes_muestra == 4:
        datos = np.frombuffer(bruto, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Formato WAV no soportado: {bytes_muestra} bytes por muestra")

    if canales > 1:
        datos = datos.reshape(-1, canales).mean(axis=1)
    return datos, fs


def leer_senal(ruta: Path, fs_por_defecto: float = FS_POR_DEFECTO) -> tuple[np.ndarray, float]:
    extension = ruta.suffix.lower()
    if extension == ".wav":
        return leer_wav(ruta)
    if extension in {".csv", ".txt"}:
        return leer_csv(ruta, fs_por_defecto)
    raise ValueError("Formato no soportado. Usa .csv, .txt o .wav.")


def preparar_senal(senal: np.ndarray) -> np.ndarray:
    senal = np.asarray(senal, dtype=np.float64).reshape(-1)
    senal = senal[np.isfinite(senal)]
    if senal.size < 8:
        raise ValueError("La señal no contiene suficientes muestras válidas.")
    senal = senal - float(np.mean(senal))
    maximo = float(np.max(np.abs(senal)))
    if maximo > EPS:
        senal = senal / maximo
    return senal


def extraer_caracteristicas(senal: np.ndarray, fs: float) -> dict[str, float]:
    """Calcula un conjunto pequeño de rasgos interpretables."""
    senal = preparar_senal(senal)
    frecuencias, potencia, energias_trama = _espectro_promedio(senal, fs)

    potencia_total = float(np.sum(potencia) + EPS)
    distribucion = potencia / potencia_total
    energia_rms = float(np.sqrt(np.mean(senal**2)))
    cruces_cero = float(np.mean(senal[:-1] * senal[1:] < 0.0))
    centroide_hz = float(np.sum(frecuencias * potencia) / potencia_total)
    banda_baja, banda_media, banda_alta = _bandas_analisis(fs)
    baja = _potencia_banda(frecuencias, potencia, *banda_baja)
    media = _potencia_banda(frecuencias, potencia, *banda_media)
    alta = _potencia_banda(frecuencias, potencia, *banda_alta)
    relacion_soplo = float((media + alta) / (baja + EPS))
    entropia = float(-np.sum(distribucion * np.log(distribucion + EPS)) / math.log(len(distribucion)))
    irregularidad = float(np.std(energias_trama) / (np.mean(energias_trama) + EPS))

    return {
        "energia_rms": energia_rms,
        "cruces_cero": cruces_cero,
        "centroide_hz": centroide_hz,
        "relacion_soplo": relacion_soplo,
        "entropia_espectral": entropia,
        "irregularidad_temporal": irregularidad,
    }


def clasificar(caracteristicas: dict[str, float], modelo: dict[str, Any]) -> dict[str, Any]:
    nombres = modelo["feature_names"]
    vector = np.asarray([caracteristicas[nombre] for nombre in nombres], dtype=np.float64)
    media = np.asarray(modelo["mean"], dtype=np.float64)
    escala = np.asarray(modelo["scale"], dtype=np.float64)
    vector_z = (vector - media) / np.maximum(escala, EPS)

    if "references" in modelo:
        return _clasificar_por_referencias(vector_z, modelo)

    distancias: dict[str, float] = {}
    for clase, prototipo in modelo["prototypes"].items():
        prototipo_z = np.asarray(prototipo, dtype=np.float64)
        distancias[clase] = float(np.linalg.norm(vector_z - prototipo_z))

    clase_predicha = min(distancias, key=distancias.get)
    pesos = {clase: math.exp(-distancia) for clase, distancia in distancias.items()}
    suma_pesos = sum(pesos.values()) + EPS
    confianza = float(pesos[clase_predicha] / suma_pesos)

    return {
        "clase": clase_predicha,
        "descripcion": modelo["class_descriptions"][clase_predicha],
        "confianza": confianza,
        "distancias": distancias,
    }


def _clasificar_por_referencias(vector_z: np.ndarray, modelo: dict[str, Any]) -> dict[str, Any]:
    referencias = modelo["references"]
    matriz = np.asarray([ref["features_z"] for ref in referencias], dtype=np.float64)
    distancias = np.linalg.norm(matriz - vector_z, axis=1)
    k = int(modelo.get("knn_k", 5))
    k = max(1, min(k, len(referencias)))
    indices = np.argsort(distancias)[:k]

    pesos_por_clase = {clase: 0.0 for clase in modelo["class_descriptions"]}
    distancia_minima_por_clase = {clase: float("inf") for clase in modelo["class_descriptions"]}
    vecinos = []

    for i, distancia in enumerate(distancias):
        clase = referencias[i]["class"]
        distancia_minima_por_clase[clase] = min(distancia_minima_por_clase[clase], float(distancia))

    for indice in indices:
        ref = referencias[int(indice)]
        distancia = float(distancias[int(indice)])
        peso = 1.0 / (distancia + EPS)
        pesos_por_clase[ref["class"]] += peso
        vecinos.append(
            {
                "archivo": ref["source_file"],
                "clase": ref["class"],
                "distancia": distancia,
            }
        )

    clase_predicha = max(pesos_por_clase, key=pesos_por_clase.get)
    suma_pesos = sum(pesos_por_clase.values()) + EPS
    confianza = float(pesos_por_clase[clase_predicha] / suma_pesos)

    return {
        "clase": clase_predicha,
        "descripcion": modelo["class_descriptions"][clase_predicha],
        "confianza": confianza,
        "distancias": distancia_minima_por_clase,
        "vecinos": vecinos,
        "tipo_modelo": "knn_ponderado",
    }


def analizar_archivo(ruta: Path, modelo: dict[str, Any], fs_por_defecto: float = FS_POR_DEFECTO) -> dict[str, Any]:
    senal, fs = leer_senal(ruta, fs_por_defecto)
    caracteristicas = extraer_caracteristicas(senal, fs)
    resultado = clasificar(caracteristicas, modelo)
    return {
        "archivo": str(ruta),
        "frecuencia_hz": fs,
        "muestras": int(len(senal)),
        "duracion_s": float(len(senal) / fs),
        "caracteristicas": caracteristicas,
        "resultado": resultado,
    }


def cargar_modelo(ruta: Path) -> dict[str, Any]:
    with ruta.open("r", encoding="utf-8") as f:
        return json.load(f)


def imprimir_resultado(analisis: dict[str, Any]) -> None:
    resultado = analisis["resultado"]
    print(f"Archivo analizado: {analisis['archivo']}")
    print(f"Frecuencia de muestreo: {analisis['frecuencia_hz']:.1f} Hz")
    print(f"Duración: {analisis['duracion_s']:.3f} s")
    print("")
    print("Características extraídas:")
    for nombre, valor in analisis["caracteristicas"].items():
        etiqueta = NOMBRES_CARACTERISTICAS.get(nombre, nombre)
        print(f"  - {etiqueta}: {valor:.6f}")
    print("")
    print(f"Clasificación: {resultado['descripcion']}")
    print(f"Confianza aproximada: {resultado['confianza']:.3f}")
    if resultado.get("tipo_modelo") == "knn_ponderado":
        print("Distancia al audio de referencia más cercano por clase:")
    else:
        print("Distancias al prototipo:")
    for clase, distancia in resultado["distancias"].items():
        etiqueta = NOMBRES_CLASES.get(clase, clase)
        print(f"  - {etiqueta}: {distancia:.6f}")
    print("")
    print("Aviso: esta demo es educativa y no sustituye al sistema experimental completo ni a un diagnóstico clínico.")


def _es_numero(texto: str) -> bool:
    try:
        float(texto)
        return True
    except ValueError:
        return False


def _espectro_promedio(senal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitud_ventana = 150
    salto = 75
    n_fft = 250
    if senal.size < longitud_ventana:
        senal = np.pad(senal, (0, longitud_ventana - senal.size))

    ventana = np.hamming(longitud_ventana)
    espectros: list[np.ndarray] = []
    energias: list[float] = []
    for inicio in range(0, senal.size - longitud_ventana + 1, salto):
        tramo = senal[inicio : inicio + longitud_ventana] * ventana
        espectro = np.abs(np.fft.rfft(tramo, n=n_fft)) ** 2
        espectros.append(espectro)
        energias.append(float(np.mean(tramo**2)))

    if not espectros:
        tramo = np.pad(senal, (0, max(0, longitud_ventana - senal.size)))[:longitud_ventana] * ventana
        espectros.append(np.abs(np.fft.rfft(tramo, n=n_fft)) ** 2)
        energias.append(float(np.mean(tramo**2)))

    potencia = np.mean(np.stack(espectros, axis=0), axis=0)
    frecuencias = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return frecuencias, potencia, np.asarray(energias, dtype=np.float64)


def _potencia_banda(frecuencias: np.ndarray, potencia: np.ndarray, f_min: float, f_max: float) -> float:
    mascara = (frecuencias >= f_min) & (frecuencias < f_max)
    if not np.any(mascara):
        return 0.0
    return float(np.sum(potencia[mascara]))


def _bandas_analisis(fs: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Usa bandas fijas a 8000 Hz y bandas relativas en demos de baja frecuencia."""
    nyquist = fs / 2.0
    if nyquist >= 1000.0:
        return (20.0, 150.0), (150.0, 400.0), (400.0, 1000.0)

    f_min = max(1.0, 0.02 * nyquist)
    baja_max = max(f_min + 1.0, 0.28 * nyquist)
    media_max = max(baja_max + 1.0, 0.65 * nyquist)
    alta_max = max(media_max + 1.0, 0.95 * nyquist)
    return (f_min, baja_max), (baja_max, media_max), (media_max, alta_max)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasifica una señal cardíaca de ejemplo como sana o patológica."
    )
    parser.add_argument("archivo", type=Path, help="Ruta del fichero .csv, .txt o .wav que se quiere clasificar.")
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path(__file__).resolve().parent / "modelo_basico.json",
        help="Ruta del modelo básico en formato JSON.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=FS_POR_DEFECTO,
        help="Frecuencia de muestreo usada si el CSV no contiene columna de tiempo.",
    )
    parser.add_argument("--json", action="store_true", help="Muestra el resultado en formato JSON.")
    args = parser.parse_args()

    modelo = cargar_modelo(args.modelo)
    analisis = analizar_archivo(args.archivo, modelo, fs_por_defecto=args.fs)
    if args.json:
        print(json.dumps(analisis, indent=2, ensure_ascii=False))
    else:
        imprimir_resultado(analisis)


if __name__ == "__main__":
    main()
