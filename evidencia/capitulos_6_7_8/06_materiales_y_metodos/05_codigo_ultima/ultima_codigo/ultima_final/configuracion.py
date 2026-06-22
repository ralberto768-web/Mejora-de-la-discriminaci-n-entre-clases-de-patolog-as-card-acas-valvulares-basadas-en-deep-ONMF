from __future__ import annotations

from pathlib import Path


CLASES = ("N", "AS", "MR", "MS", "MVP")
SNR_OBJETIVOS = (8.0, 5.0, 0.0, -5.0, -8.0)
BASE_ORIGINAL = "original"
BASE_SNR0 = "SNR_0db"

REPRESENTACIONES_CLASICAS = ("STFT", "MFCC", "Mel", "LogMel")
METRICAS = ("Accuracy", "Sensitivity", "Specificity", "Precision", "Score")

DOCUMENTOS_FINALES = {
    "optimizacion_original": "optimizacion_dataset_original.pdf",
    "optimizacion_snr0": "optimizacion_dataset_SNR0db.pdf",
    "resultados_original": "Resultados_Optimizacion_original.pdf",
    "resultados_snr0": "Resultados_Optimizacion_SNR0db.pdf",
}


def nombre_base_snr(snr_db: float) -> str:
    entero = int(snr_db)
    return f"SNR_{entero}db"


def carpetas_obligatorias(raiz: Path) -> list[Path]:
    return [
        raiz / "01_optimizacion_dataset_original",
        raiz / "02_optimizacion_dataset_SNR0db",
        raiz / "03_resultados_optimizacion_original",
        raiz / "04_resultados_optimizacion_SNR0db",
        raiz / "datasets_ruidosos",
        raiz / "documentos_finales",
        raiz / "codigo",
        raiz / "auditoria",
    ]
