from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

KEY_FILES = [
    "informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf",
    "verificacion/MANIFIESTO_ARCHIVOS.csv",
    "resultados/07_comparacion_temporal_espectral_h3_w/prueba_nmf_w_h/tablas_csv/resumen_accuracy_nmf_w_h.csv",
    "resultados/07_comparacion_temporal_espectral_h3_w/prueba_nmf_w_h/tablas_csv/resultados_nmf_w_h.csv",
]


def long_path(path: Path | str) -> str:
    raw = os.path.abspath(str(path))
    if os.name != "nt":
        return raw
    if raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw[2:]
    return "\\\\?\\" + raw


def exists(path: Path) -> bool:
    return os.path.exists(long_path(path))


def size(path: Path) -> int:
    return os.path.getsize(long_path(path))


def count_csv_rows(path: Path) -> int:
    with open(long_path(path), "r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main() -> int:
    print("Resumen de resultados y evidencia")
    print("Repositorio:", ROOT)
    for rel in KEY_FILES:
        path = ROOT / rel
        if not exists(path):
            print(f"FALTA  {rel}")
            continue
        size_mb = size(path) / (1024 * 1024)
        extra = ""
        if path.suffix.lower() == ".csv":
            try:
                extra = f" | filas={count_csv_rows(path)}"
            except Exception as exc:  # noqa: BLE001
                extra = f" | no se pudo contar filas: {exc}"
        print(f"OK     {rel} | {size_mb:.2f} MB{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
