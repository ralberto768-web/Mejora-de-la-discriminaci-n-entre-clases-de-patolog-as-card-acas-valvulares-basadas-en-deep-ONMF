from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "github" / "MANIFIESTO_REPOSITORIO.csv"

REQUIRED = [
    "README.md",
    "TITULO_TFG.md",
    "requirements.txt",
    "run_all.bat",
    "run_all.ps1",
    "docs/LECTURA_RAPIDA.md",
    "docs/GUIA_TRIBUNAL.md",
    "docs/GUIA_ESTRUCTURA_REPOSITORIO.md",
    "docs/REPRODUCIBILIDAD.md",
    "metodologia/README.md",
    "resultados/README.md",
    "verificacion/README.md",
    "verificacion/MANIFIESTO_ARCHIVOS.csv",
    "informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf",
    "informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.md",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(long_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_required() -> list[str]:
    errors = []
    for rel in REQUIRED:
        path = ROOT / rel
        if not exists(path):
            errors.append(f"Falta obligatorio: {rel}")
    return errors


def check_manifest(full_hash: bool) -> list[str]:
    errors = []
    if not exists(MANIFEST):
        return ["Falta manifiesto: github/MANIFIESTO_REPOSITORIO.csv"]
    with open(long_path(MANIFEST), "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row["ruta_relativa"]
            expected_size = int(row["tamano_bytes"])
            expected_hash = row.get("sha256", "")
            path = ROOT / rel
            if not exists(path):
                errors.append(f"Falta archivo manifestado: {rel}")
                continue
            actual_size = size(path)
            if actual_size != expected_size:
                errors.append(f"Tamano distinto: {rel} esperado={expected_size} actual={actual_size}")
                continue
            if full_hash and expected_hash and sha256(path) != expected_hash:
                errors.append(f"Hash distinto: {rel}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["rapido", "completo"], default="rapido")
    args = parser.parse_args()

    errors = []
    errors.extend(check_required())
    errors.extend(check_manifest(full_hash=args.modo == "completo"))

    if errors:
        print("VERIFICACION: ERROR")
        for err in errors[:100]:
            print("-", err)
        if len(errors) > 100:
            print(f"... {len(errors) - 100} errores mas")
        return 1

    print("VERIFICACION: OK")
    print("Modo:", args.modo)
    print("Repositorio:", ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
