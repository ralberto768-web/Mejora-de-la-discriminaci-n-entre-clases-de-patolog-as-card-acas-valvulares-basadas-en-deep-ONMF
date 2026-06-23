from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "github" / "MANIFIESTO_REPOSITORIO.csv"
LFS = ROOT / "github" / "ARCHIVOS_GRANDES_GIT_LFS.csv"
HASH_LIMIT = 0
LFS_LIMIT = 50 * 1024 * 1024

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
SKIP_FILES = {
    "github/MANIFIESTO_REPOSITORIO.csv",
    "github/ARCHIVOS_GRANDES_GIT_LFS.csv",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def long_path(path: Path | str) -> str:
    raw = os.path.abspath(str(path))
    if os.name != "nt":
        return raw
    if raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw[2:]
    return "\\\\?\\" + raw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(long_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files() -> list[Path]:
    files: list[Path] = []
    for current, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        current_path = Path(current)
        for name in names:
            path = current_path / name
            rel_path = rel(path)
            if rel_path in SKIP_FILES:
                continue
            files.append(path)
    return sorted(files, key=rel)


def main() -> int:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    files = iter_files()

    with open(MANIFEST, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ruta_relativa", "tamano_bytes", "sha256", "estado"])
        writer.writeheader()
        for path in files:
            size = os.path.getsize(long_path(path))
            file_hash = sha256(path) if size <= HASH_LIMIT else ""
            writer.writerow(
                {
                    "ruta_relativa": rel(path),
                    "tamano_bytes": size,
                    "sha256": file_hash,
                    "estado": "ok",
                }
            )

    with open(LFS, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ruta_relativa", "tamano_bytes", "tamano_mb"])
        writer.writeheader()
        for path in files:
            size = os.path.getsize(long_path(path))
            if size >= LFS_LIMIT:
                writer.writerow(
                    {
                        "ruta_relativa": rel(path),
                        "tamano_bytes": size,
                        "tamano_mb": f"{size / (1024 * 1024):.2f}",
                    }
                )

    total_size = sum(os.path.getsize(long_path(path)) for path in files)
    print(f"archivos={len(files)}")
    print(f"tamano_gb={total_size / (1024 ** 3):.2f}")
    print(MANIFEST)
    print(LFS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
