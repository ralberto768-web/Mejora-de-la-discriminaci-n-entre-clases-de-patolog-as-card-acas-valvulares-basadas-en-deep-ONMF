from __future__ import annotations

import importlib.util
import platform
import sys


MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "PyMuPDF": "fitz",
    "PyWavelets": "pywt",
}

OPTIONAL_MODULES = {
    "openpyxl": "openpyxl",
}


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("Plataforma:", platform.platform())
    missing = []
    for label, module in MODULES.items():
        ok = importlib.util.find_spec(module) is not None
        print(f"{label:14} {'OK' if ok else 'FALTA'}")
        if not ok:
            missing.append(label)
    for label, module in OPTIONAL_MODULES.items():
        ok = importlib.util.find_spec(module) is not None
        print(f"{label:14} {'OK' if ok else 'OPCIONAL_NO_INSTALADO'}")
    if missing:
        print("\nFaltan dependencias. Instalar con: python -m pip install -r requirements.txt")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
