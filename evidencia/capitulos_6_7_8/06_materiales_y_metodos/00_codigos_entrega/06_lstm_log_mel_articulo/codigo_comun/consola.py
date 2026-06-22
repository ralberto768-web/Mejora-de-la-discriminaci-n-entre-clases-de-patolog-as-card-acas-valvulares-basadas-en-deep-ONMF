from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


class Registrador:
    """Escribe cada mensaje en terminal y tambien en un archivo de texto."""

    def __init__(self, archivo: Path) -> None:
        self.archivo = archivo
        self.archivo.parent.mkdir(parents=True, exist_ok=True)
        self._manejador = self.archivo.open("w", encoding="utf-8")

    def escribir(self, mensaje: str = "") -> None:
        print(mensaje, flush=True)
        self._manejador.write(mensaje + "\n")
        self._manejador.flush()

    def cerrar(self) -> None:
        self._manejador.close()

    def __enter__(self) -> "Registrador":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cerrar()


def crear_registrador(carpeta_resultados: Path, nombre_archivo: str) -> Registrador:
    """Abre un registro y deja constancia de la fecha de ejecucion."""

    registro = Registrador(carpeta_resultados / nombre_archivo)
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registro.escribir("=" * 72)
    registro.escribir(f"Registro iniciado: {fecha}")
    registro.escribir(f"Python usado: {sys.executable}")
    registro.escribir("=" * 72)
    return registro


def titulo(registro: Registrador, texto: str) -> None:
    registro.escribir("")
    registro.escribir("=" * 72)
    registro.escribir(texto.upper())
    registro.escribir("=" * 72)
