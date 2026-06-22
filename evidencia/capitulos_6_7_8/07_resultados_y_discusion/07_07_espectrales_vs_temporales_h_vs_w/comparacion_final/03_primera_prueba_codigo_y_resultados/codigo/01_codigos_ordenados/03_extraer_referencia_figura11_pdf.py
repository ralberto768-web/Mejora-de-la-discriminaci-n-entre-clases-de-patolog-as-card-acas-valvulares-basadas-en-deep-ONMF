from __future__ import annotations

"""Extrae la Figura 11 del PDF objetivo y crea una referencia de estudio.

La Figura 11 original contiene CNN, DWT, MFCC y deep ONMF. STFT se anade en el
PDF final como panel local extra cuando ya existe la reproduccion Python.
"""

from pathlib import Path
import os

CARPETA_COMPARACION = Path(__file__).resolve().parents[1]
MPL_CACHE = CARPETA_COMPARACION / "02_resultados" / ".cache_matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


RAIZ_OBJETIVO = CARPETA_COMPARACION.parent
PDF_OBJETIVO = RAIZ_OBJETIVO / "articulo_objetivo.pdf"
CARPETA_REFERENCIA = CARPETA_COMPARACION / "02_resultados" / "04_referencia_pdf_articulo"
CARPETA_FOTOS = CARPETA_REFERENCIA / "01_figuras_separadas_articulo"
CARPETA_PDF = CARPETA_REFERENCIA / "02_pdf_lado_a_lado"
STFT_LOCAL = CARPETA_COMPARACION / "02_resultados" / "01_figuras_separadas" / "05_Figura11_extra_STFT.png"

# Coordenadas en puntos PDF para la pagina 22 del articulo objetivo local.
PAGINA_FIGURA_11 = 21
RECT_FIGURA_COMPLETA = fitz.Rect(45, 52, 395, 294)
PANELES = {
    "CNN": (fitz.Rect(40, 50, 220, 158), "01_Articulo_Figura11A_CNN.png"),
    "DWT": (fitz.Rect(220, 50, 399, 158), "02_Articulo_Figura11B_DWT.png"),
    "MFCC": (fitz.Rect(40, 160, 220, 267), "03_Articulo_Figura11C_MFCC.png"),
    "Deep ONMF": (fitz.Rect(220, 160, 399, 267), "04_Articulo_Figura11D_Deep_ONMF.png"),
}


def preparar_carpetas() -> None:
    for carpeta in (CARPETA_REFERENCIA, CARPETA_FOTOS, CARPETA_PDF):
        carpeta.mkdir(parents=True, exist_ok=True)


def guardar_clip(pagina: fitz.Page, rectangulo: fitz.Rect, ruta: Path, zoom: int = 5) -> None:
    pixmap = pagina.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rectangulo, alpha=False)
    pixmap.save(ruta)


def figura_referencia_pdf(rutas_paneles: dict[str, Path]) -> Path:
    ruta_pdf = CARPETA_PDF / "Referencia_Articulo_Figura11_y_STFT_extra.pdf"
    with PdfPages(ruta_pdf) as pdf:
        fig, ejes = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [1.35, 1.0]})
        eje_articulo, eje_stft = ejes
        eje_articulo.imshow(plt.imread(CARPETA_FOTOS / "00_Articulo_Figura11_completa.png"))
        eje_articulo.set_title("Figura 11 original: CNN, DWT, MFCC y deep ONMF", fontsize=12)
        eje_articulo.axis("off")

        if STFT_LOCAL.exists():
            eje_stft.imshow(plt.imread(STFT_LOCAL))
            eje_stft.set_title("STFT extra - reproduccion local", fontsize=11)
        else:
            eje_stft.text(
                0.5,
                0.5,
                "STFT local no encontrada.\nEjecuta primero 01_comparar_figura11.py.",
                va="center",
                ha="center",
            )
        eje_stft.axis("off")
        fig.suptitle("Referencia de la Figura 11 extraida del PDF objetivo", fontsize=16)
        fig.text(
            0.02,
            0.02,
            "Lectura: en la figura original el panel (d) deep ONMF es el que muestra la separacion "
            "mas clara. STFT se conserva como panel extra local porque no pertenece a la Figura 11 del PDF.",
            fontsize=11,
            wrap=True,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        pdf.savefig(fig)
        fig.savefig(CARPETA_PDF / "Referencia_Articulo_Figura11_y_STFT_extra.png", dpi=280)
        plt.close(fig)
    return ruta_pdf


def guardar_explicacion(ruta_pdf: Path, rutas_paneles: dict[str, Path]) -> None:
    fotos = "\n".join(f"- `{ruta.relative_to(CARPETA_REFERENCIA)}`" for ruta in rutas_paneles.values())
    texto = f"""# Referencia de la Figura 11 del articulo objetivo

## Que contiene esta carpeta

Esta carpeta no sustituye la reproduccion local. Sirve para estudiar la figura
que aparece en el propio `articulo_objetivo.pdf`.

## Fotos separadas extraidas del PDF

{fotos}

- `01_figuras_separadas_articulo/00_Articulo_Figura11_completa.png`

## PDF lado a lado

- `{ruta_pdf.relative_to(CARPETA_REFERENCIA)}`

La Figura 11 del articulo incluye:

1. `(a) CNN`.
2. `(b) DWT`.
3. `(c) MFCC`.
4. `(d) deep ONMF`.

STFT se anade en el PDF de esta carpeta como panel extra local si ya se ha
generado la reproduccion Python. Se marca como extra para no confundirla con la
figura original.

## Lectura visual

En la figura del articulo el panel `(d) deep ONMF` muestra grupos mas compactos y
separados que los paneles CNN, DWT y MFCC. Esta es la referencia visual que el
articulo usa para defender la separabilidad de su metodo propuesto.
"""
    (CARPETA_REFERENCIA / "00_EXPLICACION_REFERENCIA_ARTICULO.md").write_text(texto, encoding="utf-8")


def main() -> int:
    preparar_carpetas()
    with fitz.open(PDF_OBJETIVO) as pdf:
        pagina = pdf[PAGINA_FIGURA_11]
        guardar_clip(pagina, RECT_FIGURA_COMPLETA, CARPETA_FOTOS / "00_Articulo_Figura11_completa.png")
        rutas_paneles: dict[str, Path] = {}
        for nombre, (rectangulo, nombre_archivo) in PANELES.items():
            ruta = CARPETA_FOTOS / nombre_archivo
            guardar_clip(pagina, rectangulo, ruta)
            rutas_paneles[nombre] = ruta

    ruta_pdf = figura_referencia_pdf(rutas_paneles)
    guardar_explicacion(ruta_pdf, rutas_paneles)
    print(f"Referencia extraida desde: {PDF_OBJETIVO}")
    print(f"PDF de referencia generado: {ruta_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
