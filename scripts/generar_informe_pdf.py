from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "informe_general" / "INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf"


SECTIONS = [
    (
        "1. Proposito del repositorio",
        [
            "Este informe resume el material incluido para evaluar un sistema de clasificacion de sonidos cardiacos basado en Deep-ONMF.",
            "El objetivo es revisar de forma ordenada el metodo, las evidencias experimentales y las conclusiones tecnicas sin depender de rutas locales ni de documentos privados.",
        ],
    ),
    (
        "2. Metodo Deep-ONMF y clasificacion",
        [
            "El flujo experimental parte de senales de fonocardiograma. A partir de ellas se extraen caracteristicas temporales mediante Deep-ONMF, obteniendo matrices internas como W, H y H3.",
            "La representacion H3 se usa como entrada temporal principal para el sistema de clasificacion, mientras que W se conserva para comparaciones de separabilidad y rendimiento.",
            "La evidencia tecnica se conserva en metodologia/, donde se agrupan codigo, configuraciones y documentos de apoyo.",
        ],
    ),
    (
        "3. Bases de datos y escenarios",
        [
            "La evaluacion distingue un escenario sin ruido y un escenario con ruido AWGN. El primero funciona como referencia; el segundo permite estudiar robustez ante degradacion artificial de la senal.",
            "Los audios fuente completos no se distribuyen en GitHub. Para repetir ejecuciones completas deben colocarse en datos_externos/ siguiendo la guia correspondiente.",
        ],
    ),
    (
        "4. Metricas y evaluacion",
        [
            "Las salidas experimentales incluyen accuracy, precision, recall, F1-score, matrices de confusion y tablas resumen cuando estan disponibles.",
            "La evaluacion se organiza con validacion cruzada k-fold para reducir dependencia de una unica particion.",
        ],
    ),
    (
        "5. Optimizacion Deep-ONMF",
        [
            "La optimizacion estudia el efecto del numero de capas y de las dimensiones internas del modelo.",
            "La evidencia principal se encuentra en resultados/04_optimizacion_deep_onmf/. Las ejecuciones historicas complementarias estan en resultados/04_optimizacion_deep_onmf_historico/.",
            "En la revision debe separarse la configuracion objetivamente evaluada de la interpretacion sobre la configuracion mas defendible.",
        ],
    ),
    (
        "6. Escenario sin ruido",
        [
            "El escenario sin ruido permite comparar el comportamiento del enfoque temporal Deep-ONMF frente a representaciones clasicas.",
            "La evidencia esta en resultados/05_escenario_sin_ruido/ y sirve como referencia para valorar rendimiento y dimension de entrada.",
        ],
    ),
    (
        "7. Escenario ruidoso AWGN",
        [
            "El escenario con AWGN analiza la estabilidad del sistema ante ruido artificial.",
            "La evidencia esta en resultados/06_escenario_ruidoso_awgn/ y debe interpretarse por tendencias de robustez, no solo por un valor puntual.",
        ],
    ),
    (
        "8. Comparacion temporal-espectral y H3-W",
        [
            "La carpeta resultados/07_comparacion_temporal_espectral_h3_w/ contiene la evidencia para comparar caracteristicas temporales frente a espectrales y para contrastar H3 frente a W.",
            "La comparacion combina representabilidad visual mediante proyecciones o nubes de puntos y rendimiento cuantitativo mediante metricas de clasificacion.",
        ],
    ),
    (
        "9. Discusion tecnica",
        [
            "Los resultados deben interpretarse considerando rendimiento, dimension de entrada, coste computacional y robustez.",
            "Aunque las representaciones temporales no siempre superen a las clasicas en todas las metricas, pueden aportar una entrada de menor dimension y un coste potencialmente menor.",
        ],
    ),
    (
        "10. Conclusiones y verificacion",
        [
            "El repositorio permite defender un flujo completo de extraccion temporal, clasificacion y evaluacion bajo condiciones limpias y ruidosas.",
            "La integridad del paquete puede comprobarse ejecutando run_all.bat todo. Los manifiestos estan en github/ y verificacion/.",
        ],
    ),
]


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(1.8 * cm, 1.0 * cm, "Informe general de resultados Deep-ONMF")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def main() -> int:
    PDF.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="MainTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#102033"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=colors.HexColor("#1f3a5f"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Mono",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            backColor=colors.HexColor("#f2f4f7"),
            borderPadding=4,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.6 * cm,
    )

    story = [
        Paragraph("Informe general de resultados Deep-ONMF", styles["MainTitle"]),
        Paragraph(
            "Repositorio publico de codigo, resultados y evidencia experimental para clasificacion de patologias cardiacas valvulares a partir de sonidos cardiacos.",
            styles["Body"],
        ),
        Spacer(1, 12),
        Paragraph("Rutas principales", styles["SectionTitle"]),
    ]

    table_data = [
        ["Ruta", "Uso"],
        ["metodologia/", "Metodo Deep-ONMF, extraccion y clasificacion."],
        ["resultados/04_optimizacion_deep_onmf/", "Busqueda de capas y dimensiones."],
        ["resultados/05_escenario_sin_ruido/", "Evaluacion en condiciones limpias."],
        ["resultados/06_escenario_ruidoso_awgn/", "Evaluacion con AWGN y niveles SNR."],
        ["resultados/07_comparacion_.../", "Comparacion H3, W y representaciones espectrales."],
        ["verificacion/", "Manifiestos de integridad de la evidencia."],
    ]
    table = Table(table_data, colWidths=[7.1 * cm, 8.9 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2ef")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#aab7c4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Comprobacion rapida", styles["SectionTitle"]))
    story.append(Paragraph("run_all.bat todo", styles["Mono"]))
    story.append(PageBreak())

    for idx, (title, paragraphs) in enumerate(SECTIONS):
        story.append(Paragraph(title, styles["SectionTitle"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["Body"]))
        if idx != len(SECTIONS) - 1:
            story.append(PageBreak())

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
