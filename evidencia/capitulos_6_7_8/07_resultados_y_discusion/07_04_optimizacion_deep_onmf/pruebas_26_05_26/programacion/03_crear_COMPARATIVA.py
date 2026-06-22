from __future__ import annotations

"""Crea el documento COMPARATIVA uniendo inicializaciones y comparacion final."""

import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def localizar_raiz_objetivo() -> Path:
    archivo = Path(__file__).resolve()
    candidatos = [archivo.parent, *archivo.parents, Path.cwd(), *Path.cwd().parents]
    for base in candidatos:
        if (base / "comparacion final").exists() and (base / "pruebas para el 26-05-26").exists():
            return base
        candidato = base / "Programacion objetivo"
        if (candidato / "comparacion final").exists() and (candidato / "pruebas para el 26-05-26").exists():
            return candidato
    raise RuntimeError("No se ha encontrado la carpeta 'Programacion objetivo'.")


RAIZ_OBJETIVO = localizar_raiz_objetivo()
CARPETA_PRUEBA = RAIZ_OBJETIVO / "pruebas para el 26-05-26"
CARPETA_RESULTADOS = CARPETA_PRUEBA / "resultados comparativos"
CARPETA_IMAGENES = CARPETA_RESULTADOS / "COMPARATIVA_imagenes"
CARPETA_PROGRAMACION = CARPETA_PRUEBA / "programacion"

METRICAS_INICIALIZACIONES = CARPETA_RESULTADOS / "metricas_inicializaciones_deep_onmf.csv"
METRICAS_FINAL = (
    RAIZ_OBJETIVO
    / "comparacion final"
    / "04_prueba_ajustada_codigo_y_resultados"
    / "resultados"
    / "final_clave"
    / "metricas_comparacion_ajustada.csv"
)
FIGURA_FINAL_LADO_A_LADO = (
    RAIZ_OBJETIVO
    / "comparacion final"
    / "02_resultados"
    / "02_pdf_comparativo"
    / "Comparacion_Figura11_lado_a_lado.png"
)
FIGURA_DEEP_OPTIMIZADO = (
    RAIZ_OBJETIVO
    / "comparacion final"
    / "02_resultados"
    / "01_figuras_separadas"
    / "04_Figura11D_Deep_ONMF.png"
)
FIGURA_INICIALIZACIONES = CARPETA_RESULTADOS / "comparativa_tSNE_inicializaciones.png"


def copiar(origen: Path, nombre: str) -> Path:
    destino = CARPETA_IMAGENES / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)
    return destino


def preparar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inicializaciones = pd.read_csv(METRICAS_INICIALIZACIONES, encoding="utf-8-sig")
    final = pd.read_csv(METRICAS_FINAL, encoding="utf-8-sig")

    final_tabla = final.copy()
    final_tabla.insert(0, "bloque", "Comparacion final Figura 11")
    final_tabla["metodo_comparado"] = final_tabla["metodo"]
    final_tabla["nota"] = final_tabla["metodo"].map(
        {
            "Deep ONMF": "Deep ONMF optimizado: resultado principal del TFG.",
            "CNN": "CNN gana en silhouette de rasgos, pero no en t-SNE.",
            "DWT": "Representacion wavelet clasica; queda por debajo en t-SNE.",
            "MFCC": "Rasgos cepstrales; mejora a DWT/STFT en silhouette t-SNE, pero no a Deep ONMF.",
            "STFT": "Espectro base; muchos rasgos y peor Davies-Bouldin t-SNE.",
        }
    )
    final_tabla["error_relativo_final_medio"] = pd.NA
    final_tabla["tipo"] = "tecnologia"

    init_tabla = inicializaciones.copy()
    init_tabla.insert(0, "bloque", "Inicializaciones Deep ONMF")
    init_tabla["metodo_comparado"] = init_tabla["metodo"]
    init_tabla["tipo"] = "inicializacion"
    init_tabla["nota"] = init_tabla["codigo_metodo"].map(
        {
            "aleatoria_actual": "Referencia actual: mejor error de reconstruccion.",
            "nndsvd": "Mejor inicializacion para la foto t-SNE entre las tres propuestas.",
            "nndsvda": "Mejor separacion en rasgos SBV entre inicializaciones, pero peor t-SNE que NNDSVD.",
            "nndsvdar": "Parecida a NNDSVD en Davies-Bouldin t-SNE, pero peor silhouette t-SNE.",
        }
    )

    columnas = [
        "bloque",
        "tipo",
        "metodo_comparado",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
        "error_relativo_final_medio",
        "nota",
    ]
    tabla_global = pd.concat(
        [
            final_tabla[columnas],
            init_tabla[columnas],
        ],
        ignore_index=True,
    )
    return inicializaciones, final, tabla_global


def formato(valor: object) -> str:
    if pd.isna(valor):
        return "-"
    if isinstance(valor, float):
        return f"{valor:.6f}"
    return str(valor)


def tabla_markdown(tabla: pd.DataFrame) -> str:
    columnas = list(tabla.columns)
    lineas = [
        "| " + " | ".join(columnas) + " |",
        "| " + " | ".join("---" for _ in columnas) + " |",
    ]
    for fila in tabla.itertuples(index=False, name=None):
        lineas.append("| " + " | ".join(formato(valor) for valor in fila) + " |")
    return "\n".join(lineas)


def guardar_csv(tabla_global: pd.DataFrame) -> None:
    tabla_global.to_csv(CARPETA_RESULTADOS / "COMPARATIVA_tabla_global.csv", index=False, encoding="utf-8-sig")


def crear_tabla_imagen(tabla_global: pd.DataFrame) -> Path:
    columnas = [
        "bloque",
        "metodo_comparado",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
        "error_relativo_final_medio",
    ]
    tabla = tabla_global[columnas].copy()
    for columna in columnas[2:]:
        tabla[columna] = tabla[columna].map(formato)
    ruta = CARPETA_IMAGENES / "COMPARATIVA_tabla_global.png"
    fig, eje = plt.subplots(figsize=(18, 6.2))
    eje.axis("off")
    objeto = eje.table(cellText=tabla.values, colLabels=tabla.columns, cellLoc="center", loc="center")
    objeto.auto_set_font_size(False)
    objeto.set_fontsize(7.4)
    objeto.scale(1, 1.6)
    eje.set_title("COMPARATIVA global: tecnologias finales e inicializaciones Deep ONMF", fontsize=14, pad=18)
    fig.tight_layout()
    fig.savefig(ruta, dpi=220)
    plt.close(fig)
    return ruta


def crear_markdown(tabla_global: pd.DataFrame, imagenes: dict[str, Path]) -> None:
    final = tabla_global[tabla_global["bloque"] == "Comparacion final Figura 11"].copy()
    inicializaciones = tabla_global[tabla_global["bloque"] == "Inicializaciones Deep ONMF"].copy()

    mejor_final_tsne = final.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_final_db = final.sort_values("davies_bouldin_tsne", ascending=True).iloc[0]
    mejor_final_rasgos = final.sort_values("silhouette_features", ascending=False).iloc[0]
    mejor_init_tsne = inicializaciones.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_init_error = inicializaciones.sort_values("error_relativo_final_medio", ascending=True).iloc[0]
    mejor_init_rasgos = inicializaciones.sort_values("silhouette_features", ascending=False).iloc[0]

    rel = lambda ruta: ruta.relative_to(CARPETA_RESULTADOS).as_posix().replace(" ", "%20")
    lineas = [
        "# COMPARATIVA",
        "",
        "## Que compara este documento",
        "",
        "Este documento junta dos niveles de comparacion:",
        "",
        "1. La comparacion final de Figura 11 entre CNN, DWT, MFCC, Deep ONMF optimizado y STFT.",
        "2. La prueba pedida por Juan para Deep ONMF con inicializaciones NNDSVD, NNDSVDa y NNDSVDar, manteniendo tambien la inicializacion aleatoria actual como referencia.",
        "",
        "No son exactamente la misma prueba: la primera compara tecnologias completas y la segunda compara solo la forma de inicializar W y H dentro de Deep ONMF. Aun asi, juntas sirven para explicar que se queda como resultado principal y que inicializacion conviene estudiar.",
        "",
        "## Tabla global",
        "",
        tabla_markdown(
            tabla_global[
                [
                    "bloque",
                    "metodo_comparado",
                    "silhouette_features",
                    "davies_bouldin_features",
                    "silhouette_tsne",
                    "davies_bouldin_tsne",
                    "error_relativo_final_medio",
                    "nota",
                ]
            ]
        ),
        "",
        "## Figuras comparativas",
        "",
        "### Comparacion final de tecnologias",
        "",
        f"![Comparacion final Figura 11]({rel(imagenes['final_lado_a_lado'])})",
        "",
        "### Deep ONMF optimizado dentro de la comparacion final",
        "",
        f"![Deep ONMF optimizado]({rel(imagenes['deep_optimizado'])})",
        "",
        "### Inicializaciones Deep ONMF",
        "",
        f"![Inicializaciones Deep ONMF]({rel(imagenes['inicializaciones'])})",
        "",
        "### Tabla visual",
        "",
        f"![Tabla global]({rel(imagenes['tabla'])})",
        "",
        "## Que metodo se queda",
        "",
        "Para el resultado principal del TFG se debe mantener **Deep ONMF optimizado**.",
        "",
        f"- En la comparacion final, el mejor `silhouette_tsne` es **{mejor_final_tsne['metodo_comparado']}** con `{mejor_final_tsne['silhouette_tsne']:.6f}`.",
        f"- En la comparacion final, el mejor `davies_bouldin_tsne` es **{mejor_final_db['metodo_comparado']}** con `{mejor_final_db['davies_bouldin_tsne']:.6f}`.",
        f"- En la comparacion final, el mejor `silhouette_features` es **{mejor_final_rasgos['metodo_comparado']}** con `{mejor_final_rasgos['silhouette_features']:.6f}`.",
        "",
        "La lectura importante es esta: **CNN separa mejor en una metrica del espacio de rasgos original**, pero **Deep ONMF optimizado gana en la representacion visual t-SNE y en Davies-Bouldin**, que es lo que mas se conecta con la Figura 11.",
        "",
        "## Que inicializacion conviene destacar",
        "",
        f"Entre las inicializaciones pedidas por Juan, la mas interesante es **{mejor_init_tsne['metodo_comparado']}**.",
        "",
        f"- Mejor `silhouette_tsne` entre inicializaciones: **{mejor_init_tsne['metodo_comparado']}** con `{mejor_init_tsne['silhouette_tsne']:.6f}`.",
        f"- Menor `davies_bouldin_tsne` entre inicializaciones: **{inicializaciones.sort_values('davies_bouldin_tsne', ascending=True).iloc[0]['metodo_comparado']}** con `{inicializaciones.sort_values('davies_bouldin_tsne', ascending=True).iloc[0]['davies_bouldin_tsne']:.6f}`.",
        f"- Mejor `silhouette_features` entre inicializaciones: **{mejor_init_rasgos['metodo_comparado']}** con `{mejor_init_rasgos['silhouette_features']:.6f}`.",
        f"- Menor error de reconstruccion: **{mejor_init_error['metodo_comparado']}** con `{mejor_init_error['error_relativo_final_medio']:.6f}`.",
        "",
        "Esto significa que **NNDSVD mejora la foto t-SNE frente a la inicializacion aleatoria**, aunque la inicializacion aleatoria reconstruye con menos error. No hay contradiccion: reconstruir mejor la matriz X no siempre implica separar mejor las clases en la visualizacion.",
        "",
        "## Explicacion comparativa entre ellas",
        "",
        "### Aleatoria actual",
        "",
        "Es la referencia de partida. Tiene el menor error de reconstruccion, por lo que ajusta bien los espectrogramas. Su punto debil es que la separacion t-SNE queda por debajo de NNDSVD.",
        "",
        "### NNDSVD",
        "",
        "Arranca W y H usando una estructura extraida por SVD. Aunque su error de reconstruccion medio es mayor que el aleatorio, consigue la mejor separacion visual t-SNE entre las inicializaciones. Es la opcion mas defendible si Juan pregunta por inicializaciones comunes.",
        "",
        "### NNDSVDa",
        "",
        "Rellena los ceros de NNDSVD con la media de X. Mejora ligeramente el `silhouette_features` frente a la aleatoria, pero empeora la foto t-SNE. Puede ayudar a evitar ceros, pero aqui no es la mejor visualmente.",
        "",
        "### NNDSVDar",
        "",
        "Rellena los ceros con ruido pequeno. En esta prueba no supera a NNDSVD ni a la aleatoria en las metricas principales. Queda como variante explorada, no como candidata principal.",
        "",
        "### Deep ONMF optimizado",
        "",
        "Es el resultado que se debe presentar como linea principal del TFG porque en la comparacion global obtiene la mejor separacion t-SNE y el mejor Davies-Bouldin t-SNE. Las inicializaciones de Juan sirven como prueba adicional para mejorar el arranque de W y H, no para sustituir automaticamente la comparacion optimizada.",
        "",
        "## Conclusion para explicar a Juan",
        "",
        "La prueba muestra que **NNDSVD es la mejor de las tres inicializaciones propuestas para mejorar la visualizacion t-SNE de Deep ONMF**. Sin embargo, el resultado global que se mantiene como mejor para el TFG es **Deep ONMF optimizado**, porque frente a CNN, DWT, MFCC y STFT presenta la separacion t-SNE mas clara y el menor Davies-Bouldin en t-SNE.",
        "",
    ]
    (CARPETA_RESULTADOS / "COMPARATIVA.md").write_text("\n".join(lineas), encoding="utf-8")


def pagina_texto(pdf: PdfPages, titulo: str, lineas: list[str]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.94, titulo, fontsize=18, weight="bold", va="top")
    y = 0.88
    for linea in lineas:
        fig.text(0.06, y, linea, fontsize=10.5, va="top", wrap=True)
        y -= 0.038
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11.69, 8.27))
            y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pagina_imagen(pdf: PdfPages, ruta: Path, titulo: str) -> None:
    imagen = plt.imread(ruta)
    fig, eje = plt.subplots(figsize=(11.69, 8.27))
    eje.imshow(imagen)
    eje.axis("off")
    eje.set_title(titulo, fontsize=14, pad=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def crear_pdf(tabla_global: pd.DataFrame, imagenes: dict[str, Path]) -> None:
    final = tabla_global[tabla_global["bloque"] == "Comparacion final Figura 11"].copy()
    inicializaciones = tabla_global[tabla_global["bloque"] == "Inicializaciones Deep ONMF"].copy()
    mejor_final = final.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_init = inicializaciones.sort_values("silhouette_tsne", ascending=False).iloc[0]
    ruta_pdf = CARPETA_RESULTADOS / "COMPARATIVA.pdf"
    with PdfPages(ruta_pdf) as pdf:
        pagina_texto(
            pdf,
            "COMPARATIVA",
            [
                "Documento conjunto con la comparacion final de tecnologias y la prueba de inicializaciones Deep ONMF.",
                "",
                f"Resultado principal del TFG: {mejor_final['metodo_comparado']} por mejor silhouette t-SNE ({mejor_final['silhouette_tsne']:.4f}) y mejor separacion visual global.",
                f"Inicializacion mas interesante para Juan: {mejor_init['metodo_comparado']} por mejor silhouette t-SNE entre inicializaciones ({mejor_init['silhouette_tsne']:.4f}).",
                "",
                "CNN gana en silhouette del espacio de rasgos original, pero Deep ONMF optimizado gana en la lectura visual de Figura 11 y en Davies-Bouldin t-SNE.",
                "NNDSVD mejora la foto de Deep ONMF respecto a la aleatoria, aunque no reduce el error de reconstruccion.",
            ],
        )
        pagina_imagen(pdf, imagenes["tabla"], "Tabla global de metricas")
        pagina_imagen(pdf, imagenes["final_lado_a_lado"], "Comparacion final Figura 11")
        pagina_imagen(pdf, imagenes["deep_optimizado"], "Deep ONMF optimizado")
        pagina_imagen(pdf, imagenes["inicializaciones"], "Inicializaciones Deep ONMF")
        pagina_texto(
            pdf,
            "Lectura final",
            [
                "1. Se queda Deep ONMF optimizado como resultado principal porque es el que mejor sostiene la Figura 11 frente al resto de tecnologias.",
                "2. Se destaca NNDSVD como inicializacion recomendada para seguir probando porque mejora la separacion t-SNE frente al arranque aleatorio.",
                "3. NNDSVDa mejora algo la separacion en rasgos, pero no la visualizacion t-SNE.",
                "4. NNDSVDar no queda como opcion principal en esta prueba.",
                "5. El error de reconstruccion y la separacion de clases no miden exactamente lo mismo: por eso la aleatoria puede reconstruir mejor y NNDSVD puede separar mejor visualmente.",
            ],
        )


def copiar_script_final() -> None:
    destino = CARPETA_PROGRAMACION / "03_crear_COMPARATIVA.py"
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), destino)


def main() -> int:
    CARPETA_IMAGENES.mkdir(parents=True, exist_ok=True)
    copiar_script_final()
    inicializaciones, final, tabla_global = preparar_datos()
    guardar_csv(tabla_global)
    imagenes = {
        "final_lado_a_lado": copiar(FIGURA_FINAL_LADO_A_LADO, "01_comparacion_final_figura11_lado_a_lado.png"),
        "deep_optimizado": copiar(FIGURA_DEEP_OPTIMIZADO, "02_deep_onmf_optimizado_figura11d.png"),
        "inicializaciones": copiar(FIGURA_INICIALIZACIONES, "03_comparativa_inicializaciones_deep_onmf.png"),
    }
    imagenes["tabla"] = crear_tabla_imagen(tabla_global)
    crear_markdown(tabla_global, imagenes)
    crear_pdf(tabla_global, imagenes)
    print(f"COMPARATIVA.md: {CARPETA_RESULTADOS / 'COMPARATIVA.md'}")
    print(f"COMPARATIVA.pdf: {CARPETA_RESULTADOS / 'COMPARATIVA.pdf'}")
    print(f"COMPARATIVA_tabla_global.csv: {CARPETA_RESULTADOS / 'COMPARATIVA_tabla_global.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
