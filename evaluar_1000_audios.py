import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from clasificar_senal import analizar_archivo, cargar_modelo


BASE = Path(__file__).resolve().parent
CLASES = {
    "N": "sana",
    "AS": "estenosis_aortica",
    "MR": "regurgitacion_mitral",
    "MS": "estenosis_mitral",
    "MVP": "prolapso_mitral",
}
ORDEN_CLASES = [
    "sana",
    "estenosis_aortica",
    "regurgitacion_mitral",
    "estenosis_mitral",
    "prolapso_mitral",
]


def descripcion_clase(modelo: dict, clase: str) -> str:
    return modelo.get("class_descriptions", {}).get(clase, clase)


def modelo_sin_audio(modelo: dict, source_file: str) -> dict:
    copia = dict(modelo)
    copia["references"] = [
        referencia
        for referencia in modelo["references"]
        if referencia.get("source_file") != source_file
    ]
    return copia


def cargar_audios(carpeta_datos: Path) -> list[tuple[Path, str, str]]:
    audios: list[tuple[Path, str, str]] = []
    for carpeta, clase in CLASES.items():
        carpeta_clase = carpeta_datos / carpeta
        if not carpeta_clase.exists():
            raise FileNotFoundError(f"No existe la carpeta esperada: {carpeta_clase}")
        for ruta in sorted(carpeta_clase.glob("*.wav")):
            audios.append((ruta, carpeta, clase))
    return audios


def guardar_detalle(resultados: list[dict[str, object]], carpeta_salida: Path) -> None:
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta_salida / "detalle_evaluacion_1000.csv"
    campos = [
        "audio",
        "clase_esperada",
        "clase_obtenida",
        "correcto",
        "confianza",
    ]
    with ruta_csv.open("w", newline="", encoding="utf-8") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=campos)
        escritor.writeheader()
        for fila in resultados:
            escritor.writerow({campo: fila[campo] for campo in campos})


def imprimir_resumen(
    resultados: list[dict[str, object]],
    modelo: dict,
    max_fallos: int,
) -> None:
    total = len(resultados)
    correctos = sum(1 for fila in resultados if fila["correcto"])
    fallos = total - correctos
    exactitud = correctos / total if total else 0.0

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for fila in resultados:
        confusion[str(fila["clase_esperada"])][str(fila["clase_obtenida"])] += 1

    print("Evaluación completa sobre 1000 audios")
    print("Protocolo: leave-one-out. Cada audio probado se elimina antes de las referencias.")
    print()
    print(f"Audios probados: {total}")
    print(f"Correctos: {correctos}")
    print(f"Fallos: {fallos}")
    print(f"Exactitud: {exactitud:.4f} ({exactitud * 100:.2f} %)")
    print()

    print("Matriz de confusión")
    cabecera = ["esperada \\ obtenida"] + [descripcion_clase(modelo, c) for c in ORDEN_CLASES]
    anchuras = [max(18, len(texto)) for texto in cabecera]
    for esperada in ORDEN_CLASES:
        anchuras[0] = max(anchuras[0], len(descripcion_clase(modelo, esperada)))
        for indice, obtenida in enumerate(ORDEN_CLASES, start=1):
            anchuras[indice] = max(anchuras[indice], len(str(confusion[esperada][obtenida])))

    print(" | ".join(texto.ljust(anchuras[i]) for i, texto in enumerate(cabecera)))
    print("-+-".join("-" * ancho for ancho in anchuras))
    for esperada in ORDEN_CLASES:
        fila = [descripcion_clase(modelo, esperada)]
        fila.extend(str(confusion[esperada][obtenida]) for obtenida in ORDEN_CLASES)
        print(" | ".join(texto.ljust(anchuras[i]) for i, texto in enumerate(fila)))

    errores = [fila for fila in resultados if not fila["correcto"]]
    print()
    print(f"Primeros fallos mostrados: {min(max_fallos, len(errores))} de {len(errores)}")
    for fila in errores[:max_fallos]:
        print(
            f"- {fila['audio']}: esperado "
            f"{descripcion_clase(modelo, str(fila['clase_esperada']))}; obtenido "
            f"{descripcion_clase(modelo, str(fila['clase_obtenida']))}; "
            f"confianza {float(fila['confianza']):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa los 1000 audios PCG con exclusión leave-one-out."
    )
    parser.add_argument("--datos", default="datos_1000", help="Carpeta con subcarpetas N, AS, MR, MS y MVP.")
    parser.add_argument("--modelo", default="modelo_basico.json", help="Fichero JSON del modelo.")
    parser.add_argument("--salida", default="resultados", help="Carpeta donde guardar el CSV de detalle.")
    parser.add_argument("--max-fallos", type=int, default=15, help="Número máximo de fallos que se imprimen.")
    parser.add_argument("--no-guardar", action="store_true", help="No guarda el CSV de detalle.")
    args = parser.parse_args()

    carpeta_datos = (BASE / args.datos).resolve()
    modelo = cargar_modelo(BASE / args.modelo)
    audios = cargar_audios(carpeta_datos)

    resultados: list[dict[str, object]] = []
    for ruta, carpeta, clase_esperada in audios:
        source_file = f"{carpeta}/{ruta.name}"
        modelo_evaluacion = modelo_sin_audio(modelo, source_file)
        analisis = analizar_archivo(ruta, modelo_evaluacion)
        resultado = analisis["resultado"]
        clase_obtenida = resultado["clase"]
        resultados.append(
            {
                "audio": source_file,
                "clase_esperada": clase_esperada,
                "clase_obtenida": clase_obtenida,
                "correcto": clase_obtenida == clase_esperada,
                "confianza": float(resultado["confianza"]),
            }
        )

    imprimir_resumen(resultados, modelo, args.max_fallos)

    if not args.no_guardar:
        guardar_detalle(resultados, BASE / args.salida)
        print()
        print(f"Detalle guardado en: {args.salida}/detalle_evaluacion_1000.csv")


if __name__ == "__main__":
    main()
