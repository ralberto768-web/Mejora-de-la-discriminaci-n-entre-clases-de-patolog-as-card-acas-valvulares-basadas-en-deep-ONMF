from pathlib import Path
from random import SystemRandom

from clasificar_senal import analizar_archivo, cargar_modelo


BASE = Path(__file__).resolve().parent
MODELO = cargar_modelo(BASE / "modelo_basico.json")

CASOS = [
    ("audio1.wav", "sana"),
    ("audio2.wav", "estenosis_aortica"),
    ("audio3.wav", "regurgitacion_mitral"),
    ("audio4.wav", "estenosis_mitral"),
    ("audio5.wav", "prolapso_mitral"),
]


def descripcion_clase(clase: str) -> str:
    return MODELO.get("class_descriptions", {}).get(clase, clase)


def comprobar(nombre_fichero: str, clase_esperada: str) -> dict[str, object]:
    ruta = BASE / "datos" / nombre_fichero
    analisis = analizar_archivo(ruta, MODELO)
    resultado = analisis["resultado"]
    clase_obtenida = resultado["clase"]
    return {
        "audio": nombre_fichero,
        "esperada": clase_esperada,
        "obtenida": clase_obtenida,
        "descripcion_obtenida": resultado["descripcion"],
        "confianza": resultado["confianza"],
        "correcto": clase_obtenida == clase_esperada,
    }


if __name__ == "__main__":
    casos = CASOS[:]
    SystemRandom().shuffle(casos)

    print("Ejecución aleatoria de la demo")
    print("El orden cambia en cada lanzamiento.\n")
    print("Resultados durante la prueba:")

    resultados = []
    for indice, (nombre_fichero, clase_esperada) in enumerate(casos, start=1):
        resultado = comprobar(nombre_fichero, clase_esperada)
        resultados.append(resultado)
        print(
            f"{indice}. {resultado['audio']} -> "
            f"{resultado['descripcion_obtenida']} "
            f"(confianza {resultado['confianza']:.3f})"
        )

    print("\nResumen final con la clase correcta:")
    aciertos = 0
    for resultado in resultados:
        correcto = bool(resultado["correcto"])
        if correcto:
            aciertos += 1
        estado = "CORRECTO" if correcto else "ERROR"
        print(
            f"- {resultado['audio']}: {estado} | "
            f"esperado: {descripcion_clase(str(resultado['esperada']))} | "
            f"obtenido: {descripcion_clase(str(resultado['obtenida']))}"
        )

    total = len(resultados)
    fallos = total - aciertos
    if fallos:
        raise AssertionError(f"Verificación fallida: {fallos} fallo(s) de {total} pruebas.")

    print(f"\nVerificación completada correctamente: {aciertos}/{total} aciertos.")
