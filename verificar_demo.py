from pathlib import Path

from clasificar_senal import analizar_archivo, cargar_modelo


BASE = Path(__file__).resolve().parent
MODELO = cargar_modelo(BASE / "modelo_basico.json")


def comprobar(nombre_fichero: str, clase_esperada: str) -> None:
    ruta = BASE / "datos" / nombre_fichero
    analisis = analizar_archivo(ruta, MODELO)
    clase_obtenida = analisis["resultado"]["clase"]
    if clase_obtenida != clase_esperada:
        raise AssertionError(
            f"{nombre_fichero}: se esperaba {clase_esperada}, pero se obtuvo {clase_obtenida}"
        )
    print(f"{nombre_fichero}: OK -> {analisis['resultado']['descripcion']}")


if __name__ == "__main__":
    comprobar("pcg_sano.wav", "sana")
    comprobar("pcg_estenosis_aortica.wav", "estenosis_aortica")
    comprobar("pcg_regurgitacion_mitral.wav", "regurgitacion_mitral")
    comprobar("pcg_estenosis_mitral.wav", "estenosis_mitral")
    comprobar("pcg_prolapso_mitral.wav", "prolapso_mitral")
    print("Verificacion completada correctamente.")
