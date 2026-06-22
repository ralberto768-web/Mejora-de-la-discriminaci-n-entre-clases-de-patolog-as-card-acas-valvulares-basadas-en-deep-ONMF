from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


CARPETA = Path(__file__).resolve().parent
RAIZ_CODIGOS = CARPETA.parent
CARPETAS = [
    "01_deep_onmf_H_articulo_original",
    "02_deep_onmf_H_rellenando_audios",
    "03_deep_onmf_H_mejorado_NNDSVD",
    "04_matriz_W_por_audio",
    "05_cnn_log_mel_articulo",
    "06_lstm_log_mel_articulo",
    "07_baselines_STFT_MFCC_DWT",
    "08_comparador_figura11",
]


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verificador maestro de la carpeta Codigos.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--limite-por-clase", type=int, default=2)
    parser.add_argument("--solo-sintaxis", action="store_true", help="No ejecuta verificadores rapidos.")
    return parser.parse_args()


def ejecutar(comando: list[str], cwd: Path) -> tuple[bool, str]:
    proceso = subprocess.run(comando, cwd=cwd, text=True, capture_output=True)
    salida = (proceso.stdout or "") + (proceso.stderr or "")
    return proceso.returncode == 0, salida.strip()


def main() -> int:
    args = parsear_argumentos()
    datos = args.datos.expanduser().resolve() if args.datos is not None else None
    salida = CARPETA / "resultados_verificacion_entrega"
    salida.mkdir(parents=True, exist_ok=True)
    filas = []

    for nombre in CARPETAS:
        carpeta = RAIZ_CODIGOS / nombre
        ok_sintaxis, salida_sintaxis = ejecutar([sys.executable, "-m", "compileall", "-q", str(carpeta)], carpeta)
        filas.append({"carpeta": nombre, "comprobacion": "compileall", "ok": ok_sintaxis, "detalle": salida_sintaxis})

        if args.solo_sintaxis:
            continue
        verificar = carpeta / "verificar.py"
        comando = [sys.executable, str(verificar), "--limite-por-clase", str(args.limite_por_clase)]
        if datos is not None:
            comando += ["--datos", str(datos)]
        ok_verificacion, salida_verificacion = ejecutar(comando, carpeta)
        filas.append({"carpeta": nombre, "comprobacion": "verificar.py", "ok": ok_verificacion, "detalle": salida_verificacion})

    ruta_csv = salida / "informe_verificacion_entrega.csv"
    with ruta_csv.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["carpeta", "comprobacion", "ok", "detalle"])
        escritor.writeheader()
        escritor.writerows(filas)

    ruta_txt = salida / "informe_verificacion_entrega.txt"
    lineas = ["INFORME DE VERIFICACION DE CODIGOS", ""]
    for fila in filas:
        estado = "OK" if fila["ok"] else "ERROR"
        lineas.append(f"{estado} | {fila['carpeta']} | {fila['comprobacion']}")
        if fila["detalle"]:
            lineas.append(fila["detalle"])
        lineas.append("")
    ruta_txt.write_text("\n".join(lineas), encoding="utf-8")

    todo_ok = all(fila["ok"] for fila in filas)
    print(f"Informe CSV: {ruta_csv}")
    print(f"Informe TXT: {ruta_txt}")
    if not todo_ok:
        raise SystemExit("Alguna comprobacion ha fallado. Revisa el informe.")
    print("Verificacion maestra OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
