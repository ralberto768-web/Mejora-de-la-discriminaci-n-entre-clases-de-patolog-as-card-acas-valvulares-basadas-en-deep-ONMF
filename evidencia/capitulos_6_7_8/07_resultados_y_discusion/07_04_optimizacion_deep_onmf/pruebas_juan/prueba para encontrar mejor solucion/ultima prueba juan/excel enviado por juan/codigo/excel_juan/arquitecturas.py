from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class ParejaArquitecturas:
    pareja: int
    inicio: int
    fin: int
    numero_capas: int
    decreciente: tuple[int, ...]
    creciente: tuple[int, ...]


def etiqueta(rangos: tuple[int, ...]) -> str:
    return "-".join(str(int(valor)) for valor in rangos)


def clave(rangos: tuple[int, ...]) -> str:
    return "_".join(str(int(valor)) for valor in rangos)


def convertir_etiqueta(valor: str) -> tuple[int, ...]:
    return tuple(int(numero) for numero in str(valor).split("-"))


def generar_arquitecturas() -> list[tuple[int, ...]]:
    arquitecturas: list[tuple[int, ...]] = []
    for inicio in range(32, 7, -2):
        for fin in range(2, 11, 2):
            if inicio <= fin:
                continue
            for numero_capas in (2, 3, 4):
                capas = tuple(
                    int(valor)
                    for valor in np.round(
                        np.geomspace(inicio, fin, numero_capas)
                    ).astype(int)
                )
                arquitecturas.append(capas)
    return arquitecturas


def _cadenas_compartidas(archivo: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archivo.namelist():
        return []
    raiz = ET.fromstring(archivo.read("xl/sharedStrings.xml"))
    return [
        "".join(
            nodo.text or ""
            for nodo in elemento.iter(
                "{http://schemas.openxmlformats.org/"
                "spreadsheetml/2006/main}t"
            )
        )
        for elemento in raiz
    ]


def leer_arquitecturas_excel(ruta: Path) -> list[tuple[int, ...]]:
    with zipfile.ZipFile(ruta) as archivo:
        compartidas = _cadenas_compartidas(archivo)
        libro = ET.fromstring(archivo.read("xl/workbook.xml"))
        relaciones = ET.fromstring(
            archivo.read("xl/_rels/workbook.xml.rels")
        )
        destinos = {
            relacion.attrib["Id"]: relacion.attrib["Target"]
            for relacion in relaciones
        }
        hoja_configuraciones = None
        for hoja in libro.find("m:sheets", NS):
            if hoja.attrib["name"] != "Configuraciones":
                continue
            id_relacion = hoja.attrib[
                "{http://schemas.openxmlformats.org/"
                "officeDocument/2006/relationships}id"
            ]
            hoja_configuraciones = destinos[id_relacion]
            break
        if hoja_configuraciones is None:
            raise ValueError("El Excel no contiene la hoja Configuraciones")
        if not hoja_configuraciones.startswith("xl/"):
            hoja_configuraciones = (
                "xl/" + hoja_configuraciones.lstrip("/")
            )
        raiz = ET.fromstring(archivo.read(hoja_configuraciones))
        valores: list[str] = []
        for celda in raiz.findall(".//m:sheetData/m:row/m:c", NS):
            if not str(celda.attrib.get("r", "")).startswith("A"):
                continue
            nodo = celda.find("m:v", NS)
            if nodo is None:
                continue
            valor = nodo.text or ""
            if celda.attrib.get("t") == "s":
                valor = compartidas[int(valor)]
            valores.append(valor)
    if not valores or valores[0] != "Arquitectura_lista":
        raise ValueError("Encabezado inesperado en la hoja Configuraciones")
    arquitecturas = [
        tuple(int(numero) for numero in re.findall(r"\d+", valor))
        for valor in valores[1:]
    ]
    if any(len(rangos) not in (2, 3, 4) for rangos in arquitecturas):
        raise ValueError("El Excel contiene una profundidad no permitida")
    return arquitecturas


def construir_plan(ruta_excel: Path) -> pd.DataFrame:
    del_excel = leer_arquitecturas_excel(ruta_excel)
    generadas = generar_arquitecturas()
    if del_excel != generadas:
        raise AssertionError(
            "Las arquitecturas del Excel no coinciden con el codigo de Juan"
        )
    if len(del_excel) != 186 or len(set(del_excel)) != 186:
        raise AssertionError("Se esperaban 186 arquitecturas unicas")
    filas: list[dict[str, object]] = []
    for pareja, decreciente in enumerate(del_excel, start=1):
        creciente = tuple(reversed(decreciente))
        inicio, fin = decreciente[0], decreciente[-1]
        repetida = any(
            izquierda == derecha
            for izquierda, derecha in zip(
                decreciente,
                decreciente[1:],
            )
        )
        for sentido, rangos in (
            ("decreciente", decreciente),
            ("creciente", creciente),
        ):
            filas.append(
                {
                    "pareja": pareja,
                    "sentido": sentido,
                    "inicio": inicio,
                    "fin": fin,
                    "numero_capas": len(rangos),
                    "distribucion": etiqueta(rangos),
                    "distribucion_inversa": etiqueta(
                        creciente if sentido == "decreciente" else decreciente
                    ),
                    "representacion": f"DeepONMF_H{len(rangos)}",
                    "dimensiones_repetidas": repetida,
                    "orden_excel": (pareja - 1) * 2
                    + (0 if sentido == "decreciente" else 1),
                }
            )
    plan = pd.DataFrame(filas)
    if len(plan) != 372:
        raise AssertionError("Se esperaban 372 orientaciones")
    return plan

