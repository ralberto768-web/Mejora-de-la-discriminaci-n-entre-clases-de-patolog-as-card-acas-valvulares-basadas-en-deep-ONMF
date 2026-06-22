from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from .audio import construir_matriz_clase, descubrir_audios
from .configuracion import Configuracion
from .estadistica import caracteristicas_por_audio, distancias_figura_7, resumen_auditoria, tabla_2_desde_w
from .evaluacion import (
    caracteristicas_sbv_por_audio,
    dividir_entrenamiento_test,
    evaluar_por_reconstruccion,
    matriz_confusion_df,
    resumen_metricas,
    resumen_particion,
)
from .graficos import figura_5_sbv, figura_7_distancias, figura_11d_tsne, tabla_2_imagen
from .onmf import ResultadoONMF, deep_onmf


PORCENTAJES_ENTRENAMIENTO = (65, 70, 75, 80, 85)


def _siguiente_carpeta_prueba(carpeta_resultados: Path, etiqueta: str) -> Path:
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    numeros = []
    for carpeta in carpeta_resultados.iterdir():
        if not carpeta.is_dir():
            continue
        coincidencia = re.match(r"resultado(\d+)", carpeta.name)
        if coincidencia:
            numeros.append(int(coincidencia.group(1)))
    indice = max(numeros, default=0) + 1

    while True:
        candidata = carpeta_resultados / f"resultado{indice}-{etiqueta}"
        if not candidata.exists():
            candidata.mkdir(parents=True)
            return candidata
        indice += 1


def _guardar_json(datos: object, ruta: Path) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def _separador(titulo: str) -> str:
    linea = "=" * 110
    return f"\n{linea}\n{titulo}\n{linea}\n"


def _tabla_texto(tabla: pd.DataFrame, indice: bool = False) -> str:
    return tabla.to_string(index=indice, justify="center")


def _tabla_capas(resultados_onmf: dict[str, ResultadoONMF]) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    for clase, resultado in resultados_onmf.items():
        for capa in resultado.capas:
            filas.append(
                {
                    "clase": clase,
                    "capa": capa.indice,
                    "rango": capa.rango,
                    "entrada": f"{capa.forma_entrada[0]} x {capa.forma_entrada[1]}",
                    "W": f"{capa.forma_w[0]} x {capa.forma_w[1]}",
                    "H": f"{capa.forma_h[0]} x {capa.forma_h[1]}",
                    "error_relativo": f"{capa.error_relativo:.6f}",
                    "ortogonalidad_media": f"{capa.ortogonalidad_media:.6f}",
                    "segundos": f"{capa.segundos:.2f}",
                }
            )
    return pd.DataFrame(filas)


def _tabla_errores(resultados_onmf: dict[str, ResultadoONMF]) -> pd.DataFrame:
    filas = []
    for clase, resultado in resultados_onmf.items():
        filas.append(
            {
                "clase": clase,
                "error_relativo_final": f"{resultado.error_relativo_final:.6f}",
                "capas": len(resultado.capas),
            }
        )
    return pd.DataFrame(filas)


def _preparar_tabla_2(tabla_2: pd.DataFrame) -> pd.DataFrame:
    tabla = tabla_2.copy()
    tabla["p-valor"] = tabla["p-valor"].map(lambda valor: f"{valor:.3e}")
    return tabla


def _accuracy(predicciones: pd.DataFrame) -> float:
    return float(predicciones["correcto"].mean())


def _resumen_global_metricas(
    pred_entrenamiento: pd.DataFrame,
    pred_test: pd.DataFrame,
    metricas_entrenamiento: pd.DataFrame,
    metricas_test: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "particion": "entrenamiento",
                "muestras": len(pred_entrenamiento),
                "accuracy": _accuracy(pred_entrenamiento),
                "macro_f1": float(metricas_entrenamiento.loc[metricas_entrenamiento["clase"] == "TOTAL", "f1"].iloc[0]),
            },
            {
                "particion": "test",
                "muestras": len(pred_test),
                "accuracy": _accuracy(pred_test),
                "macro_f1": float(metricas_test.loc[metricas_test["clase"] == "TOTAL", "f1"].iloc[0]),
            },
        ]
    )


def _crear_informe(
    nombre_prueba: str,
    porcentaje_entrenamiento: int,
    porcentaje_test: int,
    carpeta_prueba: Path,
    carpeta_tablas: Path,
    carpeta_tecnica: Path,
    segundos: float,
    configuracion: Configuracion,
    auditoria_total: pd.DataFrame,
    particion: pd.DataFrame,
    tabla_2: pd.DataFrame,
    distancias: dict[str, pd.DataFrame],
    resultados_onmf: dict[str, ResultadoONMF],
    metricas_globales: pd.DataFrame,
    metricas_entrenamiento: pd.DataFrame,
    metricas_test: pd.DataFrame,
    matriz_confusion_test: pd.DataFrame,
) -> str:
    tabla_2_mostrar = _preparar_tabla_2(tabla_2)
    mejor = tabla_2.sort_values("p-valor").iloc[0]

    lineas = [
        f"INFORME DE RESULTADOS - {nombre_prueba}",
        "",
        f"Prueba ejecutada: {porcentaje_entrenamiento}% entrenamiento / {porcentaje_test}% test.",
        "Implementación en Python del método deep ONMF aplicado a señales PCG.",
        "",
        f"Carpeta de salida: {carpeta_prueba}",
        f"Carpeta de tablas y figuras: {carpeta_tablas}",
        f"Carpeta de documentación técnica: {carpeta_tecnica}",
        f"Tiempo total de ejecución: {segundos:.2f} segundos",
        "",
        "IMPORTANTE:",
        "En esta versión no se descarta ningún WAV. Si un audio dura menos de 2 segundos,",
        "se rellena con ceros hasta 2 segundos para poder calcular el espectrograma.",
        "Esta decisión se toma porque la base disponible es la que se va a usar en el TFG.",
    ]

    lineas.append(_separador("1. ORGANIZACIÓN DE LA CARPETA"))
    lineas.extend(
        [
            f"{nombre_prueba}/",
            "  00_RESULTADOS_Y_EXPLICACION.txt",
            "  tablas_y_figuras/",
            f"    {nombre_prueba}_Figura_5_SBV_por_clase.png",
            f"    {nombre_prueba}_Tabla_2_estadistica_SBV.png",
            f"    {nombre_prueba}_Tabla_2_estadistica_SBV.csv",
            f"    {nombre_prueba}_Figura_7_distancias_euclideas.png",
            f"    {nombre_prueba}_Figura_11D_tSNE_deep_ONMF.png",
            "  documentacion_tecnica/",
            "    predicciones_test.csv",
            "    metricas_test.csv",
            "    matriz_confusion_test.csv",
            "    caracteristicas_sbv_por_audio.csv",
            "    parametros_configuracion.json",
        ]
    )

    lineas.append(_separador("2. AUDITORÍA TOTAL DE LA BASE DE DATOS"))
    lineas.extend(
        [
            "La columna 'audios_cortos_menores_2s' indica cuántos audios reales duran menos de 2 segundos.",
            "Esos audios se han usado igualmente, rellenándolos con ceros hasta alcanzar 2 segundos.",
            "",
            _tabla_texto(auditoria_total),
        ]
    )

    lineas.append(_separador("3. PARTICIÓN ENTRENAMIENTO / TEST"))
    lineas.extend(
        [
            "La división es estratificada: cada clase mantiene el mismo porcentaje de entrenamiento y test.",
            "",
            _tabla_texto(particion),
        ]
    )

    lineas.append(_separador("4. PARÁMETROS DEL MÉTODO"))
    lineas.extend(
        [
            f"Frecuencia esperada de muestreo: {configuracion.frecuencia_esperada_hz} Hz",
            f"Trama PCG: {configuracion.duracion_trama_s:.1f} s",
            f"Solape entre tramas PCG: {configuracion.solape_trama_s:.1f} s",
            f"Ventana STFT: Hamming de {configuracion.longitud_ventana} muestras",
            f"Salto STFT: {configuracion.salto_ventana} muestras",
            f"FFT: {configuracion.puntos_fft} puntos",
            f"Bins de frecuencia: {configuracion.bins_frecuencia}",
            f"Rangos deep ONMF: {configuracion.rangos_onmf}",
            f"Iteraciones por capa: {configuracion.iteraciones_onmf}",
            f"Penalización de ortogonalidad sobre H: {configuracion.penalizacion_ortogonal}",
            f"Semilla aleatoria base: {configuracion.semilla}",
            "",
            "Entrenamiento del modelo:",
            "Para evitar fuga de información, las matrices W de deep ONMF se aprenden solo con los audios",
            "de entrenamiento. El test se evalúa después, proyectándolo sobre las bases aprendidas.",
        ]
    )

    lineas.append(_separador("5. RESULTADOS DE LA FACTORIZACIÓN DEEP ONMF"))
    lineas.extend(
        [
            "Errores relativos finales por clase, calculados sobre las matrices de entrenamiento:",
            "",
            _tabla_texto(_tabla_errores(resultados_onmf)),
            "",
            "Detalle por capa:",
            "",
            _tabla_texto(_tabla_capas(resultados_onmf)),
        ]
    )

    lineas.append(_separador("6. RESULTADOS DE CLASIFICACIÓN"))
    lineas.extend(
        [
            "La clasificación se hace por error de reconstrucción: un audio se asigna a la clase cuya",
            "base W reconstruye mejor su espectrograma. El test no participa en el aprendizaje de W.",
            "",
            "Resumen global:",
            "",
            _tabla_texto(metricas_globales),
            "",
            "Métricas por clase en entrenamiento:",
            "",
            _tabla_texto(metricas_entrenamiento),
            "",
            "Métricas por clase en test:",
            "",
            _tabla_texto(metricas_test),
            "",
            "Matriz de confusión en test:",
            "",
            _tabla_texto(matriz_confusion_test, indice=True),
        ]
    )

    lineas.append(_separador("7. TABLA 2 - ESTADÍSTICA DE SBV 1 A SBV 7"))
    lineas.extend(
        [
            "La Tabla 2 se calcula usando las bases W aprendidas en entrenamiento.",
            "",
            _tabla_texto(tabla_2_mostrar),
            "",
            f"SBV con menor p-valor: {mejor['Número de característica']} con p = {mejor['p-valor']:.3e}.",
        ]
    )

    lineas.append(_separador("8. FIGURA 5 - SBV POR CLASE"))
    lineas.extend(
        [
            f"Archivo: tablas_y_figuras/{nombre_prueba}_Figura_5_SBV_por_clase.png",
            "Representa los cinco primeros SBV de cada clase aprendidos con entrenamiento.",
        ]
    )

    lineas.append(_separador("9. FIGURA 7 - DISTANCIAS EUCLÍDEAS"))
    lineas.extend(
        [
            f"Archivo: tablas_y_figuras/{nombre_prueba}_Figura_7_distancias_euclideas.png",
            "Las distancias se calculan sobre las características SBV proyectadas por audio.",
            "",
            "Datos numéricos usados en la Figura 7:",
        ]
    )
    for clave, tabla in distancias.items():
        lineas.extend(["", clave, _tabla_texto(tabla)])

    lineas.append(_separador("10. FIGURA 11D - t-SNE DEEP ONMF"))
    lineas.extend(
        [
            f"Archivo: tablas_y_figuras/{nombre_prueba}_Figura_11D_tSNE_deep_ONMF.png",
            "El t-SNE se calcula con las siete características SBV de cada audio, incluyendo",
            "entrenamiento y test, para visualizar la separación de clases.",
        ]
    )

    lineas.append(_separador("11. CONCLUSIÓN DE ESTA PRUEBA"))
    accuracy_test = _accuracy_from_global(metricas_globales, "test")
    macro_f1_test = _macro_f1_from_global(metricas_globales, "test")
    lineas.extend(
        [
            f"Resultado test para {porcentaje_entrenamiento}/{porcentaje_test}:",
            f"- Accuracy test: {accuracy_test:.4f}",
            f"- Macro F1 test: {macro_f1_test:.4f}",
            "",
            "Estos valores son los que deben compararse con el resto de particiones para decidir",
            "qué reparto entrenamiento/test se comporta mejor con esta base de datos.",
        ]
    )

    return "\n".join(lineas)


def _accuracy_from_global(metricas_globales: pd.DataFrame, particion: str) -> float:
    return float(metricas_globales.loc[metricas_globales["particion"] == particion, "accuracy"].iloc[0])


def _macro_f1_from_global(metricas_globales: pd.DataFrame, particion: str) -> float:
    return float(metricas_globales.loc[metricas_globales["particion"] == particion, "macro_f1"].iloc[0])


def ejecutar_prueba_train_test(
    configuracion: Configuracion,
    registros: list,
    porcentaje_entrenamiento: int,
) -> Path:
    porcentaje_test = 100 - porcentaje_entrenamiento
    etiqueta = f"{porcentaje_entrenamiento}_{porcentaje_test}_entrenamiento_test"
    inicio_total = time.perf_counter()
    carpeta_prueba = _siguiente_carpeta_prueba(configuracion.carpeta_resultados, etiqueta)
    nombre_prueba = carpeta_prueba.name
    carpeta_tablas = carpeta_prueba / "tablas_y_figuras"
    carpeta_tecnica = carpeta_prueba / "documentacion_tecnica"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)
    carpeta_tecnica.mkdir(parents=True, exist_ok=True)

    print(f"Carpeta de salida: {carpeta_prueba}")
    print(f"Partición: {porcentaje_entrenamiento}% entrenamiento / {porcentaje_test}% test")

    division = dividir_entrenamiento_test(
        registros,
        configuracion.clases,
        porcentaje_entrenamiento=porcentaje_entrenamiento,
        semilla=configuracion.semilla,
    )

    datos_por_clase = {}
    resultados_onmf: dict[str, ResultadoONMF] = {}
    w_por_clase = {}
    capas_json = {}

    for posicion, clase in enumerate(configuracion.clases, start=1):
        print(f"[{posicion}/{len(configuracion.clases)}] Entrenando clase {clase}")
        datos = construir_matriz_clase(clase, division.entrenamiento, configuracion)
        datos_por_clase[clase] = datos
        resultado = deep_onmf(
            datos.matriz,
            rangos=configuracion.rangos_onmf,
            iteraciones=configuracion.iteraciones_onmf,
            penalizacion_ortogonal=configuracion.penalizacion_ortogonal,
            semilla=configuracion.semilla + porcentaje_entrenamiento * 100 + posicion,
        )
        resultados_onmf[clase] = resultado
        w_por_clase[clase] = resultado.w_final
        capas_json[clase] = [capa.__dict__ for capa in resultado.capas]

    auditoria_total = resumen_auditoria(registros, {
        clase: construir_matriz_clase(clase, registros, configuracion) for clase in configuracion.clases
    }, configuracion.clases)
    particion = resumen_particion(
        registros,
        division.entrenamiento,
        division.test,
        configuracion.clases,
        configuracion.duracion_trama_s,
    )

    print("Calculando características SBV por audio")
    caracteristicas_entrenamiento = caracteristicas_sbv_por_audio(
        division.entrenamiento, w_por_clase, configuracion, "entrenamiento"
    )
    caracteristicas_test = caracteristicas_sbv_por_audio(division.test, w_por_clase, configuracion, "test")
    caracteristicas = pd.concat([caracteristicas_entrenamiento, caracteristicas_test], ignore_index=True)

    print("Evaluando entrenamiento y test")
    pred_entrenamiento = evaluar_por_reconstruccion(
        division.entrenamiento, w_por_clase, configuracion, "entrenamiento"
    )
    pred_test = evaluar_por_reconstruccion(division.test, w_por_clase, configuracion, "test")
    metricas_entrenamiento = resumen_metricas(pred_entrenamiento, configuracion.clases)
    metricas_test = resumen_metricas(pred_test, configuracion.clases)
    metricas_globales = _resumen_global_metricas(
        pred_entrenamiento,
        pred_test,
        metricas_entrenamiento,
        metricas_test,
    )
    matriz_confusion_test = matriz_confusion_df(pred_test, configuracion.clases)

    tabla_2 = tabla_2_desde_w(w_por_clase, configuracion.clases)
    distancias = distancias_figura_7(caracteristicas, configuracion.clases)

    _guardar_json(
        {
            **configuracion.como_diccionario(),
            "porcentaje_entrenamiento": porcentaje_entrenamiento,
            "porcentaje_test": porcentaje_test,
            "politica_audios_menores_2s": "relleno_con_ceros_hasta_2s",
        },
        carpeta_tecnica / "parametros_configuracion.json",
    )
    _guardar_json(capas_json, carpeta_tecnica / "detalle_capas_onmf.json")

    auditoria_total.to_csv(carpeta_tecnica / "auditoria_total_base_datos.csv", index=False, encoding="utf-8-sig")
    particion.to_csv(carpeta_tecnica / "particion_entrenamiento_test.csv", index=False, encoding="utf-8-sig")
    caracteristicas.to_csv(carpeta_tecnica / "caracteristicas_sbv_por_audio.csv", index=False, encoding="utf-8-sig")
    pred_entrenamiento.to_csv(carpeta_tecnica / "predicciones_entrenamiento.csv", index=False, encoding="utf-8-sig")
    pred_test.to_csv(carpeta_tecnica / "predicciones_test.csv", index=False, encoding="utf-8-sig")
    metricas_entrenamiento.to_csv(carpeta_tecnica / "metricas_entrenamiento.csv", index=False, encoding="utf-8-sig")
    metricas_test.to_csv(carpeta_tecnica / "metricas_test.csv", index=False, encoding="utf-8-sig")
    metricas_globales.to_csv(carpeta_tecnica / "metricas_globales.csv", index=False, encoding="utf-8-sig")
    matriz_confusion_test.to_csv(carpeta_tecnica / "matriz_confusion_test.csv", encoding="utf-8-sig")

    for nombre, tabla in distancias.items():
        tabla.to_csv(carpeta_tecnica / f"{nombre}.csv", index=False, encoding="utf-8-sig")

    np.savez(
        carpeta_tecnica / "matrices_w_finales_por_clase.npz",
        **{f"W_{clase}": matriz for clase, matriz in w_por_clase.items()},
    )

    tabla_2.to_csv(
        carpeta_tablas / f"{nombre_prueba}_Tabla_2_estadistica_SBV.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figura_5_sbv(
        w_por_clase,
        configuracion.clases,
        configuracion.frecuencia_esperada_hz,
        carpeta_tablas / f"{nombre_prueba}_Figura_5_SBV_por_clase.png",
    )
    tabla_2_imagen(tabla_2, carpeta_tablas / f"{nombre_prueba}_Tabla_2_estadistica_SBV.png")
    figura_7_distancias(distancias, carpeta_tablas / f"{nombre_prueba}_Figura_7_distancias_euclideas.png")
    figura_11d_tsne(
        caracteristicas,
        configuracion.clases,
        carpeta_tablas / f"{nombre_prueba}_Figura_11D_tSNE_deep_ONMF.png",
        carpeta_tecnica / "coordenadas_figura_11D_tSNE.csv",
    )

    segundos = time.perf_counter() - inicio_total
    informe = _crear_informe(
        nombre_prueba=nombre_prueba,
        porcentaje_entrenamiento=porcentaje_entrenamiento,
        porcentaje_test=porcentaje_test,
        carpeta_prueba=carpeta_prueba,
        carpeta_tablas=carpeta_tablas,
        carpeta_tecnica=carpeta_tecnica,
        segundos=segundos,
        configuracion=configuracion,
        auditoria_total=auditoria_total,
        particion=particion,
        tabla_2=tabla_2,
        distancias=distancias,
        resultados_onmf=resultados_onmf,
        metricas_globales=metricas_globales,
        metricas_entrenamiento=metricas_entrenamiento,
        metricas_test=metricas_test,
        matriz_confusion_test=matriz_confusion_test,
    )
    (carpeta_prueba / "00_RESULTADOS_Y_EXPLICACION.txt").write_text(informe, encoding="utf-8-sig")

    return carpeta_prueba


def ejecutar_pruebas_train_test(configuracion: Configuracion) -> list[Path]:
    registros = descubrir_audios(configuracion.carpeta_base_datos, configuracion.clases)
    if not registros:
        raise RuntimeError("No se han encontrado audios WAV en la base de datos.")

    carpetas = []
    for porcentaje in PORCENTAJES_ENTRENAMIENTO:
        carpetas.append(ejecutar_prueba_train_test(configuracion, registros, porcentaje))
    return carpetas


def ejecutar_prueba(configuracion: Configuracion) -> Path:
    return ejecutar_pruebas_train_test(configuracion)[-1]


def _crear_informe_articulo(
    nombre_prueba: str,
    carpeta_prueba: Path,
    carpeta_tablas: Path,
    carpeta_tecnica: Path,
    segundos: float,
    configuracion: Configuracion,
    auditoria: pd.DataFrame,
    tabla_2: pd.DataFrame,
    distancias: dict[str, pd.DataFrame],
    resultados_onmf: dict[str, ResultadoONMF],
) -> str:
    tabla_2_mostrar = _preparar_tabla_2(tabla_2)
    mejor = tabla_2.sort_values("p-valor").iloc[0]
    duracion_trama = f"{configuracion.duracion_trama_s:g}"
    solape_trama = f"{configuracion.solape_trama_s:g}"
    if configuracion.rellenar_audios_cortos:
        criterio_audios = (
            f"- No se descarta ningun audio: los audios menores de {duracion_trama} segundos "
            "se rellenan con ceros."
        )
        texto_auditoria = [
            "En esta ejecucion no se elimina ningun WAV por duracion.",
            f"Si un audio dura menos de {duracion_trama} segundos, se rellena con ceros hasta formar una trama completa.",
            "Esto permite usar la base completa disponible en el TFG.",
        ]
    else:
        criterio_audios = f"- Descarte de audios que no alcanzan una trama completa de {duracion_trama} segundos."
        texto_auditoria = [
            f"Aqui se aplica el criterio configurado: si un audio dura menos de {duracion_trama} segundos,",
            "no entra en la matriz X de su clase.",
        ]

    lineas = [
        f"INFORME MODO ARTICULO - {nombre_prueba}",
        "",
        "Esta ejecucion sigue el procedimiento deep-ONMF descrito en el articulo objetivo:",
        "- Uso de toda la base Yaseen disponible.",
        f"- Tramas PCG de {duracion_trama} segundos con {solape_trama} segundos de solape.",
        criterio_audios,
        "- STFT con ventana Hamming de 150 muestras, salto 75 y FFT de 250 puntos.",
        "- Deep ONMF con tres capas y rangos 9, 8 y 7.",
        "",
        f"Carpeta de salida: {carpeta_prueba}",
        f"Carpeta de tablas y figuras: {carpeta_tablas}",
        f"Carpeta de documentación técnica: {carpeta_tecnica}",
        f"Tiempo total de ejecución: {segundos:.2f} segundos",
    ]

    lineas.append(_separador("1. AUDITORÍA DE DATOS USADOS"))
    lineas.extend(
        texto_auditoria
        + [
            "",
            _tabla_texto(auditoria),
        ]
    )

    lineas.append(_separador("2. PARÁMETROS DEL ARTÍCULO"))
    lineas.extend(
        [
            f"Frecuencia esperada de muestreo: {configuracion.frecuencia_esperada_hz} Hz",
            f"Trama PCG: {configuracion.duracion_trama_s:.1f} s",
            f"Solape entre tramas PCG: {configuracion.solape_trama_s:.1f} s",
            f"Ventana STFT: Hamming de {configuracion.longitud_ventana} muestras",
            f"Salto STFT: {configuracion.salto_ventana} muestras",
            f"FFT: {configuracion.puntos_fft} puntos",
            f"Bins de frecuencia: {configuracion.bins_frecuencia}",
            f"Rangos deep ONMF: {configuracion.rangos_onmf}",
            f"Iteraciones por capa en esta implementación: {configuracion.iteraciones_onmf}",
            f"Rellenar audios cortos: {configuracion.rellenar_audios_cortos}",
        ]
    )

    lineas.append(_separador("3. RESULTADOS DEEP ONMF"))
    lineas.extend(
        [
            "Errores relativos de reconstrucción por clase:",
            "",
            _tabla_texto(_tabla_errores(resultados_onmf)),
            "",
            "Detalle por capa:",
            "",
            _tabla_texto(_tabla_capas(resultados_onmf)),
        ]
    )

    lineas.append(_separador("4. TABLA 2"))
    lineas.extend(
        [
            "Tabla 2 calculada con los SBV obtenidos desde las matrices W finales.",
            "",
            _tabla_texto(tabla_2_mostrar),
            "",
            f"SBV con menor p-valor: {mejor['Número de característica']} con p = {mejor['p-valor']:.3e}.",
        ]
    )

    lineas.append(_separador("5. FIGURA 5, FIGURA 7 Y FIGURA 11D"))
    lineas.extend(
        [
            f"Figura 5: tablas_y_figuras/{nombre_prueba}_Figura_5_SBV_por_clase.png",
            f"Tabla 2: tablas_y_figuras/{nombre_prueba}_Tabla_2_estadistica_SBV.png",
            f"Figura 7: tablas_y_figuras/{nombre_prueba}_Figura_7_distancias_euclideas.png",
            f"Figura 11D: tablas_y_figuras/{nombre_prueba}_Figura_11D_tSNE_deep_ONMF.png",
            "",
            "Datos numéricos usados en la Figura 7:",
        ]
    )
    for clave, tabla in distancias.items():
        lineas.extend(["", clave, _tabla_texto(tabla)])

    lineas.append(_separador("6. LECTURA HONESTA DEL RESULTADO"))
    lineas.extend(
        [
            "Esta ejecución reproduce la lógica experimental del documento, no una clasificación train/test.",
            "Por tanto sus resultados principales son p-valores, distancias y visualización de separabilidad.",
            "No debe compararse directamente con accuracy de test, porque el artículo no obtiene sus valores",
            "principales con esa evaluación.",
        ]
    )

    return "\n".join(lineas)


def _ejecutar_articulo_deep_onmf(
    configuracion: Configuracion,
    etiqueta: str,
    modo_json: str,
    mensaje_modo: str,
) -> Path:
    inicio_total = time.perf_counter()
    carpeta_prueba = _siguiente_carpeta_prueba(configuracion.carpeta_resultados, etiqueta)
    nombre_prueba = carpeta_prueba.name
    carpeta_tablas = carpeta_prueba / "tablas_y_figuras"
    carpeta_tecnica = carpeta_prueba / "documentacion_tecnica"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)
    carpeta_tecnica.mkdir(parents=True, exist_ok=True)

    print(f"Carpeta de salida: {carpeta_prueba}")
    print(mensaje_modo)

    registros = descubrir_audios(configuracion.carpeta_base_datos, configuracion.clases)
    if not registros:
        raise RuntimeError("No se han encontrado audios WAV en la base de datos.")

    datos_por_clase = {}
    resultados_onmf: dict[str, ResultadoONMF] = {}
    w_por_clase = {}
    h_por_clase = {}
    capas_json = {}

    for posicion, clase in enumerate(configuracion.clases, start=1):
        print(f"[{posicion}/{len(configuracion.clases)}] Deep ONMF clase {clase}")
        datos = construir_matriz_clase(clase, registros, configuracion)
        datos_por_clase[clase] = datos
        resultado = deep_onmf(
            datos.matriz,
            rangos=configuracion.rangos_onmf,
            iteraciones=configuracion.iteraciones_onmf,
            penalizacion_ortogonal=configuracion.penalizacion_ortogonal,
            semilla=configuracion.semilla + posicion * 100,
        )
        resultados_onmf[clase] = resultado
        w_por_clase[clase] = resultado.w_final
        h_por_clase[clase] = resultado.h_final
        capas_json[clase] = [capa.__dict__ for capa in resultado.capas]

    auditoria = resumen_auditoria(registros, datos_por_clase, configuracion.clases)
    caracteristicas = caracteristicas_por_audio(datos_por_clase, h_por_clase)
    tabla_2 = tabla_2_desde_w(w_por_clase, configuracion.clases)
    distancias = distancias_figura_7(caracteristicas, configuracion.clases)

    _guardar_json(
        {**configuracion.como_diccionario(), "modo": modo_json},
        carpeta_tecnica / "parametros_configuracion.json",
    )
    _guardar_json(capas_json, carpeta_tecnica / "detalle_capas_onmf.json")
    auditoria.to_csv(carpeta_tecnica / "auditoria_base_datos.csv", index=False, encoding="utf-8-sig")
    caracteristicas.to_csv(carpeta_tecnica / "caracteristicas_sbv_por_audio.csv", index=False, encoding="utf-8-sig")
    for nombre, tabla in distancias.items():
        tabla.to_csv(carpeta_tecnica / f"{nombre}.csv", index=False, encoding="utf-8-sig")

    np.savez(
        carpeta_tecnica / "matrices_w_finales_por_clase.npz",
        **{f"W_{clase}": matriz for clase, matriz in w_por_clase.items()},
    )

    tabla_2.to_csv(
        carpeta_tablas / f"{nombre_prueba}_Tabla_2_estadistica_SBV.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figura_5_sbv(
        w_por_clase,
        configuracion.clases,
        configuracion.frecuencia_esperada_hz,
        carpeta_tablas / f"{nombre_prueba}_Figura_5_SBV_por_clase.png",
    )
    tabla_2_imagen(tabla_2, carpeta_tablas / f"{nombre_prueba}_Tabla_2_estadistica_SBV.png")
    figura_7_distancias(distancias, carpeta_tablas / f"{nombre_prueba}_Figura_7_distancias_euclideas.png")
    figura_11d_tsne(
        caracteristicas,
        configuracion.clases,
        carpeta_tablas / f"{nombre_prueba}_Figura_11D_tSNE_deep_ONMF.png",
        carpeta_tecnica / "coordenadas_figura_11D_tSNE.csv",
    )

    segundos = time.perf_counter() - inicio_total
    informe = _crear_informe_articulo(
        nombre_prueba=nombre_prueba,
        carpeta_prueba=carpeta_prueba,
        carpeta_tablas=carpeta_tablas,
        carpeta_tecnica=carpeta_tecnica,
        segundos=segundos,
        configuracion=configuracion,
        auditoria=auditoria,
        tabla_2=tabla_2,
        distancias=distancias,
        resultados_onmf=resultados_onmf,
    )
    (carpeta_prueba / "00_RESULTADOS_Y_EXPLICACION.txt").write_text(informe, encoding="utf-8-sig")

    return carpeta_prueba


def ejecutar_articulo_original(configuracion: Configuracion) -> Path:
    cfg = replace(configuracion, rellenar_audios_cortos=False)
    return _ejecutar_articulo_deep_onmf(
        cfg,
        etiqueta="articulo_original",
        modo_json="articulo_original_descarta_menores_2s",
        mensaje_modo="Modo articulo original: se descartan audios menores de 2 segundos.",
    )


def ejecutar_articulo_sin_descartar(configuracion: Configuracion) -> Path:
    cfg = replace(configuracion, rellenar_audios_cortos=True)
    return _ejecutar_articulo_deep_onmf(
        cfg,
        etiqueta="deep_onmf_sin_descartar_menores_2s",
        modo_json="deep_onmf_sin_descartar_menores_2s",
        mensaje_modo=(
            "Modo deep-ONMF sin descartar: los audios menores de 2 segundos "
            "se rellenan con ceros hasta 2 segundos."
        ),
    )
