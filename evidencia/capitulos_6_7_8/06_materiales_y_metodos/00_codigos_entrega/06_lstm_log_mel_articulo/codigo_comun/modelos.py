from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResultadoEntrenamiento:
    predicciones: np.ndarray
    reales: np.ndarray
    perdida_final: float


def comprobar_torch():
    """Carga PyTorch solo cuando hace falta entrenar redes neuronales."""

    try:
        import torch
        import torch.nn as nn
        import torch.utils.data as data
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se ha encontrado PyTorch. Instala dependencias con: "
            "python -m pip install -r requisitos.txt"
        ) from exc
    return torch, nn, data


def entrenar_modelo(
    tipo_modelo: str,
    x: np.ndarray,
    y: np.ndarray,
    indices_entrenamiento: np.ndarray,
    indices_prueba: np.ndarray,
    clases: list[str],
    epocas: int,
    tamano_lote: int,
    tasa_aprendizaje: float,
    semilla: int,
    ruta_modelo: Path,
) -> ResultadoEntrenamiento:
    """Entrena una CNN o LSTM con PyTorch y devuelve predicciones de prueba."""

    torch, nn, data = comprobar_torch()
    torch.manual_seed(semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_ent = torch.tensor(x[indices_entrenamiento], dtype=torch.float32)
    y_ent = torch.tensor(y[indices_entrenamiento], dtype=torch.long)
    x_pru = torch.tensor(x[indices_prueba], dtype=torch.float32)
    y_pru = torch.tensor(y[indices_prueba], dtype=torch.long)

    if tipo_modelo == "cnn":
        x_ent = x_ent.unsqueeze(1)
        x_pru = x_pru.unsqueeze(1)
        modelo = crear_red_convolucional(nn, numero_clases=len(clases))
    elif tipo_modelo == "lstm":
        x_ent = x_ent.transpose(1, 2)
        x_pru = x_pru.transpose(1, 2)
        modelo = crear_red_lstm(nn, entrada=x_ent.shape[2], numero_clases=len(clases))
    else:
        raise ValueError(f"Modelo no reconocido: {tipo_modelo}")

    modelo = modelo.to(dispositivo)
    pesos = _pesos_clase(y_ent, len(clases), torch).to(dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    conjunto = data.TensorDataset(x_ent, y_ent)
    cargador = data.DataLoader(conjunto, batch_size=tamano_lote, shuffle=True)

    perdida_final = 0.0
    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_acumulada = 0.0
        for lote_x, lote_y in cargador:
            lote_x = lote_x.to(dispositivo)
            lote_y = lote_y.to(dispositivo)
            optimizador.zero_grad()
            salida = modelo(lote_x)
            perdida = criterio(salida, lote_y)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += float(perdida.item())
        perdida_final = perdida_acumulada / max(1, len(cargador))
        print(f"Epoca {epoca:03d}/{epocas:03d} - perdida media: {perdida_final:.6f}")

    modelo.eval()
    with torch.no_grad():
        logits = modelo(x_pru.to(dispositivo))
        predicciones = torch.argmax(logits, dim=1).cpu().numpy()

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tipo_modelo": tipo_modelo,
            "estado": modelo.state_dict(),
            "clases": clases,
            "forma_entrada": list(x.shape[1:]),
        },
        ruta_modelo,
    )
    return ResultadoEntrenamiento(
        predicciones=predicciones,
        reales=y[indices_prueba],
        perdida_final=perdida_final,
    )


def _pesos_clase(y_entrenamiento, numero_clases: int, torch):
    conteos = torch.bincount(y_entrenamiento, minlength=numero_clases).float()
    pesos = 1.0 / torch.clamp(conteos, min=1.0)
    return pesos / pesos.mean()


def crear_red_convolucional(nn, numero_clases: int):
    """CNN inspirada en la arquitectura MATLAB MyNet del repositorio original."""

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 100),
        nn.ReLU(),
        nn.Linear(100, numero_clases),
    )


def crear_red_lstm(nn, entrada: int, numero_clases: int):
    """LSTM de dos capas, siguiendo el planteamiento del articulo."""

    class RedLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=entrada, hidden_size=100, num_layers=2, batch_first=True)
            self.clasificador = nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, numero_clases),
            )

        def forward(self, x):
            salida, _ = self.lstm(x)
            ultima_salida = salida[:, -1, :]
            return self.clasificador(ultima_salida)

    return RedLSTM()
