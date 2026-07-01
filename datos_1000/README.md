# Base de evaluación de 1000 audios

Esta carpeta contiene los 1000 segmentos PCG reales preparados para la demostración completa:

- `N`: 200 señales sanas.
- `AS`: 200 señales con estenosis aórtica.
- `MR`: 200 señales con regurgitación mitral.
- `MS`: 200 señales con estenosis mitral.
- `MVP`: 200 señales con prolapso mitral.

Todos los ficheros son WAV de 2 segundos a 8000 Hz. El script `evaluar_1000_audios.py` recorre esta carpeta y aplica una evaluación leave-one-out: cada audio se elimina de las referencias del modelo antes de clasificarlo.
