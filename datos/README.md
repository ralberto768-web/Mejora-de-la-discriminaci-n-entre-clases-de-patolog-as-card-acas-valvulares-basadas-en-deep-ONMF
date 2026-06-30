# Datos de ejemplo

Esta carpeta contiene cinco audios reales de la base preparada del TFG, todos en formato WAV, a 8000 Hz y con duración de 2 segundos.

Los ficheros se llaman `audio1.wav`, `audio2.wav`, `audio3.wav`, `audio4.wav` y `audio5.wav`. El nombre no indica la clase para que la prueba sea ciega durante la ejecución. Al lanzar `verificar_demo.py`, el script los analiza en orden aleatorio y muestra al final la clase correcta de cada uno.

Se incluye un audio por clase para que el tribunal pueda ejecutar la prueba sin descargar la base completa de 1000 audios. El modelo básico se ha calculado con los 1000 audios preparados en `segmentos_2_0s`: 200 por clase.
