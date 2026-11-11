# Arquitecturas configurables

En «4 · Arquitecturas» puedes elegir una plantilla, editar el JSON o importar un archivo. El editor define la red del próximo entrenamiento; cada entrenamiento empieza con pesos nuevos. Exportar JSON guarda la definición, no los pesos. Los pesos y la arquitectura se guardan juntos automáticamente en resultados/.../biomas_cnn.pth.

«5 · Red neuronal» muestra texto seleccionable con las capas, dimensiones de salida y parámetros. Se actualiza al editar el JSON, cambiar el tamaño o seleccionar biomas. Si hay un modelo cargado, muestra por separado su arquitectura: ese modelo sigue siendo el usado para predicciones hasta cargar o entrenar otro.

Se incluyen ejemplos en arquitecturas/original.json y arquitecturas/compacta.json.

## Formato

Un objeto con exactamente dos propiedades: `name` (nombre) y `layers` (lista ordenada). Cada capa usa `type` y los parámetros de esta tabla. Los canales/características de entrada se deducen automáticamente.

| type | Parámetros |
| --- | --- |
| Conv2d | out_channels, kernel_size; opcionales stride=1, padding=0 |
| BatchNorm2d | Ninguno |
| ReLU | Ninguno |
| MaxPool2d / AvgPool2d | kernel_size; opcional stride (por defecto kernel_size) |
| AdaptiveAvgPool2d | output_size (entero: tamaño cuadrado) |
| Flatten | Ninguno; conserva la dimensión de batch |
| Linear | out_features (entero o "classes") |
| Dropout | p, por ejemplo 0.25 (por defecto PyTorch: 0.5) |

Usa Flatten antes de Linear y termina con Linear, out_features: "classes". Esto adapta la salida al número de biomas seleccionados. No añadas Softmax: CrossEntropyLoss recibe logits y la predicción ya calcula probabilidades.

Ejemplo mínimo:

```json
{
  "name": "CNN sencilla",
  "layers": [
    {"type": "Conv2d", "out_channels": 16, "kernel_size": 3, "padding": 1},
    {"type": "ReLU"},
    {"type": "AdaptiveAvgPool2d", "output_size": 2},
    {"type": "Flatten"},
    {"type": "Linear", "out_features": "classes"}
  ]
}
```

El formato permite redes secuenciales, no conexiones residuales ni ramas. Rechaza tipos desconocidos, parámetros incorrectos y dimensiones incompatibles. No ejecuta Python. Máximos: 64 capas, 20 millones de parámetros y 4 millones de activaciones por imagen/capa. Estos límites no garantizan que cualquier batch quepa en la GPU. BatchNorm2d se restringe a mapas con más de un valor espacial para admitir batches de una imagen.

Los modelos antiguos sin arquitectura declarada se cargan usando la CNN original y sus nombres de pesos originales. Exportar gráficas ahora genera red_neuronal.txt junto a las curvas y matriz de confusión.

## Actualizar

Cierra la app y copia los archivos de esta versión a la carpeta donde tienes app.py, junto a venv312. Incluye architectures.py y conserva tus carpetas de datos, resultados y entorno virtual. No hace falta reinstalar PyTorch ni cambiar CUDA.

Ejecuta como antes: `.\venv312\Scripts\python.exe .\app.py`.
