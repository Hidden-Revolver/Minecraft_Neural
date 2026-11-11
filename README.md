# Minecraft Neural

Aplicación de escritorio para clasificar biomas de Minecraft con Python,
Tkinter y PyTorch. Entrena en el ordenador del usuario con CPU o CUDA NVIDIA;
permite configurar arquitecturas JSON, comparar métricas y guardar modelos.

## Empezar en Windows

Requisitos: Python **3.12 de 64 bits** (también 3.11–3.13), pip y Tkinter.
Para CUDA necesitas una GPU NVIDIA y su controlador actualizado. No hace falta
instalar el CUDA Toolkit por separado. Python, Tkinter del sistema y el driver
se instalan manualmente; el resto lo prepara el instalador.

Desde la raíz del repositorio:

```powershell
py -3.12 setup.py
```

La primera vez prepara el dataset, crea `.venv`, instala las dependencias,
verifica PyTorch y abre la aplicación. La descarga total puede ser de varios GB.
Las siguientes ejecuciones verifican y reutilizan los datos y el entorno.
El dataset necesita al menos 1 GiB libre durante su preparación.

En **Explorar carpeta raíz…** se propone el dataset ya preparado. Marca entre
5 y 10 biomas y configura el entrenamiento. Para usar una red ya entrenada,
pulsa **Cargar modelo / resultados…** y selecciona un `.pth` en
`minecraft_biomas/resultados/`.

En Linux/macOS usa `python3 setup.py`. En Fedora, Tkinter requiere
`sudo dnf install python3-tkinter`; en Ubuntu/Debian, `sudo apt install python3-tk`.
En macOS la aplicación usa CPU. CUDA requiere Windows o Linux y NVIDIA.

## Opciones del instalador

| Opción tras `py -3.12 setup.py` | Uso |
| --- | --- |
| `--cpu` | Instalar y ejecutar con PyTorch CPU. |
| `--cuda` | Exigir CUDA; informar si no está disponible. |
| `--install-only` | Preparar datos y dependencias sin abrir la ventana. |
| `--dataset-only` | Preparar solo los datos, sin PyTorch ni interfaz. |
| `--dataset-source "C:/Descargas/minecraft-biomes.zip"` | Importar un ZIP local en lugar de descargarlo. |
| `--dataset-source "C:/Datos/preprocessed_data"` | Copiar y renombrar una carpeta local, conservando el origen. |
| `--skip-dataset` | Trabajar con otros datos o solo cargar modelos, sin preparar el dataset. |
| `--reinstall` | Reinstalar dependencias en `.venv`. |

Se pueden combinar opciones, por ejemplo `--dataset-only --dataset-source RUTA`.
Si instalas con `--cpu`, úsalo de nuevo al abrir para mantener esa distribución.

## Dataset reproducible

Fuente: [Minecraft Biomes de Willow C en Kaggle](https://www.kaggle.com/datasets/willowc/minecraft-biomes),
versión 1, licencia CC0. Contiene **22.169 imágenes y 41 biomas**.
La descarga pública se ha probado sin credenciales. Si Kaggle rechaza la descarga,
puedes obtener el ZIP desde su web e importarlo con `--dataset-source`.

El instalador transforma `preprocessed_data/biome_ID` en
`minecraft_biomas/minecraft_biomas/nombre_del_bioma`, conservando exactamente los
nombres publicados anteriormente, incluido `Plains`. No cambia las imágenes ni
sus nombres de archivo. La correspondencia, recuentos y huellas SHA-256 están en
[`scripts/dataset_manifest.json`](scripts/dataset_manifest.json).

La preparación se hace en una carpeta temporal y se valida antes de colocar el
resultado en su destino. Si hay un dataset existente incompleto o modificado,
**no se sobrescribe ni se borra**: muévelo a otra carpeta para regenerarlo, o usa
`--skip-dataset` para seguir trabajando con él. La misma opción permite abrir la
app sin conexión cuando se usan datos propios.

## Documentación y estructura

- [Manual de uso](docs/USO.md): datos, entrenamiento, validación, gráficas y predicción.
- [Arquitecturas JSON](docs/ARQUITECTURAS.md): formato de capas y ejemplos.
- `setup.py`: entrada única para instalar y ejecutar.
- `scripts/`: preparación del dataset, sin dependencias de PyTorch.
- `minecraft_biomas/app.py`: interfaz; `core.py`: datos y entrenamiento;
  `architectures.py`: construcción de redes; `plots.py`: gráficas.
- `tests/`: pruebas de datos, arquitecturas, entrenamiento e instalador.
- `minecraft_biomas/resultados/`: experimentos históricos publicados para ver
  la evolución y cargar redes entrenadas. `archivo_OutPut/` conserva las salidas antiguas.
- `OutPut/`: nuevos entrenamientos y destino inicial para exportar gráficas;
  Git solo guarda `.gitkeep`. El resto son archivos personales ignorados.
- `.venv/` y `minecraft_biomas/minecraft_biomas/`: dependencias y datos locales,
  generados por el instalador y excluidos de Git.

`setup.py` es un instalador/lanzador, no un archivo de empaquetado setuptools.
No se instala el proyecto con `pip install .`.

## Desarrollo y pruebas

```powershell
py -3.12 setup.py --cpu --skip-dataset --install-only
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

En Linux sustituye `.\.venv\Scripts\python.exe` por `.venv/bin/python`.
Se utiliza `pytest` porque recoge tanto las pruebas `unittest` como las funciones
de prueba de arquitecturas. Las pruebas sintéticas de CPU no miden precisión
sobre Minecraft ni validan el rendimiento CUDA. Las pruebas del importador son
offline; la descarga completa se comprueba aparte con `--dataset-only`.

## Actualizar una copia existente

Antes del primer `git pull` que retire los datos de Git, copia fuera del repositorio
tu dataset, `OutPut` y cualquier configuración local que quieras conservar:
Git puede eliminar archivos que antes estaban versionados. Después puedes importar
la copia con `--dataset-source` y devolver tus salidas personales a `OutPut`.
Los modelos históricos siguen publicados. Las selecciones antiguas con rutas
absolutas de otro ordenador requieren volver a elegir las carpetas desde la app.

Para descargar los modelos e imágenes históricos almacenados con Git LFS, usa
Git LFS al clonar y ejecuta `git lfs pull`. Si un `.pth` es solo un pequeño puntero
de texto, todavía no se ha descargado el modelo. El ZIP de código de GitHub puede
contener esos punteros dependiendo de la configuración del repositorio.

Eliminar datos del árbol actual no borra los commits antiguos: reducir el tamaño
del historial requeriría una operación separada con copias de seguridad.

