# Minecraft Neural — instalación automática

Aplicación local de clasificación de biomas de Minecraft usando **PyTorch** y **Tkinter**. El entrenamiento utiliza la CPU o, si hay una NVIDIA compatible, CUDA de manera nativa. No es una aplicación web.

## Requisitos previos

- Python **3.12 de 64 bits** (también se admiten 3.11 y 3.13), con pip y Tkinter. En Windows, instálalo desde python.org.
- Acceso a Internet durante la primera instalación.
- Para entrenar con CUDA, una GPU NVIDIA y sus controladores actualizados. No necesitas instalar por separado el CUDA Toolkit.
- El instalador automatiza los paquetes Python; **no** instala Python, drivers de NVIDIA, Tkinter del sistema ni descarga datasets.

## Instalación y ejecución (Windows PowerShell)

Abre PowerShell desde la **raíz del repositorio**, no desde la subcarpeta `minecraft_biomas`:

```powershell
py -3.12 setup.py
```

La primera ejecución creará `.venv`, descargará las dependencias, comprobará PyTorch y abrirá la interfaz. Las siguientes ejecuciones reutilizarán el entorno si las dependencias siguen correctas. El tamaño de la descarga inicial puede ser de varios GB.

En Linux o macOS, utiliza `python3 setup.py` con una versión compatible de Python. En Fedora, si falta Tkinter, instala antes `python3-tkinter`.

## Opciones

```powershell
py -3.12 setup.py --install-only  # Preparar y comprobar sin abrir
py -3.12 setup.py --cuda          # Exigir CUDA; fallar si no funciona
py -3.12 setup.py --cpu           # Forzar una instalación CPU
py -3.12 setup.py --reinstall     # Reparar/reinstalar dependencias
```

Si instalas con `--cpu`, usa de nuevo `--cpu` para abrir la aplicación sin reemplazar la distribución de PyTorch. Para cambiar entre CPU y CUDA, el instalador reinstalará PyTorch cuando sea necesario.

Para ejecutar pruebas tras instalar:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\minecraft_biomas\tests -v
```

Puedes consultar el [manual original](minecraft_biomas/README.md) para conocer los controles de entrenamiento, las arquitecturas JSON y cómo cargar o guardar modelos.

### Importante

Los modelos, imágenes, arquitecturas y resultados que ya tengas no se borran. El entorno nuevo `.venv` se guarda en la raíz y no modifica el `venv312` anterior. Los entornos virtuales no deben subirse al repositorio.

Si falla la verificación de CUDA, actualiza el driver de NVIDIA o prueba `--cpu`. Las distribuciones oficiales utilizadas son `torch==2.10.0` y `torchvision==0.25.0`, con índice CUDA 12.6 o 12.8 según el controlador. No está comprobado todavía en Windows con la GPU del autor.

## Datos, resultados históricos y salidas personales

- `minecraft_biomas/minecraft_biomas/`: el dataset se conserva con sus nombres actuales. Este instalador todavía no descarga ni renombra el dataset.
- `minecraft_biomas/resultados/`: entrenamientos históricos versionados para mostrar la evolución y cargar una red entrenada.
- `minecraft_biomas/resultados/archivo_OutPut/`: copia íntegra del antiguo OutPut, incluida la selección, el mejor modelo y las gráficas. Las rutas de la selección son las del ordenador original; en otro equipo hay que volver a elegir las carpetas.
- `OutPut/`: nuevos entrenamientos y destino inicial al exportar gráficas. Su contenido es local y se ignora en Git; solo se versiona `.gitkeep`, porque Git no guarda carpetas vacías.
- `venv312`, `__pycache__` y `.idea` dejan de estar versionados. El setup crea el entorno `.venv`.

### Actualizar una copia existente

Antes de fusionar esta limpieza y hacer `git pull`, copia fuera del repositorio tu
`OutPut` y cualquier configuración de `.idea` que quieras conservar.
Git puede eliminar los archivos previamente versionados al aplicar la limpieza.
El OutPut publicado queda archivado en resultados; los archivos personales que
solo existan en tu ordenador deben copiarse antes. Tras actualizar, puedes devolver
tus salidas a `OutPut`. Ejecuta `py -3.12 setup.py` para usar el nuevo entorno.
No se reescribe el historial: los commits anteriores siguen ocupando espacio.
