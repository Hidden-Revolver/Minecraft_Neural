"""Descarga e importa el dataset original sin modificar datos del usuario."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile

MANIFEST = Path(__file__).with_name('dataset_manifest.json')
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class DatasetError(Exception):
    """El dataset no se pudo preparar sin riesgo de perder datos."""


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def fingerprint(folder):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise DatasetError(f'Imagen inválida o enlace: {path}')
        with path.open('rb') as stream:
            content_hash = hashlib.file_digest(stream, 'sha256').digest()
        digest.update(path.name.encode('utf-8') + b'\0' + content_hash)
    return len(files), digest.hexdigest()


def validate_dataset(root, manifest):
    """Comprueba todas las imágenes, no solo un marcador de instalación."""
    expected = {item['name'] for item in manifest['classes'].values()}
    actual = {p.name for p in root.iterdir() if p.is_dir()}
    if actual != expected:
        raise DatasetError('Las carpetas no coinciden con los 41 biomas originales.')
    for item in manifest['classes'].values():
        folder = root / item['name']
        if folder.is_symlink() or any(p.is_dir() for p in folder.iterdir()):
            raise DatasetError(f'Carpeta inesperada o enlace: {folder}')
        if fingerprint(folder) != (item['files'], item['sha256']):
            raise DatasetError(f'Imágenes ausentes, añadidas o modificadas en {folder.name}.')


def download_archive(destination, manifest, report):
    report('[Dataset] Descargando Kaggle v1 (unos 217 MB descomprimidos)...')
    request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'MinecraftNeural/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, destination.open('wb') as out:
            total = 0
            next_report = 20 * 1024 * 1024
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise DatasetError('La descarga supera el tamaño esperado.')
                out.write(chunk)
                if total >= next_report:
                    report(f'[Dataset] Descargados {total // (1024 * 1024)} MiB...')
                    next_report += 20 * 1024 * 1024
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DatasetError(
            'No se pudo descargar Kaggle. Descarga el ZIP desde '
            f"{manifest['source_page']} y usa --dataset-source RUTA_AL_ZIP, "
            'o --skip-dataset para usar tus propios datos. No se ha modificado el destino.'
        ) from exc


def extract_archive(archive, destination):
    """Extrae solo rutas relativas normales; valida antes de escribir."""
    try:
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if sum(info.file_size for info in members) > MAX_EXTRACTED_BYTES:
                raise DatasetError('El ZIP descomprimido supera 1 GiB.')
            seen = set()
            for info in members:
                # ZipInfo normaliza separadores en Windows y trunca en NUL.
                # Validar el nombre original evita diferencias entre plataformas.
                original = info.orig_filename
                path = PurePosixPath(original)
                mode = info.external_attr >> 16
                if (path.is_absolute() or '..' in path.parts or '\\' in original
                        or ':' in original or '\x00' in original or not path.parts
                        or stat.S_ISLNK(mode)):
                    raise DatasetError(f'Ruta insegura en ZIP: {original!r}')
                key = str(path).casefold()
                if key in seen:
                    raise DatasetError(f'Ruta duplicada en ZIP: {info.filename}')
                seen.add(key)
            for info in members:
                target = destination.joinpath(*PurePosixPath(info.filename).parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise DatasetError('El archivo no es un ZIP válido y completo del dataset.') from exc


def source_folders(source, manifest):
    """Acepta preprocessed_data, su carpeta exterior o nombres ya convertidos."""
    candidates = [source, source / 'preprocessed_data', source / 'minecraft_biomas']
    matches = []
    for root in candidates:
        if not root.is_dir() or root.is_symlink():
            continue
        folders = {}
        valid = True
        for ident, item in manifest['classes'].items():
            aliases = {f'biome_{ident}', ident, item['name'], item['name'].lower()}
            found = [p for p in root.iterdir() if p.name in aliases and p.is_dir()]
            if len(found) != 1 or found[0].is_symlink():
                valid = False
                break
            folders[item['name']] = found[0]
        if valid:
            extra = {p for p in root.iterdir() if p.is_dir()} - set(folders.values())
            if extra:
                raise DatasetError(f'Carpetas adicionales o biomas duplicados en {root}.')
            matches.append(folders)
    if len(matches) != 1:
        raise DatasetError('No se encuentra un único dataset completo con los biomas esperados.')
    return matches[0]


def prepare_dataset(destination, source=None, *, manifest=None, report=print):
    """Publica el dataset solo tras validarlo; conserva el origen y destinos existentes."""
    manifest = manifest or load_manifest()
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise DatasetError('El destino del dataset no puede ser un enlace.')
    if destination.exists():
        try:
            validate_dataset(destination, manifest)
        except (DatasetError, OSError) as exc:
            raise DatasetError(
                f'El dataset existente en {destination} está incompleto o es distinto: {exc} '
                'Se conserva sin cambios. Muévelo a otra carpeta antes de repetir, '
                'o usa --skip-dataset para seguir trabajando con él.'
            ) from exc
        report(f'[Dataset] Verificado y reutilizado: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / '.dataset-install.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise DatasetError(
            f'Otra preparación puede estar en curso. Si terminó inesperadamente, '
            f'comprueba que no queda ningún setup ejecutándose y retira {lock}.'
        ) from exc
    try:
        if shutil.disk_usage(destination.parent).free < 1024**3:
            raise DatasetError('Se necesita al menos 1 GiB libre para preparar el dataset.')
        with tempfile.TemporaryDirectory(prefix='.dataset-tmp-', dir=destination.parent) as temporary:
            temp = Path(temporary)
            if source is None:
                archive = temp / 'dataset.zip'
                download_archive(archive, manifest, report)
                source = archive
            source = Path(source).resolve()
            if source.is_file():
                # El contenido de cada bioma también se verifica después; esto permite
                # importar un ZIP local reempaquetado sin depender de sus metadatos ZIP.
                extracted = temp / 'extracted'
                extracted.mkdir()
                extract_archive(source, extracted)
                source = extracted
            if not source.is_dir():
                raise DatasetError(f'No existe el origen: {source}')
            folders = source_folders(source, manifest)
            staged = temp / 'ready'
            staged.mkdir()
            for name, folder in folders.items():
                target = staged / name
                target.mkdir()
                for path in folder.iterdir():
                    if path.is_symlink() or path.is_dir():
                        raise DatasetError(f'Contenido inesperado: {path}')
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        shutil.copy2(path, target / path.name)
            report('[Dataset] Verificando nombres, recuentos y SHA-256 de todas las imágenes...')
            validate_dataset(staged, manifest)
            if destination.exists():
                raise DatasetError('El destino apareció durante la preparación; no se sobrescribe.')
            staged.rename(destination)
        report(f'[Dataset] Listo: {manifest["files"]} imágenes en {destination}')
        return destination
    finally:
        lock.rmdir()
