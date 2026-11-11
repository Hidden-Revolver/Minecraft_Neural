"""Pruebas offline de importación, integridad y conservación de datos."""
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_dataset import (
    DatasetError, extract_archive, fingerprint, prepare_dataset,
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'original' / 'preprocessed_data'
        self.manifest = {'files': 2, 'classes': {}}
        for ident, name in [('1', 'Plains'), ('2', 'desert')]:
            folder = self.source / f'biome_{ident}'
            folder.mkdir(parents=True)
            (folder / f'biome_{ident}_0.jpg').write_bytes(b'image-' + ident.encode())
            count, digest = fingerprint(folder)
            self.manifest['classes'][ident] = {'name': name, 'files': count, 'sha256': digest}
        self.dest = self.root / 'installed'

    def prepare(self, source=None):
        return prepare_dataset(self.dest, source or self.source.parent,
                               manifest=self.manifest, report=lambda _: None)

    def test_local_import_preserves_source_and_exact_case(self):
        self.prepare()
        self.assertEqual({p.name for p in self.dest.iterdir()}, {'Plains', 'desert'})
        self.assertTrue((self.source / 'biome_1/biome_1_0.jpg').exists())
        self.assertEqual((self.dest / 'Plains/biome_1_0.jpg').read_bytes(), b'image-1')

    def test_second_run_never_downloads_or_replaces(self):
        self.prepare()
        image = self.dest / 'Plains/biome_1_0.jpg'
        before = image.stat().st_mtime_ns
        with patch('scripts.prepare_dataset.download_archive', side_effect=AssertionError):
            prepare_dataset(self.dest, manifest=self.manifest, report=lambda _: None)
        self.assertEqual(image.stat().st_mtime_ns, before)

    def test_existing_modified_dataset_is_preserved(self):
        self.prepare()
        image = self.dest / 'Plains/biome_1_0.jpg'
        image.write_bytes(b'personal data')
        with self.assertRaisesRegex(DatasetError, 'conserva sin cambios'):
            self.prepare()
        self.assertEqual(image.read_bytes(), b'personal data')

    def test_incomplete_source_leaves_no_destination(self):
        (self.source / 'biome_2/biome_2_0.jpg').unlink()
        with self.assertRaises(DatasetError):
            self.prepare()
        self.assertFalse(self.dest.exists())
        self.assertFalse((self.root / '.dataset-install.lock').exists())

    def test_unknown_biome_and_alias_collision_are_rejected(self):
        for name in ['biome_999', 'Plains']:
            extra = self.source / name
            extra.mkdir()
            with self.assertRaises(DatasetError):
                self.prepare()
            extra.rmdir()

    def test_zip_import(self):
        archive = self.root / 'download.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for p in self.source.rglob('*.jpg'):
                z.write(p, p.relative_to(self.source.parent))
        self.prepare(archive)
        self.assertTrue((self.dest / 'Plains/biome_1_0.jpg').exists())
        self.assertTrue(archive.exists())

    def test_zip_traversal_absolute_windows_and_symlink_rejected(self):
        for name in ['../escape', '/absolute', 'C:/escape', 'folder\\escape']:
            archive = self.root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr(name, b'bad')
            with self.assertRaises(DatasetError):
                extract_archive(archive, self.root / 'extract')
        with zipfile.ZipFile(archive, 'w') as z:
            info = zipfile.ZipInfo('link')
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, '../escape')
        with self.assertRaises(DatasetError):
            extract_archive(archive, self.root / 'extract')
        self.assertFalse((self.root / 'extract').exists())

    def test_download_failure_cleans_staging(self):
        with patch('scripts.prepare_dataset.download_archive', side_effect=DatasetError('offline')):
            with self.assertRaises(DatasetError):
                prepare_dataset(self.dest, manifest=self.manifest, report=lambda _: None)
        self.assertFalse(self.dest.exists())
        self.assertFalse(list(self.root.glob('.dataset-*')))

    def test_interrupt_cleans_staging(self):
        with patch('scripts.prepare_dataset.shutil.copy2', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertFalse(self.dest.exists())
        self.assertTrue((self.source / 'biome_1/biome_1_0.jpg').exists())
        self.assertFalse(list(self.root.glob('.dataset-*')))

    def test_existing_lock_does_not_get_removed(self):
        lock = self.root / '.dataset-install.lock'
        lock.mkdir()
        with self.assertRaises(DatasetError):
            self.prepare()
        self.assertTrue(lock.exists())

    def test_manifest_matches_original_repository_names(self):
        manifest = json.loads((Path(__file__).resolve().parents[1] /
                               'scripts/dataset_manifest.json').read_text())
        self.assertEqual(len(manifest['classes']), 41)
        self.assertEqual(sum(c['files'] for c in manifest['classes'].values()), 22169)
        self.assertEqual(manifest['classes']['1']['name'], 'Plains')


if __name__ == '__main__':
    unittest.main()
