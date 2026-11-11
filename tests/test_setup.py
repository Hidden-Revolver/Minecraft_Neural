"""La selección de modo no instala paquetes ni abre ventanas en estas pruebas."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('minecraft_setup', ROOT / 'setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_broken_tk_reports_system_dependency(self):
        from types import SimpleNamespace
        with patch.object(setup.subprocess, 'run',
                          return_value=SimpleNamespace(returncode=1, stderr='Tcl/Tk missing')):
            with self.assertRaisesRegex(setup.SetupError, 'No se descargan dependencias'):
                setup.check_environment_tk(Path('python'))

    def test_dataset_only_does_not_need_torch_or_tkinter(self):
        with patch.object(sys, 'argv', ['setup.py', '--dataset-only']), \
             patch.object(setup, 'prepare_dataset') as prepare, \
             patch.object(setup, 'check_prerequisites', side_effect=AssertionError), \
             patch.object(setup, 'select_torch_index', side_effect=AssertionError):
            self.assertEqual(setup.main(), 0)
            prepare.assert_called_once()

    def test_source_option_is_forwarded(self):
        with patch.object(sys, 'argv', ['setup.py', '--dataset-only', '--dataset-source', 'local.zip']), \
             patch.object(setup, 'prepare_dataset') as prepare:
            self.assertEqual(setup.main(), 0)
            self.assertEqual(prepare.call_args.args[1], Path('local.zip'))

    def test_skip_dataset_install_only_reuses_healthy_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / 'python'
            python.touch()
            state = root / 'state.json'
            state.write_text('{"signature": "same"}')
            with patch.object(sys, 'argv', ['setup.py', '--cpu', '--skip-dataset', '--install-only']), \
                 patch.object(setup, 'ROOT', root), \
                 patch.object(setup, 'STATE_FILE', state), \
                 patch.object(setup, 'check_prerequisites'), \
                 patch.object(setup, 'check_environment_tk'), \
                 patch.object(setup, 'select_torch_index', return_value=('cpu', False)), \
                 patch.object(setup, 'setup_signature', return_value='same'), \
                 patch.object(setup, 'env_python', return_value=python), \
                 patch.object(setup, 'environment_is_healthy', return_value=True), \
                 patch.object(setup, 'prepare_dataset', side_effect=AssertionError), \
                 patch.object(setup, 'install', side_effect=AssertionError), \
                 patch.object(setup.subprocess, 'run', side_effect=AssertionError):
                self.assertEqual(setup.main(), 0)
                self.assertTrue((root / 'OutPut').is_dir())

