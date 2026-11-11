"""Reglas de selección de la UI sin requerir una pantalla."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import threading
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
from PIL import Image
from app import App

class SelectionTests(TestCase):
    def test_train_enabled_only_for_five_to_ten_when_idle(self):
        for count in (0,4,5,6,10,11,40):
            for busy in (False,True):
                view=SimpleNamespace(entries=lambda:list(range(count)),busy=busy,
                                     selection_info=Mock(),train_button=Mock(),refresh_architecture=Mock())
                App.refresh_selection(view)
                state='normal' if 5<=count<=10 and not busy else 'disabled'
                view.train_button.configure.assert_called_once_with(state=state)
                self.assertIn(f'Seleccionados: {count}',view.selection_info.set.call_args.args[0])
    def test_root_candidates_can_be_loaded_unselected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'Bosque';folder.mkdir();Image.new('RGB',(8,8)).save(folder/'test.png')
            view=SimpleNamespace(cancel=threading.Event())
            candidates=App.scan(view,[folder],selected=False)
            self.assertFalse(candidates[0]['use'])
            self.assertTrue(App.scan(view,[folder])[0]['use'])
