"""Pruebas con imágenes sintéticas; no miden precisión con Minecraft."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'minecraft_biomas'))
import tempfile
import threading
import unittest
import numpy as np
from PIL import Image
import torch
from core import Config, ConvNet, train, collect, predict, load_checkpoint, split_samples, Cancelled

torch.set_num_threads(2)

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        rng=np.random.default_rng(5); self.entries=[]
        for c,name in enumerate(['Bosque','Desierto','Montaña','Jungla','Océano']):
            folder=self.root/name; folder.mkdir()
            for i in range(25):
                Image.fromarray(rng.integers(0,256,(32,48,3),dtype=np.uint8)).save(folder/f'{i}.png')
            self.entries.append({'name':name,'folder':str(folder)})
    def tearDown(self):self.temp.cleanup()
    def test_complete_pipeline_and_legacy(self):
        events=[]
        result=train(self.entries,Config(epochs=2,image_size=32,batch_size=16,validation_percent=16,device='cpu'),
                     self.root/'runs',lambda k,v:events.append((k,v)))
        cp=load_checkpoint(result['path'])
        self.assertEqual(cp['metrics']['sizes'],{'train':80,'val':20,'test':25})
        self.assertEqual(cp['classes'],['Bosque','Desierto','Montaña','Jungla','Océano'])
        self.assertEqual(len(cp['history']),2)
        self.assertIn(cp['metrics']['selected_epoch'],[1,2])
        self.assertIn('val_loss',cp['history'][0])
        probs=predict(cp,self.root/'Bosque/0.png')
        self.assertAlmostEqual(sum(p for _,p in probs),1,places=6)
        self.assertEqual([n for n,_ in probs],cp['classes'])
        self.assertTrue(Path(result['path']).with_name('particiones.csv').exists())
        self.assertTrue(any(k=='epoch' for k,v in events))
        # Formato del proyecto anterior: mismo nombre de capas y checkpoint mínimo.
        old={'state_dict':ConvNet(64,6).state_dict(),'classes':[f'biome_{i}' for i in range(6)],'image_size':64}
        torch.save(old,self.root/'old.pth'); loaded=load_checkpoint(self.root/'old.pth')
        self.assertEqual(len(predict(loaded,self.root/'Bosque/0.png')),6)
    def test_split_dedup_and_repeatability(self):
        import shutil
        shutil.copy(self.root/'Bosque/0.png',self.root/'Bosque/duplicate.png')
        names,samples=collect(self.entries,threading.Event(),lambda s:None)
        self.assertEqual(len(samples),125)
        a=split_samples(samples,Config()); b=split_samples(samples,Config())
        self.assertEqual(a,b); self.assertEqual(len(a['test']),25); self.assertEqual(len(a['train']),100)
        self.assertFalse(set(a['train'])&set(a['test']))
        self.assertEqual(set(a['train'])|set(a['test']),set(range(125)))
        shutil.copy(self.root/'Bosque/0.png',self.root/'Desierto/conflict.png')
        with self.assertRaisesRegex(ValueError,'distintos'):collect(self.entries,threading.Event(),lambda s:None)
    def test_cancel_and_invalid_input(self):
        event=threading.Event();event.set()
        with self.assertRaises(Cancelled):train(self.entries,Config(device='cpu'),self.root/'runs',cancel=event)
        self.assertFalse((self.root/'runs').exists())
        with self.assertRaises(ValueError):Config(lr=float('nan')).validate()
        with self.assertRaisesRegex(ValueError,'único'):collect([self.entries[0]]*5,threading.Event(),lambda s:None)
        Image.new('RGB',(32,32)).save(self.root/'Bosque/temp.png')
        (self.root/'Bosque/temp.png').write_text('invalid')
        with self.assertRaisesRegex(ValueError,'ilegible'):collect(self.entries,threading.Event(),lambda s:None)
    def test_class_limits_and_ten_outputs(self):
        for n in (0, 4, 11):
            with self.assertRaisesRegex(ValueError, 'entre 5 y 10'):
                collect([self.entries[0]]*n, threading.Event(), lambda s:None)
        rng=np.random.default_rng(80)
        for c in range(5):
            folder=self.root/f'Extra_{c}'; folder.mkdir()
            for i in range(25):
                Image.fromarray(rng.integers(0,256,(32,48,3),dtype=np.uint8)).save(folder/f'{i}.png')
            self.entries.append({'name':folder.name,'folder':str(folder)})
        result=train(self.entries,Config(epochs=1,image_size=32,device='cpu'),self.root/'ten')
        cp=load_checkpoint(result['path'])
        self.assertEqual(cp['metrics']['sizes'],{'train':200,'val':0,'test':50})
        self.assertEqual(cp['state_dict']['fc2.weight'].shape[0],10)
        self.assertEqual(len(predict(cp,self.root/'Bosque/0.png')),10)

    def test_multiple_sizes(self):
        for size in (32,64,96,128):
            self.assertEqual(tuple(ConvNet(size,4)(torch.zeros(2,3,size,size)).shape),(2,4))

if __name__=='__main__':unittest.main()

