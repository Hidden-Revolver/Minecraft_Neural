import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'minecraft_biomas'))
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from core import Config, ConvNet, cap_samples, training_weights, make_optimizer, make_scheduler, train, load_checkpoint, predict

torch.set_num_threads(2)
class TrainingOptionsTests(unittest.TestCase):
    def test_cap_100_retains_all_86_and_is_reproducible(self):
        samples=[(f'{c}/{i}.png',c,f'{c}_{i}') for c,n in enumerate([3000,86,200,100,150]) for i in range(n)]
        capped=cap_samples(samples,100,42)
        self.assertEqual([sum(s[1]==c for s in capped) for c in range(5)],[100,86,100,100,100])
        self.assertEqual(capped,cap_samples(samples,100,42))
        self.assertNotEqual(capped,cap_samples(samples,100,43))
        self.assertEqual(len({s[0] for s in capped}),len(capped))
        self.assertEqual(cap_samples(samples,0,42),samples)
        self.assertEqual([s for s in capped if s[1]==1],[s for s in samples if s[1]==1])
    def test_weights_only_use_training_and_match_ce(self):
        samples=[('a',0,''),('b',0,''),('c',1,'')]+[('test',1,'')]*99
        w=training_weights(samples,[0,1,2],2,'balanced')
        torch.testing.assert_close(w,torch.tensor([.75,1.5]))
        logits=torch.tensor([[2.,0.],[0.,1.],[2.,0.]])
        y=torch.tensor([0,0,1])
        expected=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*w[y]).sum()/w[y].sum()
        torch.testing.assert_close(torch.nn.CrossEntropyLoss(weight=w)(logits,y),expected)
        self.assertIsNone(training_weights(samples,[0,1,2],2,'none'))
    def test_schedulers_and_invalid_configs(self):
        cfg=Config(device='cpu',optimizer='AdamW',scheduler='plateau',validation_percent=16,scheduler_patience=0)
        opt=make_optimizer(ConvNet(),cfg);self.assertIsInstance(opt,torch.optim.AdamW)
        scheduler=make_scheduler(opt,cfg);scheduler.step(1.);scheduler.step(1.)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],cfg.lr*.5)
        for kwargs in [{'max_images_per_class':5},{'weight_decay':float('nan')},{'scheduler':'plateau'},
                       {'selection_metric':'val_macro_f1'},{'optimizer':'invalid'}]:
            with self.assertRaises(ValueError):Config(device='cpu',**kwargs).validate()
    def test_training_with_every_optimizer_caps_weights_and_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rng=np.random.default_rng(17);entries=[]
            for c,n in enumerate([25,14,25,25,25]):
                folder=root/str(c);folder.mkdir()
                for i in range(n):Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(folder/f'{i}.png')
                entries.append({'name':str(c),'folder':str(folder)})
            for optimizer,scheduler in [('Adam','none'),('AdamW','plateau'),('SGD','cosine')]:
                cfg=Config(device='cpu',epochs=3,image_size=32,batch_size=16,validation_percent=16,
                           optimizer=optimizer,scheduler=scheduler,class_weighting='balanced',max_images_per_class=20,
                           selection_metric='val_macro_f1',weight_decay=.0001)
                result=train(entries,cfg,root/'runs')
                cp=load_checkpoint(result['path']); history=cp['history']
                self.assertEqual([r['selected'] for r in cp['data_summary']],[20,14,20,20,20])
                self.assertEqual(sum(cp['metrics']['sizes'].values()),94)
                self.assertEqual(cp['metrics']['sizes']['test'],19)
                expected=max(history,key=lambda r:r['val_macro_f1'])['epoch']
                self.assertEqual(cp['metrics']['selected_epoch'],expected)
                for row,w in zip(cp['data_summary'],cp['class_weights']):
                    self.assertAlmostEqual(w,cp['metrics']['sizes']['train']/(5*row['train']),places=6)
                self.assertIn('balanced_accuracy',cp['metrics'])
                self.assertTrue(all('objective_loss' in r and 'val_macro_f1' in r and 'lr' in r for r in history))
                if scheduler=='cosine':self.assertLess(history[-1]['lr'],history[0]['lr'])
                self.assertAlmostEqual(sum(p for _,p in predict(cp,root/'0/0.png')),1,places=6)

