import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'minecraft_biomas'))
import copy
import pytest
import torch
from architectures import ORIGINAL, COMPACT, build_architecture, validate_architecture
from core import Config, train, predict, load_checkpoint
import test_core

def test_presets_shapes_and_gradients():
    for spec in (ORIGINAL,COMPACT):
        for size in (32,64,96,128):
            for classes in (5,10):
                net=build_architecture(spec,size,classes)
                y=net(torch.randn(2,3,size,size))
                assert y.shape==(2,classes)
                y.sum().backward()
                assert all(p.grad is not None for p in net.parameters())

def test_invalid_specs():
    for layers in ([{'type':'exec'}], [{'type':'Linear','out_features':'classes'}],
                   [{'type':'MaxPool2d','kernel_size':128}],
                   [{'type':'Dropout','p':float('nan')}],
                   [{'type':'Flatten'},{'type':'Linear','out_features':4096}],
                   [{'type':'Conv2d','out_channels':16,'kernel_size':3,'oops':True}]):
        with pytest.raises(ValueError): validate_architecture({'name':'invalid','layers':layers})

def test_bn_summary_does_not_update_stats():
    net=build_architecture(COMPACT,32,5)
    assert all(m.num_batches_tracked.item()==0 for m in net.modules() if isinstance(m,torch.nn.BatchNorm2d))

def test_custom_training_reload_prediction():
    fixture=test_core.CoreTests(); fixture.setUp()
    try:
        result=train(fixture.entries,Config(epochs=1,image_size=32,batch_size=8,device='cpu',architecture=copy.deepcopy(COMPACT)),fixture.root/'custom')
        cp=load_checkpoint(result['path'])
        assert cp['architecture']==COMPACT
        probs=predict(cp,fixture.root/'Bosque/0.png')
        assert len(probs)==5
        assert sum(p for _,p in probs)==pytest.approx(1)
    finally: fixture.tearDown()

