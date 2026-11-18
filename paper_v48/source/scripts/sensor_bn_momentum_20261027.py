"""Change only BatchNorm's running-statistic update rate."""
import torch


def configure_statistics(state, momentum):
    if not 0 < momentum <= 1:
        raise ValueError('BN statistic momentum must be in (0,1]')
    names=[]
    for label,model in [('raw',state['model']),('ema',state['ema'].model)]:
        found=[]
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                assert isinstance(module,torch.nn.BatchNorm2d) and module.track_running_stats
                module.momentum=float(momentum);found.append(name)
        assert len(found)==4
        names.append(found)
    assert names[0]==names[1]
    return names[0]


def equal_learned_state(base, left, right):
    """Raw parameters, optimizer and averaged parameters must remain exact."""
    for key in ['model','ema']:
        a=left[key] if key=='model' else left[key].model
        b=right[key] if key=='model' else right[key].model
        base.equal_tree(dict(a.named_parameters()),dict(b.named_parameters()))
        ab,bb=dict(a.named_buffers()),dict(b.named_buffers())
        assert ab.keys()==bb.keys()
        for name in ab:
            if not name.endswith(('running_mean','running_var')):
                base.equal_tree(ab[name],bb[name])
    base.equal_tree(left['opt'].state_dict(),right['opt'].state_dict())
    assert left['step']==right['step'] and left['ema'].updates==right['ema'].updates
