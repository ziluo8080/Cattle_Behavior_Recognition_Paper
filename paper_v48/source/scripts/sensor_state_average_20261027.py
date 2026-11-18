"""Causal joint averaging of EMA parameters and floating buffers in an eval clone."""
import torch


@torch.no_grad()
def apply_window(model,history,epoch,warm_epochs=10,window=3):
    assert warm_epochs==10 and window==3 and history[-1]['epoch']==epoch
    assert [x['epoch'] for x in history]==list(range(epoch-len(history)+1,epoch+1))
    original={n:v.detach().clone() for n,v in model.state_dict().items()};parameters=set(dict(model.named_parameters()));averaged=[];used=[epoch]
    if epoch>warm_epochs:
        chosen=history[-window:];assert len(chosen)==window;used=[x['epoch'] for x in chosen]
        for name,target in model.state_dict().items():
            if target.is_floating_point():
                expected=torch.stack([x['model'][name].detach().cpu().double() for x in chosen]).mean(0);assert torch.isfinite(expected).all()
                target.copy_(expected.to(target));averaged.append(name)
            else:assert torch.equal(target,original[name])
        assert parameters.issubset(averaged)
    else:assert all(torch.equal(v,original[n]) for n,v in model.state_dict().items())
    assert all(v.is_floating_point() or torch.equal(v,original[n]) for n,v in model.state_dict().items())
    return dict(epoch=epoch,operator='joint_parameter_and_float_buffer_mean',state_reference_epochs=used,BN_reference_epochs=used,averaged_float_tensors=averaged,averaged_parameter_names=sorted(parameters.intersection(averaged)),nonfloat_buffers_unchanged=True,source_training_not_modified=True)
