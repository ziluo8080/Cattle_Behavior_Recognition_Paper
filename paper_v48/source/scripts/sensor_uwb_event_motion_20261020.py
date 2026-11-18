"""UWB-only native-event and explicit-motion factorial model; other branches unchanged."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from q3_sensor.endpoint_models import EndpointFusion, tokens

KINDS=('Grid','Event','GridMotion','EventMotion')


def array_sha(array):
    import hashlib
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def derive_uwb_features(x, valid, age, timestamps, starts, coordinate_std, train_mask,
                        scaler=None, original_grid=True):
    """Derive source identity and apparent motion without labels; TRAIN-only motion scale."""
    n,length,d=x.shape
    assert d==3 and valid.shape==x.shape and age.shape==(n,length,1)
    assert np.array_equal(valid.any(-1),valid.all(-1)), 'Need co-valid UWB coordinate observations'
    usable=valid.all(-1).astype(bool)
    grid=np.asarray(timestamps,dtype=np.float64)[:,None]+np.arange(1-length,1)[None]
    source=grid-np.asarray(age[:,:,0],dtype=np.float64)
    assert np.all(np.isfinite(source[usable]))
    assert np.all(source[usable]<=grid[usable]+1e-6)
    assert np.all((source>np.asarray(timestamps)[:,None]-60)[usable])
    assert np.all((source>=np.asarray(starts)[:,None])[usable])
    assert np.max(np.abs(source[usable]-np.rint(source[usable])),initial=0.)<1e-5, 'This dataset uses integer UWB source seconds'
    source_id=np.rint(np.where(usable,source,0.)).astype(np.int64)
    event_mask=np.zeros((n,length),bool)
    raw=np.zeros((n,length,6),np.float64)
    pair_mask=np.zeros((n,length),bool)
    counts=np.zeros(n,np.int64)
    # At most 60 positions per row; retain every distinct valid source once.
    for i in range(n):
        previous=None;last_source=None;last_value=None;descriptor=np.zeros(6)
        for t in range(length):
            if not usable[i,t]:continue
            current=int(source_id[i,t])
            if previous is None or current!=last_source:
                assert previous is None or current>last_source
                event_mask[i,t]=True;counts[i]+=1
                if previous is None:
                    descriptor=np.array([0.,0.,0.,0.,0.,1.])
                else:
                    dt=float(current-last_source);assert 0<dt<60
                    delta=(x[i,t].astype(np.float64)-last_value)*np.asarray(coordinate_std)
                    descriptor=np.r_[delta,np.linalg.norm(delta[:2])/dt,np.log1p(dt)/np.log(61.),0.]
                    pair_mask[i,t]=True
                previous=t;last_source=current;last_value=x[i,t].astype(np.float64)
            else:
                assert np.array_equal(x[i,t],last_value), 'A held source must retain its coordinates'
            raw[i,t]=descriptor
    if scaler is None:
        fit=pair_mask & np.asarray(train_mask,dtype=bool)[:,None]
        assert fit.any()
        observed=raw[:,:,:4][fit]
        mean=observed.mean(0);std=np.maximum(observed.std(0),1e-3)
        scaler={'mean_first4':mean.tolist(),'std_first4':std.tolist(),
                'fit_event_pairs':int(fit.sum()),'scope':'TRAIN event pairs as represented in overlapping TRAIN windows; no label input',
                'fifth_scale':'log1p(delta_seconds)/log(61)','sixth':'first-source-event flag',
                'coordinate_difference_units':'metres; velocity is apparent planar m/s'}
    # Scale held descriptors as well; unknown first-pair values remain zero.
    motion=raw.copy();known=usable & (raw[:,:,5]==0)
    motion[:,:,:4][known]=(motion[:,:,:4][known]-np.asarray(scaler['mean_first4']))/np.asarray(scaler['std_first4'])
    motion[:,:,:4][~known]=0.
    motion[~usable]=0.
    assert np.isfinite(motion).all()
    width=max(int(counts.max()),1)
    indices=np.zeros((n,width),np.int64)
    for i in range(n):
        ids=np.flatnonzero(event_mask[i]);indices[i,:len(ids)]=ids
    arrays={'motion':motion.astype(np.float32),'event_index':indices,'event_count':counts,
            'event_mask':event_mask,'source_id':source_id}
    stats={'rows':n,'events_min':int(counts.min()),'events_max':int(counts.max()),
           'events_median':float(np.median(counts)),'no_valid_rows':int((counts==0).sum()),
           'event_pairs':int(pair_mask.sum()),'original_grid':original_grid,
           'array_sha256':{k:array_sha(v) for k,v in arrays.items()}}
    return arrays,scaler,stats


class UWBEncoder(nn.Module):
    def __init__(self,original,event=False,motion=False):
        super().__init__();self.event=event;self.motion=motion
        self.project,self.gru,self.norm=original.project,original.gru,original.norm
        if motion:
            # Match all original backbone initialization and avoid consuming its RNG stream.
            with torch.random.fork_rng(devices=[]):new=nn.Linear(14,32)
            with torch.no_grad():
                new.weight[:,:8].copy_(self.project.weight);new.weight[:,8:].zero_()
                new.bias.copy_(self.project.bias)
            self.project=new

    def forward(self,data):
        value=tokens(data)
        if self.motion:value=torch.cat((value,data['motion']),-1)
        if self.event:
            ids=data['event_index']
            value=value.gather(1,ids[...,None].expand(-1,-1,value.shape[-1]))
            value=F.gelu(self.project(value))
            lengths=data['event_count'].clamp_min(1).detach().cpu()
            packed=pack_padded_sequence(value,lengths,batch_first=True,enforce_sorted=False)
            _,last=self.gru(packed);readout=last[-1]
        else:
            out,_=self.gru(F.gelu(self.project(value)));readout=out[:,-1]
        return self.norm(readout)*data['valid'][:,-1].any(-1,keepdim=True)


def make_model(kind,prior):
    if kind not in KINDS:raise ValueError(kind)
    model=EndpointFusion('gru',prior)
    if kind!='Grid':model.encoders['uwb']=UWBEncoder(model.encoders['uwb'],event=kind.startswith('Event'),motion=kind.endswith('Motion'))
    return model


class ParameterEMA:
    """Same-trajectory EMA; first post-step copy, then a predeclared decay warmup."""
    def __init__(self,model,decay=.998,warmup_steps=200):
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.decay=float(decay);self.warmup_steps=int(warmup_steps);self.updates=0

    @torch.no_grad()
    def update(self,model):
        self.updates+=1
        beta=self.decay*(1-math.exp(-self.updates/self.warmup_steps))
        if self.updates==1:beta=0.
        for averaged,current in zip(self.model.parameters(),model.parameters()):
            averaged.mul_(beta).add_(current.detach(),alpha=1-beta)
        for averaged,current in zip(self.model.buffers(),model.buffers()):averaged.copy_(current)
        return beta
