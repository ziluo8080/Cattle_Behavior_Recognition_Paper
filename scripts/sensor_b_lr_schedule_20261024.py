"""Fixed update-index learning rates; no data/model/launcher access."""
import math

ARMS=('Constant','Cosine')


def learning_rate(arm,step,total_steps,maximum=.001,minimum=.00001):
    assert arm in ARMS and total_steps>=2 and 0<=step<total_steps and 0<minimum<=maximum
    if arm=='Constant':return maximum
    return minimum+.5*(maximum-minimum)*(1+math.cos(math.pi*step/(total_steps-1)))


def checks():
    values=[learning_rate('Cosine',i,120) for i in range(120)]
    assert values[0]==.001 and values[-1]==.00001
    assert all(a>=b for a,b in zip(values,values[1:]))
    assert all(learning_rate('Constant',i,120)==.001 for i in range(120))
    assert values[:60]!=[learning_rate('Cosine',i,60) for i in range(60)]
    return dict(first_last_exact=True,monotonic_decay=True,constant_unchanged=True,fixed_horizon_not_shortened_when_selecting_earlier_epoch=True)
