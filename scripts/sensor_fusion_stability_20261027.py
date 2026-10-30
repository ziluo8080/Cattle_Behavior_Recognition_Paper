"""Metrics and checkpoint policy; no learned gate and no probability smoothing."""
import numpy as np
from fractions import Fraction
from scripts.sensor_xgb_b_fusion_20261022 import validate_probabilities, temper, fuse, confidence

def metric(prob,truth,weights):
    prob=validate_probabilities(prob);truth=np.asarray(truth,dtype=np.int64)
    assert truth.shape==(len(prob),) and np.isin(truth,range(7)).all()
    weights=np.asarray(weights,dtype=np.float64);assert weights.shape==(7,) and (weights>0).all()
    cm=np.bincount(truth*7+prob.argmax(1),minlength=49).reshape(7,7)
    scores=[Fraction(2*int(cm[i,i]),int(cm[i].sum()+cm[:,i].sum())) if cm[i].sum()+cm[:,i].sum() else Fraction() for i in range(7)]
    exact=sum(scores,Fraction())/7*100
    pt=prob[np.arange(len(truth)),truth]
    # Preserve exact zero-probability failures rather than clipping them away.
    finite=bool((pt>0).all());nll=-np.log(pt) if finite else None
    return dict(macro_f1_percent=float(exact),class_f1_percent=[float(s*100) for s in scores],confusion_matrix=cm.tolist(),
        weighted_CE=float(np.average(nll,weights=weights[truth])) if finite else None,NLL=float(nll.mean()) if finite else None,
        zero_true_probabilities=int((pt==0).sum())),exact

def fusion_metric(px,pb,truth,weights,previous=None):
    x,b=temper(validate_probabilities(px),2.),temper(validate_probabilities(pb),1.)
    choose=.7*confidence(x,'margin')>=.3*confidence(b,'margin')
    prob=fuse(x,b,'margin_select',.7);m,_=metric(prob,truth,weights)
    xp,bp=x.argmax(1),b.argmax(1);disagree=xp!=bp
    chosen=np.where(choose,xp,bp)
    m.update(tree_selected_fraction=float(choose.mean()),expert_disagreement_fraction=float(disagree.mean()),
        disagreement_choice_accuracy=float((chosen[disagree]==truth[disagree]).mean()) if disagree.any() else None,
        tree_selection_flip_fraction=float((choose!=previous).mean()) if previous is not None else None)
    return m,choose,prob

def improves(score,best):
    """Exact F1 ties keep the earlier checkpoint; no held-fusion selection."""
    return best is None or score>best
