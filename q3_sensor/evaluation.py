"""Seven fixed classes, retaining every predeclared target in the denominator."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if not len(labels) or len(labels) != len(predictions):
        raise ValueError('Nonempty matching target/prediction arrays required')
    if not np.isin(labels, range(1, 8)).all() or not np.isin(predictions, range(1, 8)).all():
        raise ValueError('Seven-class labels and predictions must be in 1..7')
    p, r, f, support = precision_recall_fscore_support(labels, predictions, labels=list(range(1, 8)), zero_division=0)
    return dict(n=len(labels), macro_f1=float(f.mean()), precision=p.tolist(), recall=r.tolist(),
                f1=f.tolist(), support=support.tolist(),
                confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(1, 8))).tolist())


def evaluate(labels, predictions, windows):
    result = metrics(labels, predictions)
    availability = np.stack([windows[m]['valid'].any(axis=(1, 2)) for m in windows], axis=1)
    count = availability.sum(axis=1)
    result['by_available_modalities'] = {str(n): metrics(np.asarray(labels)[count == n], np.asarray(predictions)[count == n])
                                        for n in range(4) if (count == n).any()}
    result['by_endpoint_age'] = {}
    for m, data in windows.items():
        age = data['age'][:, -1, 0]
        groups = {'0_to_15': (age >= 0) & (age <= 15),
                  '15_to_30': (age > 15) & (age <= 30),
                  '30_to_60': (age > 30) & (age <= 60), 'no_observation': ~np.isfinite(age)}
        result['by_endpoint_age'][m] = {name: metrics(np.asarray(labels)[keep], np.asarray(predictions)[keep])
                                      for name, keep in groups.items() if keep.any()}
    return result
