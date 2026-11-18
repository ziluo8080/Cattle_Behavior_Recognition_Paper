"""Strict evaluation replay helpers; no tolerance or training-state mutation.

Use only on a separately constructed evaluation model. Training models and their
full-state next-update witnesses must retain their existing exact checks.
"""
from collections.abc import Mapping

import numpy as np
import torch


class PredictionReplayError(AssertionError):
    """An evaluation replay failed an exact, auditable invariant."""


def assert_state_exact(model_state, saved_state):
    """Require identical tensor keys, shapes, dtypes and values (device agnostic)."""
    if not isinstance(model_state, Mapping) or not isinstance(saved_state, Mapping):
        raise PredictionReplayError('state must be a mapping of tensors')
    if set(model_state) != set(saved_state):
        raise PredictionReplayError('state keys differ')
    for key in saved_state:
        actual, expected = model_state[key], saved_state[key]
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise PredictionReplayError(f'state {key}: non-tensor value')
        if actual.dtype != expected.dtype:
            raise PredictionReplayError(f'state {key}: dtype differs')
        if actual.shape != expected.shape:
            raise PredictionReplayError(f'state {key}: shape differs')
        if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise PredictionReplayError(f'state {key}: tensor values differ')


def prepare_evaluation_replay(model, saved_state):
    """Freeze an already-loaded evaluation model after checking saved tensor state.

The caller owns model construction and strict checkpoint loading. Checking before
changing flags catches accidental wrong checkpoints; checking afterwards proves
that aligning evaluation flags did not change tensor state. Returns the same model.
"""
    assert_state_exact(model.state_dict(), saved_state)
    model.eval().requires_grad_(False)
    assert_state_exact(model.state_dict(), saved_state)
    if any(module.training for module in model.modules()):
        raise PredictionReplayError('evaluation model contains a training module')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise PredictionReplayError('evaluation model contains trainable parameters')
    return model


def verify_probabilities(reference, replay):
    """Require exact N x 7 probabilities, preserving both original arrays.

No clipping, normalization, tolerance, or replacement is performed. The failure
message reports changed elements and rows even when every predicted label agrees.
"""
    reference, replay = np.asarray(reference), np.asarray(replay)
    if reference.shape != replay.shape or reference.ndim != 2 or reference.shape[1] != 7:
        raise PredictionReplayError('probability shape differs or is not N x 7')
    if reference.shape[0] == 0:
        raise PredictionReplayError('probability arrays must contain at least one row')
    if reference.dtype != replay.dtype or reference.dtype.kind != 'f':
        raise PredictionReplayError('probability dtype differs or is not floating point')
    for name, values in [('reference', reference), ('replay', replay)]:
        if not np.isfinite(values).all():
            raise PredictionReplayError(f'{name} probabilities contain non-finite values')
        if ((values < 0) | (values > 1)).any():
            raise PredictionReplayError(f'{name} probabilities are outside [0, 1]')
    changed = reference != replay
    diagnostics = dict(
        bitwise_exact=bool(np.array_equal(reference, replay)),
        mismatched_count=int(changed.sum()),
        mismatched_rows=int(np.any(changed, axis=1).sum()),
        labels_identical=bool(np.array_equal(reference.argmax(1), replay.argmax(1))),
    )
    if not diagnostics['bitwise_exact']:
        raise PredictionReplayError(f'exact probability replay failed: {diagnostics}')
    diagnostics['reason'] = 'EXACT_PROBABILITY_REPLAY'
    return diagnostics
