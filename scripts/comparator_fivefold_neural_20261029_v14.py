"""V14 official TRAIN/VAL calibration, fresh FIT refit, and label-blind inference.

The fixed-epoch refit transfers a validation-selected budget and learning-rate
prefix. It does not claim an equivalent optimization trajectory or a stationary
refit endpoint. No worker resumes or retries an existing scientific run.
"""
import gc
import json
import math
import os
import time
from pathlib import Path

from scripts.comparator_convergence_criteria_20261027_v2 import assess
from scripts.comparator_repair_policy_20261028 import selected, stopping
from scripts.run_comparator_stability_extension_20261028_v9 import training_components

MODELS = ('GRU_Fusion', 'Transformer_Fusion', 'ModernTCN_Fusion')


def runtime():
    global np, torch, common, parent, base, core, metric
    import numpy as np
    import torch
    from scripts import comparator_fivefold_common_20261029_v14 as common
    from scripts import run_comparator_convergence_20261027_v2 as parent
    from scripts.sensor_fusion_stability_20261027 import metric
    base, core = parent.base, parent.core


def calibration_stop(history, cfg):
    legacy = assess(history, cfg['convergence'])
    training = training_components(legacy, cfg['convergence'])
    result = stopping(history, cfg['calibration_stopping'])
    accepted = training['passed'] and result['validation_plateau']
    exhausted = history[-1]['epoch'] >= cfg['calibration_stopping']['maximum_epochs']
    result.update(stop=bool(accepted or exhausted), accepted=bool(accepted),
                  training_components=training, legacy_mixed_gate=legacy,
                  reason='TRAINING_STABLE_AND_VAL_PLATEAU' if accepted else
                         'BOUNDED_BUDGET_HOLD' if exhausted else 'CONTINUE')
    return result


def refit_plan(history, cfg):
    chosen = selected(history)
    epoch = chosen['epoch']
    assert [r['epoch'] for r in history] == list(range(1, len(history) + 1))
    rates = [r['learning_rate'] for r in history[:epoch]]
    assert len(rates) == epoch and all(math.isfinite(r) and r > 0 for r in rates)
    return dict(selected_epoch=epoch, base_learning_rates=rates,
                selected_VAL=chosen['dev_ema'], EMA_decay=cfg['EMA_decay'],
                EMA_warmup_steps=cfg['EMA_warmup_steps'], evaluation_window=3,
                evaluation_warm_epochs=10,
                rule='minimum hard weighted VAL CE; exact ties retain earlier epoch',
                initialization='fresh random initialization using the same seed; no calibration weights',
                transfer='same_epochs heuristic, not equivalent optimization trajectories',
                refit_scheduler='replay actual calibration learning_rate prefix; no FIT scheduler feedback',
                refit_terminal_stationarity_claimed=False)


def _start(cfg, stage, protocol, fold, model, seed):
    assert model in MODELS and model in cfg['models']
    assert protocol in cfg['protocols'] and fold in cfg['folds'] and seed in cfg['seeds']
    out = Path(common.case_dir(cfg, stage, protocol, fold, model, seed))
    out.mkdir(parents=True, exist_ok=True)
    # O_EXCL is the admission authority even if two workers race.
    with (out / 'STARTED.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), unix=time.time(), stage=stage), stream)
    return out


def _files(out, names):
    return {str((out / name).resolve()): common.sha(out / name) for name in names}


def _receipt(out, expected):
    receipt = common.read(out / 'COMPLETE.json')
    assert receipt['status'] == expected, (out, receipt['status'])
    for path, digest in receipt['files'].items():
        assert common.sha(path) == digest, path
    return receipt


def _train_epoch(state, data, cfg, seed, epoch):
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0])).permutation(data['train'])
    numerator = denominator = 0.
    for offset in range(0, len(order), cfg['batch_size']):
        loss, mass = parent.update(state, data, order[offset:offset + cfg['batch_size']])
        numerator += loss * mass
        denominator += mass
    assert denominator > 0
    return numerator / denominator


def _budget(cfg, started):
    assert time.monotonic() - started < cfg['budget']['worker_seconds'], 'Registered worker budget exceeded'


def calibrate(cfg, protocol, fold, model, seed, slot):
    runtime()
    common.check(cfg)
    lock = common.gpu_lock(cfg, slot, kind='neural')
    try:
        out = _start(cfg, 'calibration', protocol, fold, model, seed)
        started = time.monotonic()
        data = common.load_selection(cfg, protocol, fold)
        state = parent.new(cfg, model, data['prior'], seed, 'cuda')
        history, snapshots = [], []
        best = float('inf')
        weights = data['weights'].double().cpu().numpy()
        for epoch in range(1, cfg['calibration_stopping']['maximum_epochs'] + 1):
            tick = time.monotonic()
            rate = state['opt'].param_groups[0]['lr']
            before = {n: v.detach().clone() for n, v in state['model'].named_parameters()}
            ema_before = {n: v.detach().clone() for n, v in state['ema'].model.named_parameters()}
            train_ce = _train_epoch(state, data, cfg, seed, epoch)
            raw_delta = parent.relative_delta(state['model'], before)
            ema_delta = parent.relative_delta(state['ema'].model, ema_before)
            snapshots = (snapshots + [parent.snapshot(state, epoch)])[-3:]
            rng = core.rng()
            evaluation = parent.evaluation_model(state, snapshots, epoch)
            fit = base.infer(evaluation, data, data['train'])
            val = base.infer(evaluation, data, data['evaluation'])
            fit_score, _ = metric(fit, data['fit_truth'], weights)
            val_score, _ = metric(val, data['dev_truth'], weights)
            assert val_score['weighted_CE'] is not None and math.isfinite(val_score['weighted_CE'])
            row = dict(epoch=epoch, stage='source', stage_epoch=epoch, learning_rate=rate,
                       train_ce=train_ce, fit_ema=fit_score, dev_ema=val_score,
                       raw_relative_parameter_change=raw_delta, EMA_relative_parameter_change=ema_delta,
                       seconds=time.monotonic() - tick, global_step=state['step'])
            history.append(row)
            if val_score['weighted_CE'] < best:
                best = val_score['weighted_CE']
                torch.save(dict(model=evaluation.state_dict(), epoch=epoch, model_name=model),
                           out / 'selected_model.pt')
                np.save(out / 'selected_VAL_EMA.npy', val)
            state['scheduler'].step(val_score['weighted_CE'])
            row['next_learning_rate'] = state['opt'].param_groups[0]['lr']
            core.restore_rng(rng)
            common.save(out / 'history.json', history)
            parent.checkpoint(out / 'latest.pt', state, epoch=epoch, causal_ema_snapshots=snapshots)
            decision = calibration_stop(history, cfg)
            common.save(out / 'PROGRESS.json', dict(epoch=epoch, stopping=decision,
                        selected_epoch=selected(history)['epoch'], elapsed_seconds=time.monotonic() - started))
            print(dict(stage='calibration', model=model, protocol=protocol, fold=fold, seed=seed,
                       epoch=epoch, VAL=val_score, training_stable=decision['training_components']['passed'],
                       validation_plateau=decision['validation_plateau']), flush=True)
            del evaluation, before, ema_before, fit, val
            _budget(cfg, started)
            if decision['stop']:
                break
        plan = refit_plan(history, cfg)
        payload = torch.load(out / 'selected_model.pt', map_location='cpu', weights_only=False)
        assert payload['epoch'] == plan['selected_epoch'] and payload['model_name'] == model
        evaluation = parent.model(cfg, model, data['prior'], 'cuda')
        evaluation.load_state_dict(payload['model'])
        probabilities = np.load(out / 'selected_VAL_EMA.npy', allow_pickle=False)
        np.testing.assert_array_equal(probabilities, base.infer(evaluation, data, data['evaluation']))
        verified, _ = metric(probabilities, data['dev_truth'], weights)
        assert verified == plan['selected_VAL']
        common.save(out / 'REFIT_PLAN.json', plan)
        common.save(out / 'SELECTED_VAL_METRICS.json', verified)
        common.check(cfg)
        result = dict(status='COMPLETE_CALIBRATION' if decision['accepted'] else 'COMPLETE_HOLD',
                      model=model, protocol=protocol, fold=fold, seed=seed, epochs=len(history),
                      selected_epoch=plan['selected_epoch'], stopping=decision,
                      training_components=decision['training_components'],
                      legacy_terminal_optimization=decision['legacy_mixed_gate'],
                      selected_checkpoint=selected(history), selected_checkpoint_replay_exact=True,
                      official_Test_rows_read=0, Test_labels_parsed=False,
                      elapsed_seconds=time.monotonic() - started,
                      files=_files(out, ['history.json', 'latest.pt', 'selected_model.pt',
                                        'selected_VAL_EMA.npy', 'SELECTED_VAL_METRICS.json', 'REFIT_PLAN.json']))
        common.save(out / 'COMPLETE.json', result)
        return result
    finally:
        lock.close()


def refit(cfg, protocol, fold, model, seed, slot):
    runtime()
    common.check(cfg)
    source = Path(common.case_dir(cfg, 'calibration', protocol, fold, model, seed))
    receipt = _receipt(source, 'COMPLETE_CALIBRATION')
    assert receipt['stopping']['accepted']
    plan = common.read(source / 'REFIT_PLAN.json')
    assert plan == refit_plan(common.read(source / 'history.json'), cfg)
    lock = common.gpu_lock(cfg, slot, kind='neural')
    try:
        out = _start(cfg, 'refit', protocol, fold, model, seed)
        started = time.monotonic()
        data = common.load_refit(cfg, protocol, fold)
        state = parent.new(cfg, model, data['prior'], seed, 'cuda')
        history, snapshots = [], []
        for epoch, rate in enumerate(plan['base_learning_rates'], 1):
            for group in state['opt'].param_groups:
                group['lr'] = rate
            train_ce = _train_epoch(state, data, cfg, seed, epoch)
            snapshots = (snapshots + [parent.snapshot(state, epoch)])[-3:]
            history.append(dict(epoch=epoch, learning_rate=rate, train_ce=train_ce,
                                global_step=state['step']))
            common.save(out / 'history.json', history)
            parent.checkpoint(out / 'latest.pt', state, epoch=epoch, causal_ema_snapshots=snapshots,
                              scheduler_feedback_disabled=True, refit_plan=plan)
            common.save(out / 'PROGRESS.json', dict(epoch=epoch, target_epochs=plan['selected_epoch'],
                        elapsed_seconds=time.monotonic() - started))
            _budget(cfg, started)
        assert len(history) == plan['selected_epoch']
        evaluation = parent.evaluation_model(state, snapshots, plan['selected_epoch'])
        sample = data['train'][:cfg['batch_size']]
        expected = base.infer(evaluation, data, sample)
        torch.save(dict(model=evaluation.state_dict(), epoch=plan['selected_epoch'], model_name=model),
                   out / 'final_model.pt')
        restored = parent.model(cfg, model, data['prior'], 'cuda')
        restored.load_state_dict(torch.load(out / 'final_model.pt', map_location='cpu', weights_only=False)['model'])
        np.testing.assert_array_equal(expected, base.infer(restored, data, sample))
        common.check(cfg)
        result = dict(status='COMPLETE_REFIT', model=model, protocol=protocol, fold=fold, seed=seed,
                      epochs=len(history), random_initialization=True, calibration_weights_loaded=False,
                      calibration_receipt_sha256=common.sha(source / 'COMPLETE.json'),
                      calibration_training_stable=True, refit_terminal_stationarity_claimed=False,
                      budget_transfer=plan['transfer'], FIT_sample_serialization_exact=True,
                      official_Test_rows_read=0, Test_labels_parsed=False,
                      elapsed_seconds=time.monotonic() - started,
                      files=_files(out, ['history.json', 'latest.pt', 'final_model.pt']))
        common.save(out / 'COMPLETE.json', result)
        return result
    finally:
        lock.close()


def predict(cfg, protocol, fold, model, seed, slot):
    runtime()
    common.check(cfg)
    common.check_training_lock(cfg)
    source = Path(common.case_dir(cfg, 'refit', protocol, fold, model, seed))
    receipt = _receipt(source, 'COMPLETE_REFIT')
    lock = common.gpu_lock(cfg, slot, kind='neural')
    try:
        out = _start(cfg, 'prediction', protocol, fold, model, seed)
        started = time.monotonic()
        data = common.load_test(cfg, protocol, fold)
        assert torch.all(data['tensors']['labels'] == -100)
        payload = torch.load(source / 'final_model.pt', map_location='cpu', weights_only=False)
        assert payload['model_name'] == model and payload['epoch'] == receipt['epochs']
        evaluation = parent.model(cfg, model, data['prior'], 'cuda')
        evaluation.load_state_dict(payload['model'])
        probabilities = base.infer(evaluation, data, data['evaluation'])
        np.save(out / 'test_ema.npy', probabilities)
        restored = parent.model(cfg, model, data['prior'], 'cuda')
        restored.load_state_dict(torch.load(source / 'final_model.pt', map_location='cpu', weights_only=False)['model'])
        np.testing.assert_array_equal(np.load(out / 'test_ema.npy', allow_pickle=False),
                                      base.infer(restored, data, data['evaluation']))
        _budget(cfg, started)
        common.check(cfg)
        common.check_training_lock(cfg)
        result = dict(status='COMPLETE_PREDICTION', model=model, protocol=protocol, fold=fold, seed=seed,
                      Test_labels_parsed=False, rows=len(probabilities), deserialized_probabilities_exact=True,
                      refit_receipt_sha256=common.sha(source / 'COMPLETE.json'),
                      elapsed_seconds=time.monotonic() - started, files=_files(out, ['test_ema.npy']))
        common.save(out / 'COMPLETE.json', result)
        return result
    finally:
        lock.close()


def witness(cfg, device, work_dir):
    """Synthetic only; callers own runtime setup, resource admission and receipt."""
    runtime()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=False)
    records = []
    for model in MODELS:
        batch = parent.old.synthetic_batch(device, 21)
        data = dict(tensors=dict(**batch, labels=torch.arange(21, device=device) % 7),
                    prior=np.ones(7) / 7, weights=torch.arange(1, 8, device=device).float(), train=np.arange(21))
        state = parent.new(cfg, model, data['prior'], 17, device)
        snapshots, history = [], []
        for epoch in range(1, 12):
            rate = state['opt'].param_groups[0]['lr']
            # Three real updates suffice for a nontrivial optimizer/EMA state.
            if epoch in (9, 10, 11):
                parent.update(state, data, data['train'])
                snapshots.append(parent.snapshot(state, epoch))
            history.append(dict(epoch=epoch, learning_rate=rate,
                                dev_ema=dict(weighted_CE=1. if epoch != 10 else .9)))
            state['scheduler'].step(1.)
            history[-1]['next_learning_rate'] = state['opt'].param_groups[0]['lr']
        plan = refit_plan(history, cfg)
        assert plan['selected_epoch'] == 10
        assert plan['base_learning_rates'] == [r['learning_rate'] for r in history[:10]]
        assert plan['base_learning_rates'] != [r['next_learning_rate'] for r in history[:10]]
        checkpoint = work / (model + '.pt')
        parent.checkpoint(checkpoint, state, epoch=11, causal_ema_snapshots=snapshots)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        rng = core.rng()
        raw = {n: v.detach().clone() for n, v in state['ema'].model.state_dict().items()}
        evaluation = parent.evaluation_model(state, snapshots, 11)
        base.equal_tree(raw, state['ema'].model.state_dict())
        base.equal_tree(rng, core.rng())
        for name, value in evaluation.state_dict().items():
            expected = torch.stack([s['model'][name].double() for s in snapshots]).mean(0).to(value) if value.is_floating_point() else raw[name]
            assert torch.equal(value, expected), name
        expected = parent.update(state, data, data['train'])
        expected_rng = core.rng()
        restored = parent.restore(cfg, model, data['prior'], 17, payload, device)
        base.equal_tree(expected, parent.update(restored, data, data['train']))
        base.equal_tree(expected_rng, core.rng())
        for key in ('model', 'opt', 'scheduler'):
            base.equal_tree(state[key].state_dict(), restored[key].state_dict())
        base.equal_tree(state['ema'].model.state_dict(), restored['ema'].model.state_dict())
        assert state['ema'].updates == restored['ema'].updates and state['step'] == restored['step']
        restored_before_update = parent.restore(cfg, model, data['prior'], 17, payload, device)
        replay = parent.evaluation_model(restored_before_update, payload['causal_ema_snapshots'], 11)
        np.testing.assert_array_equal(base.infer(evaluation, data, data['train']), base.infer(replay, data, data['train']))
        records.append(dict(model=model, complete_next_update_exact=True,
                            causal_three_view_and_snapshot_reload_exact=True, actual_LR_prefix_verified=True,
                            live_EMA_and_RNG_preserved=True, real_data_rows_read=0))
        del state, restored, restored_before_update, evaluation, replay, payload, data, batch
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    return records


def preflight(cfg):
    """Profile actual largest official splits; no scientific weights are saved."""
    runtime()
    largest = {}
    for stage, loader in [('calibration', common.load_selection), ('refit', common.load_refit)]:
        candidates = []
        for protocol in cfg['protocols']:
            for fold in cfg['folds']:
                data = loader(cfg, protocol, fold, device='cpu')
                train, val = len(data['train']), len(data.get('evaluation', []))
                candidates.append((train + val, train, val, protocol, fold))
                del data
        # Profile the largest TRAIN and largest VAL separately if they differ.
        largest[stage] = list({max(candidates, key=lambda r: r[1]), max(candidates, key=lambda r: r[2]),
                               max(candidates, key=lambda r: r[0])})
    records = []
    lock = common.gpu_lock(cfg, 0, kind='neural')
    try:
        for stage, candidates in largest.items():
            for _, train_count, val_count, protocol, fold in sorted(candidates):
                loader = common.load_selection if stage == 'calibration' else common.load_refit
                data = loader(cfg, protocol, fold)
                for model in MODELS:
                    torch.cuda.reset_peak_memory_stats()
                    state = parent.new(cfg, model, data['prior'], 17, 'cuda')
                    times = []
                    assert len(data['train']) >= cfg['batch_size']
                    for _ in range(3):
                        torch.cuda.synchronize()
                        tick = time.monotonic()
                        parent.update(state, data, data['train'][:cfg['batch_size']])
                        torch.cuda.synchronize()
                        times.append(time.monotonic() - tick)
                    tick = time.monotonic()
                    for ids in [data['train'], data.get('evaluation', [])]:
                        if len(ids):
                            base.infer(state['ema'].model, data, ids)
                    torch.cuda.synchronize()
                    inference = time.monotonic() - tick
                    epoch_seconds = max(times[1:]) * math.ceil(train_count / cfg['batch_size']) + inference * 2 + .5
                    maximum = epoch_seconds * cfg['calibration_stopping']['maximum_epochs'] * 1.5 + 60
                    assert maximum < cfg['budget']['worker_seconds'], (stage, model, maximum)
                    records.append(dict(stage=stage, model=model, protocol=protocol, fold=fold,
                                        TRAIN_or_FIT_rows=train_count, VAL_rows=val_count,
                                        batch_size=cfg['batch_size'], step_seconds=times,
                                        FIT_VAL_inference_seconds=inference,
                                        conservative_maximum_worker_seconds=maximum,
                                        peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                    del state
                    gc.collect()
                    torch.cuda.empty_cache()
                del data
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        lock.close()
    return dict(records=records, maximum_worker_seconds=max(r['conservative_maximum_worker_seconds'] for r in records),
                official_Test_rows_read=0, scientific_checkpoints_saved=False)
