"""Forty TRAIN/VAL choices, then sixty locked XGBoost/TCN refits."""
import argparse
import fcntl
import gc
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import sklearn
import xgboost as xgb
from q3_sensor.evaluation import metrics
from q3_sensor.protocol import MODALITIES, sha
from scripts.sensor_tcn_model_fast_20261014 import TCNFusion
from scripts.sensor_classic_features_20261012 import descriptors
from scripts.run_sensor_classic_comparison_20261012 import data_for, fallback
from scripts import run_sensor_architecture_comparison_20261011 as neural
from scripts.run_sensor_history_controls_20261011 import read_json, write_json

CONFIG = Path('configs/sensor_xgboost_tcn_fast_20261014.json')


class ExactMacroStop(xgb.callback.TrainingCallback):
    """Select on exact Python F1, avoiding XGBoost's rounded eval strings."""
    def __init__(self, evaluation, labels, available, prior, patience):
        self.evaluation, self.labels, self.available = evaluation, labels, available
        self.prior, self.patience = prior, patience
        self.best, self.best_epoch, self.stale, self.history = -1., 0, 0, []

    def after_iteration(self, model, epoch, evals_log):
        prediction = model.predict(self.evaluation).argmax(1) + 1
        prediction[~self.available] = self.prior
        score = metrics(self.labels, prediction)['macro_f1']
        self.history.append(dict(epoch=epoch + 1, val_macro_f1=score))
        if score > self.best:
            self.best, self.best_epoch, self.stale = score, epoch + 1, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def environment():
    if Path.cwd() != Path('/q3/mmcows-sensor-behavior-20261007'):
        raise RuntimeError('Cloud-only execution')
    spec = read_json('configs/sensor_xgboost_tcn_env_20261014.json')
    versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        sklearn=sklearn.__version__, xgboost=xgb.__version__)
    for key, value in versions.items():
        assert value == spec[key], (key, value, spec[key])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    return versions


def witness(cfg, root, output=None):
    versions = environment()
    neural.seed_all(17)
    batch = {m: dict(x=torch.randn(14, 60, d, device='cuda'),
        valid=torch.ones(14, 60, d, dtype=torch.bool, device='cuda'),
        fresh=torch.ones(14, 60, 1, device='cuda'), age=torch.rand(14, 60, 1, device='cuda') * 59)
        for m, d in zip(MODALITIES, (3, 6, 1))}
    for fields in batch.values():
        fields['valid'][0] = False
    model = TCNFusion([1 / 7] * 7).cuda()
    count = sum(p.numel() for p in model.parameters())
    assert count == cfg['expected_parameters']['TCN-Window-VAL']
    model.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(model(batch)[0].cpu().numpy(), model.prior_logits.cpu().numpy())
        prefix_errors = []
        for modality in MODALITIES:
            original = model.encoders[modality].sequence(batch[modality])[:, :30]
            changed = {k: v.clone() for k, v in batch[modality].items()}
            changed['x'][:, 30:] += 20
            changed['age'][:, 30:] = 0
            prefix_errors.append(float((original - model.encoders[modality].sequence(changed)[:, :30]).abs().max()))
        assert max(prefix_errors) == 0
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(batch), torch.arange(14, device='cuda') % 7)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    rng = np.random.default_rng(17)
    values = rng.normal(size=(42, 110)).astype(np.float32)
    matrix = xgb.DMatrix(values, label=np.arange(42) % 7)
    booster = xgb.train(dict(objective='multi:softprob', num_class=7, tree_method='hist',
        device='cpu', max_depth=3, seed=17, nthread=4), matrix, num_boost_round=3)
    pred = booster.predict(matrix)
    assert pred.shape == (42, 7) and np.isfinite(pred).all()
    info = dict(status='PASS', synthetic_only=True, scientific_F1=False, versions=versions,
        TCN_parameters=count, receptive_field=125, prefix_max_errors=prefix_errors,
        finite_gradients=True, optimizer_steps=3, all_missing_prior_exact=True,
        xgboost_prediction_shape=list(pred.shape), env_spec_sha256=sha(cfg['environment_spec']))
    root.mkdir(parents=True, exist_ok=True)
    write_json(Path(output) if output else root / 'ENV_WITNESS.json', info)
    del model, optimizer, batch
    gc.collect()
    torch.cuda.empty_cache()
    print('XGBOOST_TCN_ENV_WITNESS_PASS', json.dumps(info), flush=True)


def setup(cfg, root):
    environment()
    validation = read_json(cfg['optimization_validation'])
    assert validation['status'] == 'PASS' and validation['strict_deterministic_algorithms']
    for path, digest in validation['source_sha256'].items():
        assert sha(path) == digest, path
    assert read_json(root / 'ENV_WITNESS.json')['status'] == 'PASS'
    assert read_json(root / 'INDEPENDENT_ENV_WITNESS.json')['status'] == 'PASS'
    base = read_json(cfg['base_config'])
    parent = Path(cfg['reuse_architecture_root'])
    assert sha(parent / 'run_identity.json') == base['reuse_architecture_identity_sha256']
    assert sha(parent / 'METHOD_LOCK.json') == base['reuse_architecture_lock_sha256']
    old_identity = read_json(Path(base['output']) / 'run_identity.json')
    old_sources = {str(Path(p).resolve()): digest for p, digest in old_identity['sources'].items()}
    assert sha(cfg['base_config']) == old_sources[str(Path(cfg['base_config']).resolve())]
    assert sha('scripts/run_sensor_classic_comparison_20261012.py') == old_sources[str(Path('scripts/run_sensor_classic_comparison_20261012.py').resolve())]
    inherited = read_json(Path(cfg['reuse_architecture_root']) / 'run_identity.json')['source_sha256']
    for path, digest in inherited.items():
        assert sha(path) == digest, path
    files = [CONFIG, Path(__file__), Path('scripts/sensor_tcn_model_fast_20261014.py'),
        Path(cfg['optimization_validation']), Path('scripts/check_sensor_tcn_fast_20261014.py'),
        Path(cfg['base_config']),
        Path('scripts/run_sensor_classic_comparison_20261012.py'),
        Path(cfg['environment_spec']), Path('scripts/sensor_classic_features_20261012.py'),
        root / 'ENV_WITNESS.json', root / 'INDEPENDENT_ENV_WITNESS.json']
    sources = {str(p): sha(p) for p in files}
    sources.update(inherited)
    identity = dict(config=cfg, sources=sources, environment_spec_sha256=sha(cfg['environment_spec']))
    if (root / 'run_identity.json').exists():
        assert read_json(root / 'run_identity.json') == identity, 'New implementation changed after freeze'
    else:
        write_json(root / 'run_identity.json', identity)
    jobs = [f'{phase}/{protocol}/{fold}/{model}/{suffix}'
        for phase in ('selection', 'formal') for protocol in cfg['protocols'] for fold in cfg['folds']
        for model in cfg['models'] for suffix in (['candidate0', 'candidate1'] if phase == 'selection'
            else [f'seed{s}' for s in cfg['formal_seeds']])]
    manifest = dict(jobs=jobs, selection_jobs=40, formal_jobs=60, max_parallel=1,
        identity_sha256=sha(root / 'run_identity.json'), no_Test_selection=True)
    if (root / 'MANIFEST.json').exists():
        assert read_json(root / 'MANIFEST.json') == manifest
    else:
        write_json(root / 'MANIFEST.json', manifest)
    return base


def xgb_job(cfg, root, phase, protocol, fold, suffix, data, candidate, seed, rounds):
    job = f'{phase}/{protocol}/{fold}/XGBoost-Window/{suffix}'
    directory = root / job
    if (directory / 'COMPLETE.json').exists():
        neural.verify_job(directory, root, data['identity'])
        return read_json(directory / 'result.json')
    directory.mkdir(parents=True, exist_ok=True)
    inputs = dict(job_id=job, seed=seed, candidate_index=candidate, epochs_cap=rounds,
        data_identity=data['identity'], identity_sha256=sha(root / 'run_identity.json'),
        method_lock_sha256=sha(root / 'METHOD_LOCK.json') if phase == 'formal' else None)
    if (directory / 'INPUTS.json').exists():
        assert read_json(directory / 'INPUTS.json') == inputs
    write_json(directory / 'INPUTS.json', inputs)
    neural.progress(root, job, state='running', pid=__import__('os').getpid())
    started = time.monotonic()
    tr, ev = data['train_ids'], data['evaluation_ids']
    values, names = descriptors(data['arrays'], data['normalizer'], False)
    mask, prior = fallback(data, False)
    counts = np.bincount(data['labels'][tr], minlength=8)[1:8]
    assert (counts > 0).all()
    weights = counts.sum() / (7 * counts)
    train = xgb.DMatrix(values[tr], label=data['labels'][tr] - 1,
        weight=weights[data['labels'][tr] - 1], feature_names=names)
    evaluation = xgb.DMatrix(values[ev], label=data['labels'][ev] - 1, feature_names=names)
    params = {k: v for k, v in cfg['xgboost'].items() if k in
        ('objective', 'num_class', 'tree_method', 'device', 'eta', 'max_bin', 'subsample',
         'colsample_bytree', 'min_child_weight', 'lambda', 'nthread', 'disable_default_eval_metric')}
    params.update(cfg['xgboost']['candidates'][candidate], seed=seed)
    history = []
    if phase == 'selection':
        callback = ExactMacroStop(evaluation, data['labels'][ev], mask[ev], prior,
            cfg['xgboost']['early_stopping_rounds'])
        booster = xgb.train(params, train, num_boost_round=rounds, callbacks=[callback], verbose_eval=False)
        selected, trained, validation = callback.best_epoch, len(callback.history), callback.best
        history = callback.history
        booster = booster[:selected]
        booster.set_attr(best_iteration=str(selected - 1), best_score=repr(validation))
    else:
        booster = xgb.train(params, train, num_boost_round=rounds)
        selected, trained, validation = rounds, rounds, None
    assert booster.num_boosted_rounds() == selected
    prediction = booster.predict(xgb.DMatrix(values, feature_names=names)).argmax(1) + 1
    prediction[~mask] = prior
    table = data['targets'].iloc[ev][['cow_id', 'timestamp', 'label']].reset_index(drop=True)
    table['prediction'] = prediction[ev]
    table.to_csv(directory / 'predictions.csv', index=False)
    result = dict(inputs, model='XGBoost-Window', phase=phase, protocol=protocol, fold=fold,
        selected_epoch=selected, trained_epochs=trained, selected_validation_macro_f1=validation,
        metrics=metrics(data['labels'][ev], prediction[ev]), Fit_metrics=metrics(data['labels'][tr], prediction[tr]),
        parameters=None, feature_count=len(names), feature_names=names, hyperparameters=params,
        class_weights=weights.tolist(), prior_class=prior, seconds=time.monotonic() - started)
    if phase == 'selection':
        np.testing.assert_allclose(result['metrics']['macro_f1'], validation, rtol=0, atol=2e-15)
    booster.save_model(directory / 'model.ubj')
    write_json(directory / 'result.json', result)
    write_json(directory / 'history.json', history)
    write_json(directory / 'COMPLETE.json', dict(job_id=job, identity_sha256=sha(root / 'run_identity.json'),
        data_identity=data['identity'], outputs={p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}))
    neural.progress(root, job, state='completed', macro_f1=result['metrics']['macro_f1'], seconds=result['seconds'])
    print('XGBOOST_JOB_COMPLETE', job, result['metrics']['macro_f1'], selected, result['seconds'], flush=True)
    return result


def lock_method(cfg, root):
    assert read_json(root / 'SELECTION_COMPLETE.json')['expected_jobs'] == 40
    chosen, cases = {}, []
    for protocol in cfg['protocols']:
        for fold in cfg['folds']:
            for model in cfg['models']:
                options = []
                for index in range(2):
                    p = root / f'selection/{protocol}/{fold}/{model}/candidate{index}'
                    result = read_json(p / 'result.json')
                    neural.verify_job(p, root, result['data_identity'])
                    options.append((index, result))
                    cases.append(dict(job_id=result['job_id'], complete_sha256=sha(p / 'COMPLETE.json')))
                index, best = min(options, key=lambda r: (-r[1]['selected_validation_macro_f1'], r[1]['selected_epoch'], r[0]))
                chosen[f'{protocol}/{fold}/{model}'] = dict(candidate_index=index, epochs=best['selected_epoch'],
                    learning_rate=cfg['learning_rates'][index] if model == 'TCN-Window-VAL' else None,
                    selection_job=best['job_id'], validation_macro_f1=best['selected_validation_macro_f1'])
    lock = dict(identity_sha256=sha(root / 'run_identity.json'), selected=chosen, candidates=cases,
        selection_jobs=40, formal_jobs=60, historical_Test_exposure=True,
        created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if (root / 'METHOD_LOCK.json').exists():
        assert read_json(root / 'METHOD_LOCK.json')['selected'] == chosen
    else:
        write_json(root / 'METHOD_LOCK.json', lock)
    print('XGBOOST_TCN_METHOD_LOCK_COMPLETE', flush=True)


def functional_probe(cfg, root, base):
    """Small real TRAIN/VAL functionality test, never a scientific score."""
    data = data_for(base, 'selection', 'TS', 'fold_1')
    assert set(data['targets'].split) == {'train', 'val'}
    tr = np.concatenate([data['train_ids'][data['labels'][data['train_ids']] == k][:16] for k in range(1, 8)])
    ev = data['evaluation_ids'][:128]
    values, names = descriptors(data['arrays'], data['normalizer'], False)
    available, prior = fallback(data, False)
    counts = np.bincount(data['labels'][data['train_ids']], minlength=8)[1:8]
    weights = counts.sum() / (7 * counts)
    train = xgb.DMatrix(values[tr], label=data['labels'][tr] - 1,
        weight=weights[data['labels'][tr] - 1], feature_names=names)
    validation = xgb.DMatrix(values[ev], label=data['labels'][ev] - 1, feature_names=names)
    callback = ExactMacroStop(validation, data['labels'][ev], available[ev], prior, 20)
    booster = xgb.train(dict(objective='multi:softprob', num_class=7, tree_method='hist',
        device='cpu', max_depth=3, nthread=4, seed=17, eta=.05, disable_default_eval_metric=1),
        train, num_boost_round=3, callbacks=[callback])[:callback.best_epoch]
    booster.save_model(root / 'FUNCTIONAL_XGBOOST.ubj')
    restored = xgb.Booster(params={'nthread': 4, 'device': 'cpu'})
    restored.load_model(root / 'FUNCTIONAL_XGBOOST.ubj')
    np.testing.assert_array_equal(booster.predict(validation), restored.predict(validation))
    neural.seed_all(17)
    batch = {m: {k: torch.from_numpy(v[tr]).cuda() for k, v in fields.items()} for m, fields in data['arrays'].items()}
    model = TCNFusion(counts / counts.sum()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    targets = torch.as_tensor(data['labels'][tr] - 1, dtype=torch.long, device='cuda')
    class_weights = torch.as_tensor(np.sqrt(weights), dtype=torch.float32, device='cuda')
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(batch), targets, weight=class_weights)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    model.eval()
    validation_batch = {m: {k: torch.from_numpy(v[ev]).cuda() for k, v in fields.items()} for m, fields in data['arrays'].items()}
    with torch.no_grad():
        logits = model(validation_batch)
        assert logits.shape == (len(ev), 7) and torch.isfinite(logits).all()
    record = read_json(root / 'SYNTHETIC_SANITY.json')
    record['synthetic_only'] = False
    record['functional_only'] = True
    record['synthetic_component'] = 'SYNTHETIC_SANITY.json'
    record['real_TRAIN_VAL_functionality'] = dict(train_rows=len(tr), val_rows=len(ev),
        no_Test_targets=True, scientific_result=False, XGBoost_rounds_observed=len(callback.history),
        exact_metric_callback=True, UBJ_roundtrip_predictions_exact=True, TCN_updates=3,
        feature_width=len(names), TCN_VAL_shape=list(logits.shape), TCN_VAL_finite=True)
    write_json(root / 'SANITY.json', record)
    del model, optimizer, batch, validation_batch, data
    gc.collect()
    torch.cuda.empty_cache()
    print('XGBOOST_TCN_REAL_TRAIN_FUNCTIONAL_PROBE_PASS', flush=True)


def phase_run(cfg, root, base, phase):
    sanity = read_json(root / 'SANITY.json')
    assert sanity['status'] == 'PASS' and sanity['real_TRAIN_VAL_functionality']['UBJ_roundtrip_predictions_exact']
    lock = None
    if phase == 'formal':
        lock = read_json(root / 'METHOD_LOCK.json')
        assert lock['identity_sha256'] == sha(root / 'run_identity.json') and len(lock['candidates']) == 40
        for candidate in lock['candidates']:
            assert sha(root / candidate['job_id'] / 'COMPLETE.json') == candidate['complete_sha256']
    original_factory = neural.make_model
    neural.make_model = lambda kind, prior: TCNFusion(prior)
    try:
        for protocol in cfg['protocols']:
            for fold in cfg['folds']:
                data = data_for(base, phase, protocol, fold)
                for model in cfg['models']:
                    tensor_data = neural.load_data(Path(data['identity']['directory'])) if model == 'TCN-Window-VAL' else None
                    choices = [(i, cfg['selection_seed'], f'candidate{i}') for i in range(2)] if phase == 'selection' else [
                        (lock['selected'][f'{protocol}/{fold}/{model}']['candidate_index'], s, f'seed{s}') for s in cfg['formal_seeds']]
                    for candidate, seed, suffix in choices:
                        rounds = (cfg['max_epochs'] if model == 'TCN-Window-VAL' else cfg['xgboost']['max_rounds']) if phase == 'selection' else lock['selected'][f'{protocol}/{fold}/{model}']['epochs']
                        if model == 'TCN-Window-VAL':
                            neural.train_job(cfg, root, phase, protocol, fold, model, suffix, tensor_data,
                                cfg['learning_rates'][candidate], seed, rounds)
                        else:
                            xgb_job(cfg, root, phase, protocol, fold, suffix, data, candidate, seed, rounds)
                    del tensor_data
                    gc.collect()
                    torch.cuda.empty_cache()
                del data
                gc.collect()
        write_json(root / f'{phase.upper()}_COMPLETE.json', dict(status='COMPLETE',
            expected_jobs=40 if phase == 'selection' else 60, identity_sha256=sha(root / 'run_identity.json')))
    finally:
        neural.make_model = original_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['witness', 'sanity', 'selection', 'lock', 'formal'])
    parser.add_argument('--output')
    args = parser.parse_args()
    cfg = read_json(CONFIG)
    root = Path(cfg['output'])
    if args.phase == 'witness':
        witness(cfg, root, args.output)
        return
    base = setup(cfg, root)
    with (root / '.RUNNING.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.phase == 'sanity':
                witness(cfg, root, root / 'SYNTHETIC_SANITY.json')
                functional_probe(cfg, root, base)
            elif args.phase == 'lock':
                lock_method(cfg, root)
            else:
                phase_run(cfg, root, base, args.phase)
        except Exception:
            if (root / 'QUEUE_STATE.json').exists():
                current = read_json(root / 'QUEUE_STATE.json')['current_job']
                if current['state'] == 'running':
                    neural.progress(root, current['job_id'], state='failed', traceback=traceback.format_exc())
            write_json(root / f'FAILURE_{time.time_ns()}.json', dict(phase=args.phase, traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
