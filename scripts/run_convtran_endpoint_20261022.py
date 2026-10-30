"""Bounded TRAIN/VAL experiment: concat replay, full residual A, endpoint B."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts import run_convtran_context_20261022 as core
from scripts.sensor_convtran_endpoint_interaction_20261022 import (
    EndpointInteractionConvTran, FullSequenceResidualInteractionConvTran,
)
from scripts.sensor_convtran_context_model_20261022 import masked_mean

np, torch = core.np, core.torch
read, save, sha = core.read, core.save, core.sha
ROOT = core.ROOT
CONFIG = Path('configs/sensor_convtran_endpoint_pilot_20261022_v1.json')
MODULE = 'scripts.run_convtran_endpoint_20261022'


def configure():
    old = core.configure()
    cfg = read(CONFIG)
    assert cfg['arms'] == ['TwoRate_Concat', 'FullResidual_A', 'EndpointResidual_B']
    for k in ('protocols', 'fold', 'seeds', 'epochs', 'width', 'heads', 'ff', 'dropout',
              'learning_rate', 'weight_decay', 'batch_size', 'ema_decay', 'ema_warmup_steps'):
        assert cfg[k] == old[k], k
    assert cfg['max_attempts'] == 1 and cfg['max_wall_seconds'] == 14400
    assert all(cfg[k] is False for k in ('Test_allowed', 'additional_folds_allowed',
                                       'formal_refit_allowed', 'automatic_expansion'))
    return cfg


def model_for(arm, cfg, prior):
    if arm == 'TwoRate_Concat':
        return core.make_model(arm, cfg, prior)
    if arm == 'FullResidual_A':
        return FullSequenceResidualInteractionConvTran(cfg, prior)
    if arm == 'EndpointResidual_B':
        return EndpointInteractionConvTran(cfg, prior)
    raise ValueError(arm)


def tracked_files(cfg):
    old = read(Path(cfg['reference_output']) / 'SEAL.json')
    for p, digest in {**old['files'], **old['data']}.items():
        assert sha(p) == digest, f'Original seal changed: {p}'
    paths = list(old['files']) + [str(CONFIG), __file__,
        'scripts/run_convtran_context_20261022.py',
        'scripts/sensor_convtran_endpoint_interaction_20261022.py',
        'scripts/report_convtran_endpoint_20261022.py',
        'tests/test_convtran_endpoint_interaction_20261022.py',
        'tests/test_convtran_endpoint_report_20261022.py']
    return {Path(p).resolve().relative_to(ROOT).as_posix(): sha(p) for p in paths}


def evidence_files(cfg):
    reference = Path(cfg['reference_output'])
    paths = [reference / 'SEAL.json', reference / 'SUMMARY.json']
    for protocol in cfg['protocols']:
        paths.append(reference / 'data' / protocol / 'targets.jsonl')
        for arm in ('Original_ConvTran', 'TwoRate_Concat', 'TwoRate_Interaction'):
            group = reference / 'groups' / protocol / arm
            paths += [group / 'COMPLETE.json', group / 'HISTORY.json']
            paths += [group / f'seed{s}_{v}_final20_predictions.npy' for s in cfg['seeds'] for v in ('raw', 'ema')]
    old_data = read(reference / 'SEAL.json')['data']
    return {**old_data, **{p.as_posix(): sha(p) for p in paths}}


def equal_tree(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    else:
        assert a == b


def optimizer_for(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                             weight_decay=cfg['weight_decay'], eps=1e-8)


def preflight(cfg):
    lock = core.acquire_gpu(); out = Path(cfg['output'])
    assert not (out / 'SEAL.json').exists()
    files = tracked_files(cfg); records = []
    for protocol in cfg['protocols']:
        data = core.data_for(cfg, protocol)
        for arm in cfg['arms']:
            core.seed_all(20261022)
            baseline = model_for('TwoRate_Concat', cfg, data['prior']).cuda().eval()
            core.seed_all(20261022)
            model = model_for(arm, cfg, data['prior']).cuda().eval()
            sample = core.minibatch(data, data['train'][:32])
            with torch.no_grad():
                torch.testing.assert_close(model(sample), baseline(sample), rtol=0, atol=0)
            del baseline
            opt = optimizer_for(model, cfg)
            ema = core.ParameterEMA(model, cfg['ema_decay'], cfg['ema_warmup_steps'])
            losses, step_seconds = [], []
            for i in range(3):
                torch.cuda.synchronize(); tick = time.perf_counter()
                losses.append(core.update(model, opt, ema, data, data['train'][i*256:(i+1)*256])[0])
                torch.cuda.synchronize(); step_seconds.append(time.perf_counter()-tick)
            path = out / 'preflight' / f'{protocol}_{arm}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.model.state_dict(),
                            ema_updates=ema.updates, rng=core.rng()), path)
            expected_loss = core.update(model, opt, ema, data, data['train'][:256])[0]
            expected_rng = core.rng()
            payload = torch.load(path, weights_only=False)
            other = model_for(arm, cfg, data['prior']).cuda(); other.load_state_dict(payload['model'])
            other_opt = optimizer_for(other, cfg); other_opt.load_state_dict(payload['optimizer'])
            other_ema = core.ParameterEMA(other, cfg['ema_decay'], cfg['ema_warmup_steps'])
            other_ema.model.load_state_dict(payload['ema']); other_ema.updates = payload['ema_updates']
            core.restore_rng(payload['rng'])
            assert core.update(other, other_opt, other_ema, data, data['train'][:256])[0] == expected_loss
            equal_tree(model.state_dict(), other.state_dict())
            equal_tree(opt.state_dict(), other_opt.state_dict())
            equal_tree(ema.model.state_dict(), other_ema.model.state_dict())
            assert ema.updates == other_ema.updates
            equal_tree(expected_rng, core.rng())
            records.append(dict(protocol=protocol, arm=arm, train_ce=losses,
                parameters=sum(p.numel() for p in model.parameters()), initial_concat_equal=True,
                serialized_resume_bitwise=True, train_targets=len(data['train']),
                step_seconds=step_seconds, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
            del model, opt, ema, other, other_opt, other_ema, payload, sample
        del data
    estimate = sum(float(np.median(r['step_seconds'])) * math.ceil(r['train_targets']/256) * 20 * 3 for r in records) * 2
    assert estimate < cfg['max_wall_seconds'], 'Preflight cost estimate exceeds fixed budget'
    save(out / 'CUDA_PREFLIGHT.json', dict(status='PASS', records=records, files=files,
        rough_train_eval_seconds_estimate=estimate, scientific_VAL_scores_computed=0,
        gpu=torch.cuda.get_device_name(0), Test_allowed=False))
    lock.close()


def seal(cfg):
    out = Path(cfg['output']); assert not (out / 'SEAL.json').exists()
    witness = read(out / 'CUDA_PREFLIGHT.json')
    assert witness['status'] == 'PASS' and witness['files'] == tracked_files(cfg)
    assert read(out / 'CODE_REVIEW_GATE.json')['status'] == 'PASS'
    assert read(out / 'HOST_TRUST_RECEIPT.json')['trusted_fingerprint_verified'] is True
    data = evidence_files(cfg)
    for name in ('CUDA_PREFLIGHT.json', 'CODE_REVIEW_GATE.json', 'HOST_TRUST_RECEIPT.json'):
        p = out / name; data[p.as_posix()] = sha(p)
    save(out / 'SEAL.json', dict(config=cfg, files=tracked_files(cfg), data=data,
                                created_at_unix=time.time(), Test_allowed=False))


def check_seal(cfg):
    seal = read(Path(cfg['output']) / 'SEAL.json')
    assert seal['config'] == cfg and seal['files'] == tracked_files(cfg)
    for p, digest in seal['data'].items(): assert sha(p) == digest, p


@torch.no_grad()
def diagnostics(model, data, cfg):
    model.eval(); accum = {}
    for start in range(0, len(data['evaluation']), cfg['batch_size']):
        batch = core.minibatch(data, data['evaluation'][start:start+cfg['batch_size']])
        _, info = model(batch, diagnostics=True)
        masks = dict(dynamic=batch['dynamic'][:, :, 6] > .5, state=batch['state_valid'].bool())
        for branch, tail, other in (('dynamic', 90, 'state'), ('state', 55, 'dynamic')):
            delta = info[f'{branch}_delta']
            if delta.ndim == 3: delta = masked_mean(delta, masks[branch], tail)
            query_valid = info[f'{branch}_query_valid']
            support = query_valid & masks[other].any(1)
            base = info[f'{branch}_endpoint']
            dn, bn = delta.norm(dim=-1), base.norm(dim=-1)
            valid_ratio = support & (bn > 1e-8)
            row = accum.setdefault(branch, dict(targets=0, queries_valid=0, supported=0,
                ratio_targets=0, delta_norm_sum=0., base_norm_sum=0., ratio_sum=0.))
            row['targets'] += len(delta); row['queries_valid'] += int(query_valid.sum())
            row['supported'] += int(support.sum()); row['ratio_targets'] += int(valid_ratio.sum())
            row['delta_norm_sum'] += float(dn[support].sum())
            row['base_norm_sum'] += float(bn[support].sum())
            row['ratio_sum'] += float((dn[valid_ratio] / bn[valid_ratio]).sum())
    return accum


def require_concat_replays(cfg):
    for p in cfg['protocols']:
        done = read(Path(cfg['output']) / 'groups' / p / 'TwoRate_Concat' / 'COMPLETE.json')
        assert done['status'] == 'PASS' and done['baseline_replay_passed'] is True


def worker(cfg, protocol, arm):
    check_seal(cfg)
    assert protocol in cfg['protocols'] and arm in cfg['arms']
    if arm != 'TwoRate_Concat': require_concat_replays(cfg)
    lock = core.acquire_gpu(); directory = Path(cfg['output']) / 'groups' / protocol / arm
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / 'STARTED.json').exists(), 'No retries'
    save(directory / 'STARTED.json', dict(pid=os.getpid(), started_at_unix=time.time()))
    data = core.data_for(cfg, protocol); history = {}; final = {}; started = time.perf_counter()
    reference = Path(cfg['reference_output']) / 'groups' / protocol / 'TwoRate_Concat'
    old_history = read(reference / 'HISTORY.json') if arm == 'TwoRate_Concat' else None
    assert (Path(cfg['output']) / 'data' / protocol / 'targets.jsonl').read_bytes() == (
        Path(cfg['reference_output']) / 'data' / protocol / 'targets.jsonl').read_bytes()
    for seed in cfg['seeds']:
        core.seed_all(seed); model = model_for(arm, cfg, data['prior']).cuda()
        opt = optimizer_for(model, cfg)
        ema = core.ParameterEMA(model, cfg['ema_decay'], cfg['ema_warmup_steps'])
        history[str(seed)] = []
        for epoch in range(1, 21):
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(data['train'])
            numerator = denominator = 0.; tick = time.perf_counter()
            for i in range(0, len(order), cfg['batch_size']):
                loss, weight = core.update(model, opt, ema, data, order[i:i+cfg['batch_size']])
                numerator += loss * weight; denominator += weight
            record = dict(epoch=epoch, train_ce=numerator/denominator, validation={})
            for view, network in (('raw', model), ('ema', ema.model)):
                pred, score = core.evaluate(network, data, arm, data['evaluation'], cfg['batch_size'])
                record['validation'][view] = score
                if epoch == 20: np.save(directory / f'seed{seed}_{view}_final20_predictions.npy', pred)
            if old_history is not None:
                prior = old_history[str(seed)][epoch-1]
                assert record['train_ce'] == prior['train_ce']
                assert record['validation'] == prior['validation']
            elif epoch in cfg['diagnostic_epochs']:
                record['diagnostics'] = {v:diagnostics(m, data, cfg) for v,m in (('raw', model), ('ema', ema.model))}
            record['elapsed_seconds'] = time.perf_counter()-tick
            history[str(seed)].append(record); save(directory / 'HISTORY.json', history)
            temp = directory / f'seed{seed}_latest.tmp'
            torch.save(dict(epoch=epoch, seed=seed, arm=arm, protocol=protocol, model=model.state_dict(),
                ema=ema.model.state_dict(), optimizer=opt.state_dict(), ema_updates=ema.updates, rng=core.rng()), temp)
            temp.replace(directory / f'seed{seed}_latest.pt')
            print(json.dumps(dict(protocol=protocol, arm=arm, seed=seed, epoch=epoch,
                raw_f1=record['validation']['raw']['macro_f1'], ema_f1=record['validation']['ema']['macro_f1'],
                train_ce=record['train_ce'], seconds=record['elapsed_seconds'])), flush=True)
        payload = torch.load(directory / f'seed{seed}_latest.pt', weights_only=False)
        for view, key in (('raw', 'model'), ('ema', 'ema')):
            other = model_for(arm, cfg, data['prior']).cuda(); other.load_state_dict(payload[key])
            pred, score = core.evaluate(other, data, arm, data['evaluation'], cfg['batch_size'])
            np.testing.assert_array_equal(pred, np.load(directory / f'seed{seed}_{view}_final20_predictions.npy'))
            assert score == history[str(seed)][-1]['validation'][view]
            if arm == 'TwoRate_Concat':
                np.testing.assert_array_equal(pred, np.load(reference / f'seed{seed}_{view}_final20_predictions.npy'))
            del other
        final[str(seed)] = history[str(seed)][-1]['validation']
        del model, opt, ema, payload
    save(directory / 'COMPLETE.json', dict(status='PASS', final=final, checkpoint_prediction_replays=6,
        baseline_replay_passed=arm == 'TwoRate_Concat', elapsed_seconds=time.perf_counter()-started,
        parameters=sum(p.numel() for p in model_for(arm, cfg, data['prior']).parameters()), Test_allowed=False))
    lock.close()


def supervise(cfg):
    check_seal(cfg); out = Path(cfg['output']); lock = (out / 'QUEUE.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'RUN_START.json').exists()
    started = time.time()
    jobs = [dict(protocol=p, arm=a, status='pending', attempts=0) for a in cfg['arms'] for p in cfg['protocols']]
    save(out / 'RUN_START.json', dict(pid=os.getpid(), started_at_unix=started))
    for i, job in enumerate(jobs):
        remaining = cfg['max_wall_seconds']-(time.time()-started)
        if remaining <= 0: raise RuntimeError('Fixed wall budget exhausted')
        job.update(status='running', attempts=1)
        save(out / 'queue_state.json', dict(jobs=jobs, max_attempts=1))
        save(out / 'PUBLIC_PROGRESS.json', dict(status='RUNNING', completed_groups=i, total_groups=6, current=job))
        log = out / 'logs' / f'{job["protocol"]}_{job["arm"]}.log'; log.parent.mkdir(exist_ok=True)
        try:
            with log.open('wb') as stream:
                result = subprocess.run([sys.executable, '-s', '-m', MODULE, 'worker', '--protocol', job['protocol'],
                    '--arm', job['arm']], stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
            if result.returncode: raise RuntimeError(f'Worker failed: {job}; see {log}')
            assert read(out / 'groups' / job['protocol'] / job['arm'] / 'COMPLETE.json')['status'] == 'PASS'
        except BaseException:
            job['status'] = 'failed'; save(out / 'queue_state.json', dict(jobs=jobs, max_attempts=1)); raise
        job['status'] = 'completed'; save(out / 'queue_state.json', dict(jobs=jobs, max_attempts=1))
    from scripts.report_convtran_endpoint_20261022 import report
    report(cfg)
    save(out / 'PUBLIC_PROGRESS.json', dict(status='COMPLETE_STOP_FOR_REVIEW', completed_groups=6,
        total_groups=6, Test_allowed=False, automatic_expansion=False))


def launch(cfg):
    check_seal(cfg); out = Path(cfg['output']); assert not (out / 'LAUNCH.json').exists()
    with (out / 'SUPERVISOR.log').open('wb') as stream:
        p = subprocess.Popen([sys.executable, '-s', '-m', MODULE, 'supervise'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=p.pid, launched_at_unix=time.time(), Test_allowed=False, automatic_expansion=False)
    save(out / 'LAUNCH.json', receipt); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'seal', 'worker', 'supervise', 'launch'])
    parser.add_argument('--protocol', choices=['TS', 'OS']); parser.add_argument('--arm')
    args = parser.parse_args(); cfg = configure()
    try:
        if args.phase == 'worker': worker(cfg, args.protocol, args.arm)
        else: globals()[args.phase](cfg)
    except BaseException:
        save(Path(cfg['output']) / f'FAILURE_{time.time_ns()}.json', dict(phase=args.phase,
            traceback=traceback.format_exc(), automatic_retry=False))
        save(Path(cfg['output']) / 'PUBLIC_PROGRESS.json', dict(status='STOPPED_FAILURE', Test_allowed=False))
        raise
