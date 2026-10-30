"""Continue original raw20/Adam/EMA/RNG to fixed training-budget endpoints."""
import argparse
import copy
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from scripts import run_zero_head_replication_20261024 as base
from scripts.sensor_learned_fusion_data_20261023 import read, save, sha

core = base.core
CONFIG = Path('configs/sensor_b_epoch_extension_20261024_v1.json')
MODULE = 'scripts.run_b_epoch_extension_20261024'


def configure(cuda=False):
    assert Path.cwd() == core.ROOT
    assert (torch.__version__, np.__version__) == ('2.4.1+cu121', '1.24.4')
    if cuda:
        core.configure()
    else:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
    cfg = read(CONFIG)
    assert cfg['protocols'] == ['TS', 'OS'] and cfg['inners'] == [0, 1, 2] and cfg['seeds'] == [17, 23, 29]
    assert cfg['endpoints'] == [20, 30, 40, 60, 80, 100] and cfg['start_epoch'] == 20 and cfg['max_epoch'] == 100
    assert cfg['learning_rate'] == .001 and cfg['scheduler'].startswith('none') and cfg['trajectories'] == 18
    assert all(cfg[n] is False for n in ['Test_allowed', 'outer_VAL_allowed', 'formal_refit_allowed', 'automatic_expansion', 'early_stopping', 'learning_rate_search'])
    return cfg


def implementation(cfg):
    files = dict(read(Path(cfg['parent']) / 'SEAL.json')['code'])
    for p, expected in files.items():
        assert sha(p) == expected, p
    paths = [CONFIG, Path(cfg['plan']), Path('.aris/compute/bitahub-root-b-epochs-20261024.md')]
    paths += list(Path('scripts').glob('*b_epoch_extension*20261024.py'))
    files.update({str(p): sha(p) for p in paths})
    return files


def data_for(cfg, p, k):
    old = read(Path(cfg['parent']) / 'SEAL.json')['config']
    return base.load_data(old, p, k)


def bundle(cfg, data, payload, device='cuda'):
    model = base.native(cfg, data['prior'], zero=False, device=device)
    model.load_state_dict(payload['model'])
    bcfg = read(cfg['B_config'])
    opt = base.optimizer_for(model, bcfg)
    # Adam's CPU step tensors may otherwise alias the supplied payload even
    # when parameter moments move to CUDA; repeated loads must be independent.
    opt.load_state_dict(copy.deepcopy(payload['optimizer']))
    ema = base.ParameterEMA(model, bcfg['ema_decay'], bcfg['ema_warmup_steps'])
    ema.model.load_state_dict(payload['ema'])
    ema.updates = payload['ema_updates']
    core.restore_rng(payload['rng'])
    return model, opt, ema


def validate_payload(cfg, data, payload, p, k, seed):
    assert (payload['protocol'], payload['inner'], payload['seed'], payload['epoch']) == (p, k, seed, 20)
    expected = 20 * math.ceil(len(data['train']) / 256)
    assert payload['ema_updates'] == expected
    assert all(float(v['step']) == expected for v in payload['optimizer']['state'].values())
    assert len(payload['optimizer']['state']) > 0
    for group in payload['optimizer']['param_groups']:
        assert (group['lr'], group['weight_decay'], group['eps'], group['betas']) == (.001, .01, 1e-8, (.9, .999))
    assert set(payload['rng']) == {'python', 'numpy', 'torch', 'cuda'}


def export_epoch(cfg, data, model, opt, ema, folder, epoch, metadata):
    """Prediction/reload must not perturb the future optimization random stream."""
    folder = Path(folder)
    assert not (folder / 'COMPLETE.json').exists()
    before = core.rng()
    base.checkpoint(folder / 'model.pt', model, opt, ema, epoch=epoch, **metadata)
    payload = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    paths = {}
    try:
        for view, net, key in [('ema', ema.model, 'ema'), ('raw', model, 'model')]:
            prob = base.infer(net, data, data['evaluation'])
            other = base.native(cfg, data['prior'], zero=False, device=data['tensors']['dynamic'].device)
            other.eval().requires_grad_(view == 'raw')
            other.load_state_dict(payload[key])
            np.testing.assert_array_equal(prob, base.infer(other, data, data['evaluation']))
            path = folder / f'held_{view}.npy'
            np.save(path, prob)
            paths[str(path)] = sha(path)
            del other
    finally:
        core.restore_rng(before)
    base.equal_tree(before, core.rng())
    paths[str(folder / 'model.pt')] = sha(folder / 'model.pt')
    record = dict(status='COMPLETE', epoch=epoch, files=paths, rng_preserved=True, serialized_raw_EMA_exact=True,
                  optimizer_steps=[float(v['step']) for v in opt.state.values()], EMA_updates=ema.updates, **metadata)
    save(folder / 'COMPLETE.json', record)
    return record


def equal_bundles(a, b):
    for x, y in [(a[0].state_dict(), b[0].state_dict()), (a[1].state_dict(), b[1].state_dict()), (a[2].model.state_dict(), b[2].model.state_dict())]:
        base.equal_tree(x, y)
    assert a[2].updates == b[2].updates


def restore_check(cfg, data, payload, folder, device):
    ids = data['train'][:67]
    original_optimizer = copy.deepcopy(payload['optimizer'])
    a = bundle(cfg, data, payload, device)
    base.update(*a, data, ids)
    base.equal_tree(payload['optimizer'], original_optimizer)
    base.checkpoint(Path(folder) / 'partial.pt', *a, epoch=20, diagnostic_only=True)
    partial = torch.load(Path(folder) / 'partial.pt', map_location='cpu', weights_only=False)
    export_epoch(cfg, data, *a, Path(folder) / 'inference_probe', 21, dict(scope='restore diagnostic only'))
    expected_loss = base.update(*a, data, ids)
    expected_rng = core.rng()
    b = bundle(cfg, data, partial, device)
    actual_loss = base.update(*b, data, ids)
    base.equal_tree(expected_loss, actual_loss)
    base.equal_tree(expected_rng, core.rng())
    equal_bundles(a, b)
    for x, y in [(a[0], b[0]), (a[2].model, b[2].model)]:
        base.equal_tree(base.infer(x, data, ids), base.infer(y, data, ids))
    return dict(source_optimizer_payload_unchanged=True, model_optimizer_EMA_RNG_next_update_exact=True, endpoint_inference_does_not_change_next_update=True,
                serialized_raw_EMA_tail67_exact=True)


def witness(cfg, output, cpu=False):
    device = 'cpu' if cpu else 'cuda'
    lock = None if cpu else core.acquire_gpu()
    core.seed_all(17)
    n = 67
    dynamic = torch.randn(n, 100, 8, device=device)
    dynamic[:, :, 6], dynamic[:, :, 7] = 1., .1
    state = torch.randn(n, 60, 26, device=device)
    data = dict(prior=np.full(7, 1/7), train=np.arange(n), evaluation=np.arange(n), weights=torch.ones(7, device=device),
                tensors=dict(dynamic=dynamic, state=state, state_valid=torch.ones(n, 60, device=device, dtype=torch.bool),
                    context=torch.randn(n, 136, device=device), current=state[:, -1], labels=torch.arange(n, device=device) % 7))
    model = base.native(cfg, data['prior'], zero=False, device=device)
    opt = base.optimizer_for(model, read(cfg['B_config']))
    ema = base.ParameterEMA(model, .998, 200)
    for _ in range(20):
        base.update(model, opt, ema, data, data['train'])
    work = Path(output).parent / (Path(output).stem + '_work_' + str(time.time_ns()))
    base.checkpoint(work / 'source20.pt', model, opt, ema, epoch=20, protocol='TS', inner=0, seed=17)
    payload = torch.load(work / 'source20.pt', map_location='cpu', weights_only=False)
    validate_payload(cfg, data, payload, 'TS', 0, 17)
    record = restore_check(cfg, data, payload, work / 'restore', device)
    save(output, dict(status='PASS', parameters=213119, CUDA_validated=not cpu, real_data_rows_read=0,
                      synthetic_prefix_updates=20, source_full_state_verified=True, restore=record, implementation_files=implementation(cfg)))
    if lock:
        lock.close()
    print('B_EPOCH_EXTENSION_WITNESS_PASS', flush=True)


def prepare(cfg):
    out, parent = Path(cfg['output']), Path(cfg['parent'])
    assert not (out / 'PREPARE_STARTED.json').exists()
    save(out / 'PREPARE_STARTED.json', dict(pid=os.getpid(), unix=time.time()))
    old = read(parent / 'SEAL.json')
    files = {str(parent / 'SEAL.json'): sha(parent / 'SEAL.json')}
    for p, expected in {**old['code'], **old['files']}.items():
        assert sha(p) == expected, p
        files[p] = expected
    records = []
    for p in cfg['protocols']:
        for k in cfg['inners']:
            d = parent / 'data' / p / f'inner{k}'
            receipt = read(d / 'RECEIPT.json')
            for seed in cfg['seeds']:
                expert = Path(cfg['parent_experts']) / p / f'inner{k}'
                source, history = expert / f'B_seed{seed}_latest.pt', expert / f'B_seed{seed}_HISTORY.json'
                payload = torch.load(source, map_location='cpu', weights_only=False)
                validate_payload(cfg, dict(train=np.arange(receipt['fit_rows'])), payload, p, k, seed)
                h = read(history)
                assert [r['epoch'] for r in h] == list(range(1, 21)) and all(np.isfinite(r['train_ce']) for r in h)
                files[str(source)], files[str(history)] = sha(source), sha(history)
                records.append(dict(protocol=p, inner=k, seed=seed, source=str(source), source_sha256=sha(source),
                                    fit_rows=receipt['fit_rows'], original_updates=payload['ema_updates'], full_state=True))
    save(out / 'PREPARED.json', dict(status='PASS', files=files, sources=records, new_scientific_trajectories=0))


def original_index_data(cfg, p, k, packed):
    directory = Path(cfg['parent_cache']) / p / f'inner{k}'
    with np.load(directory / 'DATA.npz', allow_pickle=False) as z:
        tensors = {n: torch.as_tensor(z[n], device='cuda') for n in base.FIELDS}
        fit = z['train'].copy()
        labels = np.full(len(z['labels']), -100, dtype=np.int64)
        labels[fit] = z['labels'][fit]
    tensors['labels'] = torch.as_tensor(labels, device='cuda')
    return dict(train=fit, tensors=tensors, weights=packed['weights'], prior=packed['prior'])


def preflight(cfg):
    out = Path(cfg['output'])
    if (out / 'PREFLIGHT_STARTED.json').exists():
        assert (out / 'PREFLIGHT_OPTIMIZER_ALIAS_REPAIR.json').exists() and not (out / 'PREFLIGHT_REPAIR_STARTED.json').exists()
        assert not (out / 'SEAL.json').exists() and not (out / 'PREFLIGHT.json').exists()
        save(out / 'PREFLIGHT_REPAIR_STARTED.json', dict(pid=os.getpid(), unix=time.time(), reason='Independent optimizer state copies; initial preflight evidence retained'))
    else:
        save(out / 'PREFLIGHT_STARTED.json', dict(pid=os.getpid(), unix=time.time()))
    lock = core.acquire_gpu()
    records = []
    for p in cfg['protocols']:
        for k in cfg['inners']:
            data = data_for(cfg, p, k)
            original = original_index_data(cfg, p, k, data)
            order = np.random.default_rng(np.random.SeedSequence([17, 21])).permutation(data['train'])
            original_order = np.random.default_rng(np.random.SeedSequence([17, 21])).permutation(original['train'])
            for ix in [slice(0, 256), slice(-67, None)]:
                base.equal_tree(base.minibatch(data, order[ix]), base.minibatch(original, original_order[ix]))
            seed_records = []
            for seed in cfg['seeds']:
                source = Path(cfg['parent_experts']) / p / f'inner{k}' / f'B_seed{seed}_latest.pt'
                payload = torch.load(source, map_location='cpu', weights_only=False)
                validate_payload(cfg, data, payload, p, k, seed)
                state = bundle(cfg, data, payload)
                prob = base.infer(state[2].model, data, data['evaluation'])
                np.testing.assert_array_equal(prob, np.load(Path(cfg['parent']) / 'baseline' / p / f'inner{k}' / f'B_{seed}.npy'))
                seed_records.append(dict(seed=seed, source_state_and_EMA20_exact=True))
                del state
            source = Path(cfg['parent_experts']) / p / f'inner{k}' / 'B_seed17_latest.pt'
            payload = torch.load(source, map_location='cpu', weights_only=False)
            a = bundle(cfg, data, payload)
            expected = base.update(*a, original, original_order[:256])
            expected_rng = core.rng()
            b = bundle(cfg, data, payload)
            actual = base.update(*b, data, order[:256])
            base.equal_tree(expected, actual)
            base.equal_tree(expected_rng, core.rng())
            equal_bundles(a, b)
            restore = restore_check(cfg, data, payload, out / 'preflight' / f'{p}_{k}', 'cuda')
            state = bundle(cfg, data, payload)
            times = []
            for step in range(3):
                tick = time.monotonic()
                base.update(*state, data, order[step*256:(step+1)*256])
                torch.cuda.synchronize()
                times.append(time.monotonic() - tick)
            estimate = (max(times[1:]) * math.ceil(len(data['train']) / 256) * 80 + 90) * 1.4
            assert estimate < cfg['budget']['job_seconds']
            records.append(dict(protocol=p, inner=k, source_replays=seed_records, packed_original_next_update_exact=True,
                                restore=restore, step_seconds=times, estimated_job_seconds=estimate))
            del data, original, a, b, state, payload
            gc.collect()
            torch.cuda.empty_cache()
    lock.close()
    estimate = sum(r['estimated_job_seconds'] * 3 for r in records) + 240
    assert estimate < cfg['budget']['total_seconds']
    save(out / 'PREFLIGHT.json', dict(status='PASS', records=records, estimated_total_seconds=estimate,
                                    implementation_files=implementation(cfg), candidate_held_scores=0))


def seal(cfg):
    out = Path(cfg['output'])
    assert not (out / 'SEAL.json').exists()
    code, evidence = implementation(cfg), {}
    for n in ['INDEPENDENT_RUNTIME_WITNESS.json', 'RUNTIME_WITNESS.json', 'PREFLIGHT.json']:
        record = read(out / n)
        assert record['status'] == 'PASS' and record['implementation_files'] == code, n
        evidence[n] = sha(out / n)
    prepared = read(out / 'PREPARED.json')
    for p, expected in prepared['files'].items():
        assert sha(p) == expected, p
    evidence['PREPARED.json'] = sha(out / 'PREPARED.json')
    save(out / 'SEAL.json', dict(config=cfg, code=code, files=prepared['files'], evidence=evidence, unix=time.time(), new_endpoint_scores=0))


def check(cfg, full=False):
    out = Path(cfg['output'])
    r = read(out / 'SEAL.json')
    assert r['config'] == cfg and r['code'] == implementation(cfg)
    for n, digest in r['evidence'].items():
        assert sha(out / n) == digest, n
    if full:
        for p, digest in r['files'].items():
            assert sha(p) == digest, p


def worker(cfg, p, k, seed):
    check(cfg)
    lock = core.acquire_gpu()
    folder = Path(cfg['output']) / 'trajectories' / p / f'inner{k}' / f'seed{seed}'
    assert not (folder / 'STARTED.json').exists()
    started = time.monotonic()
    save(folder / 'STARTED.json', dict(pid=os.getpid(), unix=time.time(), start_epoch=20, target_epoch=100))
    data = data_for(cfg, p, k)
    expert = Path(cfg['parent_experts']) / p / f'inner{k}'
    source = expert / f'B_seed{seed}_latest.pt'
    payload = torch.load(source, map_location='cpu', weights_only=False)
    validate_payload(cfg, data, payload, p, k, seed)
    state = bundle(cfg, data, payload)
    metadata = dict(protocol=p, inner=k, seed=seed, source_sha256=sha(source))
    first = export_epoch(cfg, data, *state, folder / 'epoch20', 20, metadata)
    np.testing.assert_array_equal(np.load(folder / 'epoch20' / 'held_ema.npy'), np.load(Path(cfg['parent']) / 'baseline' / p / f'inner{k}' / f'B_{seed}.npy'))
    history = [dict(epoch=r['epoch'], train_ce=r['train_ce'], seconds=r['seconds'], learning_rate=.001, reused_prefix=True)
               for r in read(expert / f'B_seed{seed}_HISTORY.json')]
    for epoch in range(21, 101):
        tick = time.monotonic()
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(data['train'])
        numerator = denominator = 0.
        for at in range(0, len(order), 256):
            loss, mass = base.update(*state, data, order[at:at+256])
            numerator += loss * mass
            denominator += mass
        row = dict(epoch=epoch, train_ce=numerator / denominator, seconds=time.monotonic() - tick, learning_rate=state[1].param_groups[0]['lr'], reused_prefix=False)
        assert np.isfinite(row['train_ce']) and row['learning_rate'] == .001
        history.append(row)
        save(folder / 'history.json', history)
        base.checkpoint(folder / 'latest.pt', *state, epoch=epoch, **metadata)
        if epoch in cfg['endpoints']:
            export_epoch(cfg, data, *state, folder / f'epoch{epoch}', epoch, metadata)
        save(folder / 'PROGRESS.json', dict(**metadata, epoch=epoch, target_epoch=100, elapsed_seconds=time.monotonic()-started, updated_unix=time.time()))
        print(row, flush=True)
    check(cfg)
    save(folder / 'COMPLETE.json', dict(status='COMPLETE', **metadata, final_epoch=100, new_epochs=80,
         source_unchanged=sha(source)==metadata['source_sha256'], history_sha256=sha(folder/'history.json'), latest_sha256=sha(folder/'latest.pt'),
         endpoints={str(e): sha(folder/f'epoch{e}'/'COMPLETE.json') for e in cfg['endpoints']}, elapsed_seconds=time.monotonic()-started,
         candidate_held_scores=0, optimizer_EMA_RNG_not_reset=True))
    lock.close()


def queue(cfg):
    import signal
    out = Path(cfg['output'])
    check(cfg, True)
    assert not (out / 'LAUNCH.json').exists()
    started, active = time.monotonic(), None
    jobs = [dict(protocol=p, inner=k, seed=s, status='pending') for s in cfg['seeds'] for k in cfg['inners'] for p in cfg['protocols']]
    save(out / 'LAUNCH.json', dict(pid=os.getpid(), started_unix=time.time(), trajectories=18, additional_epochs=1440, endpoints=cfg['endpoints'], budget=cfg['budget']))

    def snapshot(phase):
        save(out/'PUBLIC_PROGRESS.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),elapsed_seconds=time.monotonic()-started,
             completed=sum(j['status']=='completed' for j in jobs),total=18,jobs=jobs))

    def stop():
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()

    try:
        for j in jobs:
            if time.monotonic()-started>cfg['budget']['total_seconds']:
                raise RuntimeError('Cumulative scientific budget exhausted')
            p,k,s=j['protocol'],j['inner'],j['seed']
            log=out/'logs'/f'{p}_{k}_{s}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('wb') as handle:
                active=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--protocol',p,'--inner',str(k),'--seed',str(s)],
                                        stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',pid=active.pid,started_unix=time.time());tick=time.monotonic()
                while active.poll() is None:
                    snapshot('EXTENDING_ORIGINAL_B')
                    if time.monotonic()-tick>cfg['budget']['job_seconds'] or time.monotonic()-started>cfg['budget']['total_seconds']:
                        stop();raise RuntimeError('Scientific budget exhausted; no automatic restart')
                    time.sleep(3)
            done=out/'trajectories'/p/f'inner{k}'/f'seed{s}'/'COMPLETE.json'
            if active.returncode!=0 or not done.exists():
                j['status']='failed';raise RuntimeError('Original continuation failed; no automatic retry')
            assert read(done)['status']=='COMPLETE';j.update(status='completed',finished_unix=time.time());snapshot('EXTENDING_ORIGINAL_B')
        check(cfg,True);snapshot('FREEZING_AND_SCORING')
        from scripts.report_b_epoch_extension_20261024 import report
        save(out/'COMPLETE.json',report(cfg));snapshot('COMPLETE_STOP_FOR_REVIEW')
    except BaseException:
        stop()
        for j in jobs:
            if j['status'] in ['pending','running']:j['status']='blocked'
        snapshot('FAILED_STOP_FOR_INSPECTION')
        save(out/'QUEUE_FAILURE.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['cpu_witness','witness','prepare','preflight','seal','launch','queue','worker'])
    p.add_argument('--witness-output');p.add_argument('--protocol',choices=['TS','OS']);p.add_argument('--inner',type=int,choices=[0,1,2]);p.add_argument('--seed',type=int,choices=[17,23,29])
    a=p.parse_args();cfg=configure(a.mode in ['witness','preflight','worker']);out=Path(cfg['output'])
    if a.mode in ['cpu_witness','witness']:witness(cfg,a.witness_output or out/'RUNTIME_WITNESS.json',a.mode=='cpu_witness')
    elif a.mode=='prepare':prepare(cfg)
    elif a.mode=='preflight':preflight(cfg)
    elif a.mode=='seal':seal(cfg)
    elif a.mode=='worker':
        assert a.protocol in cfg['protocols'] and a.inner in cfg['inners'] and a.seed in cfg['seeds'];worker(cfg,a.protocol,a.inner,a.seed)
    elif a.mode=='queue':queue(cfg)
    else:
        check(cfg,True);assert not any((out/n).exists() for n in ['DISPATCH.json','LAUNCH.json'])
        with (out/'queue.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','queue'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'DISPATCH.json',dict(pid=child.pid,unix=time.time()));print(child.pid)


if __name__=='__main__':main()
