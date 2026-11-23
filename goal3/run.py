"""Run one pinned goal-3 experiment. Defaults to a bounded smoke test."""
from importlib.metadata import distributions
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from recording import RunRecorder, atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / '.goal3-sources'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algo', choices=['MICE', 'IP3O', 'CSPO'], required=True)
    p.add_argument('--env', choices=['SafetyPointGoal1-v0', 'SafetyCarGoal1-v0', 'SafetyPointCircle1-v0', 'SafetyCarCircle1-v0'], default='SafetyPointGoal1-v0')
    p.add_argument('--mode', choices=['smoke', 'pilot', 'full'], default='smoke')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--vector-envs', type=int, default=1)
    p.add_argument('--total-steps', type=int)
    p.add_argument('--epochs', type=int, help='Explicit epoch count; cannot be combined with total-steps')
    p.add_argument('--output-dir', help='New directory for this individual run')
    p.add_argument('--save-every', type=int, default=1, help='Checkpoint interval in epochs')
    p.add_argument('--cost-limit', type=float, default=25.0)
    p.add_argument('--ip3o-alpha', type=float, default=0.1)
    p.add_argument('--cspo-alpha', type=float, default=0.3)
    p.add_argument('--check', action='store_true', help='Initialize and verify without training')
    args = p.parse_args()
    if args.epochs is not None and (args.epochs < 1 or args.total_steps is not None):
        p.error('epochs must be positive and cannot be combined with total-steps.')
    if args.save_every < 1:
        p.error('save-every must be positive.')
    if args.threads < 1 or args.vector_envs < 1 or args.cost_limit < 0 or args.ip3o_alpha <= 0:
        p.error('Threads/vector-envs must be positive; cost-limit nonnegative; IP3O alpha positive.')
    runtime = ROOT / ('work/runtimes/ip3o' if args.algo == 'IP3O' else args.algo)
    if not (runtime / 'omnisafe').exists():
        p.error('Run prepare.py first.')
    sys.path.insert(0, str(runtime))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    import torch
    import omnisafe

    steps_per_epoch = 2000 if args.mode == 'smoke' else (5000 if args.algo == 'IP3O' else 20000)
    total_steps = args.total_steps if args.total_steps is not None else {'smoke': 4000, 'pilot': 1000000, 'full': 10000000}[args.mode]
    if args.epochs is not None:
        total_steps = args.epochs * steps_per_epoch
    if total_steps <= 0 or total_steps % steps_per_epoch:
        p.error(f'total-steps must be a positive multiple of {steps_per_epoch}.')
    if steps_per_epoch % args.vector_envs or (steps_per_epoch // args.vector_envs) % 1000:
        p.error('Each vector environment needs a multiple of 1000 steps per epoch, to avoid discarded partial episodes.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable here. Use cpu, or run on a CUDA machine.')
    if args.device != 'cpu' and not args.device.startswith('cuda'):
        p.error('This OmniSafe version supports cpu or cuda; MPS is not supported.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = Path(args.output_dir).resolve() if args.output_dir else HERE / 'runs' / args.mode / f'{args.algo}_{args.env}_seed{args.seed}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    cfg = {
        'seed': args.seed,
        'train_cfgs': {'device': args.device, 'torch_threads': args.threads,
                       'parallel': 1, 'vector_env_nums': args.vector_envs, 'total_steps': total_steps},
        'algo_cfgs': {'steps_per_epoch': steps_per_epoch, 'cost_limit': args.cost_limit},
        'logger_cfgs': {'log_dir': str(out / 'training'), 'use_wandb': False,
                        'use_tensorboard': True, 'save_model_freq': args.save_every},
    }
    if args.mode == 'smoke':
        cfg['algo_cfgs'].update(update_iters=1, batch_size=128)
    if args.algo == 'IP3O':
        cfg['algo_cfgs']['alpha'] = args.ip3o_alpha
    elif args.algo == 'CSPO':
        cfg['algo_cfgs']['alpha'] = args.cspo_alpha
        cfg['lagrange_cfgs'] = {'cost_limit': args.cost_limit}
    else:
        # The extra bias diagnostic is not needed for goal 3 and consumes extra rollouts.
        cfg['algo_cfgs']['test_estimate'] = False
    manifest = {'arguments': vars(args), 'overrides': cfg,
                'sources': json.loads((HERE / 'sources.json').read_text()),
                'python': sys.version, 'omnisafe_path': omnisafe.__file__,
                'started_utc': stamp, 'status': 'initializing'}
    for name in ['MICE', 'IP3O', 'CSPO']:
        actual = subprocess.check_output(
            ['git', '-C', str(ROOT / name), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != manifest['sources'][name]['commit']:
            raise RuntimeError(f'{name}: source revision changed. Update the source lock deliberately.')
        manifest['sources'][name]['working_tree_status'] = subprocess.check_output(
            ['git', '-C', str(ROOT / name), 'status', '--porcelain'], text=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'packages.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in distributions())) + '\n')
    start = time.monotonic()
    recorder = None
    try:
        agent = omnisafe.Agent(args.algo, args.env, custom_cfgs=cfg)
        resolved = agent.cfgs.tojson()
        (out / 'resolved_config.json').write_text(resolved)
        manifest['config_sha256'] = hashlib.sha256(resolved.encode()).hexdigest()
        if args.check:
            manifest['status'] = 'initialized'
            print(f'Initialization passed: {out}')
            return
        manifest['status'] = 'running'
        manifest['planned_epochs'] = total_steps // steps_per_epoch
        manifest['completed_epochs'] = 0
        atomic_json(out / 'manifest.json', manifest)
        recorder = RunRecorder(out, args.cost_limit)
        stats = recorder.stats
        adapter = agent.agent._env
        old_log_value = adapter._log_value

        def log_value(reward, cost, info):
            raw_cost = info.get('original_cost', cost).detach().cpu()
            recorder.record_step(raw_cost.reshape(-1).tolist())
            return old_log_value(reward=reward, cost=cost, info=info)

        adapter._log_value = log_value
        logger = agent.agent.logger
        old_store, old_dump = logger.store, logger.dump_tabular

        epoch_times = {}

        def store(data=None, /, **kwargs):
            values = dict(kwargs)
            if data is not None:
                values.update(data)
            for key in ('Time/Epoch', 'Time/Rollout', 'Time/Update'):
                if key in values:
                    epoch_times[key] = float(values[key])
            if 'Metrics/EpCost' in values:
                recorder.record_episode(values['Metrics/EpRet'], values['Metrics/EpCost'], values['Metrics/EpLen'])
            return old_store(data, **kwargs)

        def dump():
            old_dump()
            recorder.end_epoch(logger.current_epoch, epoch_times)
            epoch_times.clear()
            manifest['completed_epochs'] = logger.current_epoch
            manifest['environment_steps'] = stats['environment_steps']
            atomic_json(out / 'manifest.json', manifest)

        logger.store, logger.dump_tabular = store, dump
        result = agent.learn()
        if stats['environment_steps'] != total_steps:
            raise RuntimeError(f'Collected {stats["environment_steps"]}, expected {total_steps} steps.')
        if not all(math.isfinite(float(x)) for x in result):
            raise RuntimeError('Training returned non-finite results.')
        stats.update(wall_seconds=time.monotonic() - start,
                     final_logged_return=float(result[0]), final_logged_cost=float(result[1]),
                     note='Final logged values are rolling training statistics, not independent evaluation.')
        (out / 'summary.json').write_text(json.dumps(stats, indent=2))
        manifest['status'] = 'completed'
        print(f'Completed: {out}')
    except BaseException as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        (out / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if recorder is not None:
            recorder.close()
        manifest['wall_seconds'] = time.monotonic() - start
        atomic_json(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
