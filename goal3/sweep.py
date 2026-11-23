"""Launch or list the Goal 3 comparison, one algorithm at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'experiment_defaults.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('algorithm', choices=['MICE', 'IP3O', 'CSPO'])
parser.add_argument('--env', choices=plan['envs'])
parser.add_argument('--seed', type=int, choices=plan['seeds'])
parser.add_argument('--output', type=Path, default=HERE.parent / 'goal3-results')
parser.add_argument('--list', action='store_true')
args = parser.parse_args()
for env in ([args.env] if args.env else plan['envs']):
    for seed in ([args.seed] if args.seed is not None else plan['seeds']):
        folder = args.output / args.algorithm / env / f'seed-{seed}'
        steps = plan['total_steps']
        print(args.algorithm, env, seed, steps, folder, flush=True)
        if args.list:
            continue
        manifest = folder / 'data' / 'manifest.json'
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if record['status'] == 'completed' and record['environment_steps'] == steps:
                continue
        if folder.exists():
            sys.exit(f'Existing incomplete run in {folder}. If inactive, move it outside the results tree before retrying.')
        folder.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', str(HERE / 'run.py'), '--algo', args.algorithm,
                   '--env', env, '--seed', str(seed), '--mode', 'full',
                   '--total-steps', str(steps), '--save-every', '32',
                   '--output-dir', str(folder / 'data')]
        with (folder / 'console.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
