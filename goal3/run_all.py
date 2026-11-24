"""Run the full Goal 3 plan, preserving each training attempt and its logs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = json.loads((HERE / 'experiment_defaults.json').read_text())
METHODS = ('MICE', 'IP3O', 'CSPO')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def complete(folder):
    manifest = folder / 'data' / 'manifest.json'
    if not manifest.is_file():
        return False
    info = json.loads(manifest.read_text())
    return info.get('status') == 'completed' and info.get('environment_steps') == PLAN['total_steps']


def run_batch(out):
    jobs = [(algo, env, seed) for algo in METHODS for env in PLAN['envs'] for seed in PLAN['seeds']]
    logs = []
    failures = []
    state = dict(pid=os.getpid(), started=now(), total=len(jobs), completed=0, failures=failures)

    def status(stage, **details):
        state.update(stage=stage, updated=now(), **details)
        tmp = out / 'batch-status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(out / 'batch-status.json')
        print(now(), stage, state.get('current', ''), flush=True)

    try:
        status('setup')
        subprocess.run(['bash', str(HERE / 'setup.sh')], check=True)
        python = str(ROOT / '.goal3-venv/bin/python')
        for algo, env, seed in jobs:
            key = f'{algo}/{env}/seed-{seed}'
            status('training', current=key)
            # Keep incomplete or previously launched runs intact. A restart picks
            # the first completed attempt or creates a separate numbered retry.
            bases = [out, out / 'attempts' / '1', out / 'attempts' / '2']
            done = next((base / key for base in bases if complete(base / key)), None)
            if done is None:
                for base in bases:
                    if (base / key).exists():
                        continue
                    cmd = [python, '-u', str(HERE / 'sweep.py'), algo, '--env', env,
                           '--seed', str(seed), '--output', str(base)]
                    result = subprocess.run(cmd)
                    if result.returncode == 0 and complete(base / key):
                        done = base / key
                        break
                    print(now(), 'Failed attempt:', base / key, flush=True)
            if done is None:
                failures.append(key)
            else:
                logs.append(str(done / 'data'))
                state['completed'] = len(logs)
        if logs:
            status('exporting', current='')
            subprocess.run([python, str(HERE / 'analyze.py'), '--runs', *logs,
                            '--out', str(out / 'report'), '--no-compile'], check=True)
        status('incomplete' if failures else 'completed', current='')
        return int(bool(failures))
    except BaseException as exc:
        status('failed', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'goal3-results')
    parser.add_argument('--foreground', action='store_true', help='Use inside a Renku Job')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.pop('GOAL3_BATCH_CHILD', '') == 'yes':
        return run_batch(out)
    with (out / 'batch.lock').open('a') as lock, (out / 'batch.log').open('a', buffering=1) as log:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f'A batch is already running in {out}. See {out / "batch-status.json"}')
        process = subprocess.Popen([sys.executable, '-u', __file__, '--output', str(out)],
                                   env={**os.environ, 'GOAL3_BATCH_CHILD': 'yes'},
                                   pass_fds=(lock.fileno(),), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=not args.foreground)
        print(f'Batch started (PID {process.pid}). Log: {log.name}', flush=True)
        return process.wait() if args.foreground else 0


if __name__ == '__main__':
    sys.exit(main())
