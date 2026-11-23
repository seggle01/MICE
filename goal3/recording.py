"""Durable, independent training metrics; no changes to algorithm updates."""
import csv
import json
import math
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class RunRecorder:
    def __init__(self, out, cost_limit, interval=1000):
        self.out = Path(out)
        self.limit = cost_limit
        self.interval = interval
        self.start = time.monotonic()
        self.stats = dict(environment_steps=0, total_training_cost=0.0,
                          completed_episodes=0, episode_threshold_crossings=0,
                          summed_episode_excess_cost=0.0)
        self.epoch_returns, self.epoch_costs = [], []
        self.last_step = -1
        self.files = []
        self.episodes = self.writer('episodes.csv', ['episode', 'environment_steps', 'return',
            'cost', 'length', 'exceeds_limit', 'excess_cost', 'cumulative_violations',
            'cumulative_excess_cost', 'wall_seconds'])
        self.steps = self.writer('step_totals.csv', ['environment_steps', 'total_training_cost',
            'completed_episodes', 'episode_threshold_crossings', 'summed_episode_excess_cost', 'wall_seconds'])
        self.epochs = self.writer('epochs.csv', ['epoch', 'environment_steps', 'mean_episode_return',
            'mean_episode_cost', 'episodes_in_epoch', 'violations_in_epoch', 'violation_rate_in_epoch',
            'total_training_cost', 'completed_episodes', 'cumulative_violations',
            'cumulative_violation_rate', 'cumulative_excess_cost', 'epoch_seconds', 'rollout_seconds', 'update_seconds', 'wall_seconds'])
        self.snapshot()

    def writer(self, name, fields):
        stream = (self.out / name).open('w', newline='')
        self.files.append(stream)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        return writer

    def flush(self):
        for stream in self.files:
            stream.flush()

    def record_step(self, costs):
        costs = [float(c) for c in costs]
        if not all(math.isfinite(c) for c in costs):
            raise RuntimeError('Non-finite raw training cost.')
        before = self.stats['environment_steps']
        self.stats['environment_steps'] += len(costs)
        self.stats['total_training_cost'] += sum(costs)
        if before // self.interval != self.stats['environment_steps'] // self.interval:
            self.snapshot()

    def record_episode(self, ret, cost, length):
        ret, cost, length = float(ret), float(cost), int(length)
        if not math.isfinite(ret) or not math.isfinite(cost):
            raise RuntimeError('Non-finite episode result.')
        self.stats['completed_episodes'] += 1
        violation = int(cost > self.limit)
        excess = max(0., cost - self.limit)
        self.stats['episode_threshold_crossings'] += violation
        self.stats['summed_episode_excess_cost'] += excess
        self.epoch_returns.append(ret)
        self.epoch_costs.append(cost)
        self.episodes.writerow({'episode': self.stats['completed_episodes'],
            'environment_steps': self.stats['environment_steps'], 'return': ret, 'cost': cost,
            'length': length, 'exceeds_limit': violation, 'excess_cost': excess,
            'cumulative_violations': self.stats['episode_threshold_crossings'],
            'cumulative_excess_cost': self.stats['summed_episode_excess_cost'],
            'wall_seconds': time.monotonic() - self.start})
        self.flush()

    def snapshot(self):
        data = dict(self.stats, wall_seconds=time.monotonic() - self.start)
        # A snapshot at the same step is allowed: episode completion follows step logging.
        self.steps.writerow({k: data[k] for k in self.steps.fieldnames})
        self.last_step = self.stats['environment_steps']
        self.flush()
        atomic_json(self.out / 'live_status.json', data)

    def end_epoch(self, epoch, timing=None):
        timing = timing or {}
        n = len(self.epoch_costs)
        violations = sum(c > self.limit for c in self.epoch_costs)
        completed = self.stats['completed_episodes']
        self.epochs.writerow(dict(epoch=epoch, environment_steps=self.stats['environment_steps'],
            mean_episode_return=sum(self.epoch_returns)/n if n else '',
            mean_episode_cost=sum(self.epoch_costs)/n if n else '', episodes_in_epoch=n,
            violations_in_epoch=violations, violation_rate_in_epoch=violations/n if n else '',
            total_training_cost=self.stats['total_training_cost'], completed_episodes=completed,
            cumulative_violations=self.stats['episode_threshold_crossings'],
            cumulative_violation_rate=self.stats['episode_threshold_crossings']/completed if completed else '',
            cumulative_excess_cost=self.stats['summed_episode_excess_cost'],
            epoch_seconds=timing.get('Time/Epoch', ''), rollout_seconds=timing.get('Time/Rollout', ''),
            update_seconds=timing.get('Time/Update', ''), wall_seconds=time.monotonic()-self.start))
        self.epoch_returns.clear()
        self.epoch_costs.clear()
        self.snapshot()

    def close(self):
        self.snapshot()
        for stream in self.files:
            stream.close()
