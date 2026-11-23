"""Export observed learning curves, editable PGFPlots, and a LaTeX report."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'goal3-results'
COLORS = {'MICE': '#d95f59', 'IP3O': '#2878b5', 'CSPO': '#38905b'}
METRICS = [('reward', 'Mean episode return'), ('cost', 'Mean episode cost'),
           ('total_cost', 'Total training cost'), ('violations', 'Violating episodes'),
           ('rate', 'Violation fraction'), ('excess', 'Total excess cost')]


def tex(value):
    table = {'\\': r'\textbackslash{}', '_': r'\_', '&': r'\&', '%': r'\%',
             '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}'}
    return ''.join(table.get(c, c) for c in str(value))


def latest_roots():
    # Select the latest completed run per task and seed across split batches.
    selected = {}
    for algo in COLORS:
        for path in sorted((ROOT / algo).glob('*/seed-*/data/manifest.json')):
            m = json.loads(path.read_text())
            if not (path.parent / 'episodes.csv').exists():
                continue
            args = m['arguments']
            key = algo, args['env'], args['seed']
            rank = (m['status'] == 'completed', m['started_utc'])
            if key not in selected or rank > selected[key][0]:
                selected[key] = (rank, path.parent)
    roots = [item[1] for item in selected.values()]
    return roots


def gather(roots, validation, window):
    runs, seen, notes = [], set(), []
    for root in roots:
        for path in sorted(root.rglob('manifest.json')):
            m = json.loads(path.read_text())
            args = m.get('arguments', {})
            if args.get('mode') == 'smoke' and not validation:
                notes.append(f'Excluded smoke test: {path.parent}')
                continue
            if m.get('status') in ('initialized', 'initializing'):
                continue
            episode_path = path.parent / 'episodes.csv'
            if not episode_path.exists():
                continue
            try:
                episodes = pd.read_csv(episode_path).dropna(subset=['environment_steps', 'return', 'cost'])
            except pd.errors.EmptyDataError:
                continue
            if episodes.empty:
                continue
            if not np.isfinite(episodes[['environment_steps', 'return', 'cost']].to_numpy()).all():
                raise ValueError(f'Non-finite episode observations: {episode_path}')
            algo, env, seed = args['algo'], args['env'], args['seed']
            key = (algo, env, seed)
            if key in seen:
                raise ValueError(f'Duplicate seed for {key}; select only one batch per algorithm.')
            seen.add(key)
            limit = args.get('cost_limit', 25.)
            episodes = episodes.sort_values(['environment_steps', 'episode'])
            episodes['violations'] = (episodes['cost'] > limit).cumsum()
            episodes['excess'] = (episodes['cost'] - limit).clip(lower=0).cumsum()
            episodes['rate'] = episodes['violations'] / np.arange(1, len(episodes) + 1)
            episodes['reward'] = episodes['return'].rolling(window, min_periods=1).mean()
            episodes['cost_smoothed'] = episodes['cost'].rolling(window, min_periods=1).mean()
            step_file = path.parent / 'step_totals.csv'
            if step_file.exists():
                steps = pd.read_csv(step_file).drop_duplicates('environment_steps', keep='last')
            elif validation:
                # Legacy checks consist only of complete 1,000-step episodes.
                if not np.isclose(episodes['length'].sum(), episodes['environment_steps'].iloc[-1]):
                    raise ValueError('Cannot reconstruct total cost from incomplete legacy episodes.')
                steps = pd.DataFrame({'environment_steps': episodes['environment_steps'],
                                      'total_training_cost': episodes['cost'].cumsum()})
            else:
                raise ValueError(f'Missing raw step-cost log: {step_file}')
            steps = steps.sort_values('environment_steps')
            if not np.isfinite(steps[['environment_steps', 'total_training_cost']].to_numpy()).all():
                raise ValueError(f'Non-finite cost log: {step_file}')
            runs.append(dict(algo=algo, env=env, seed=seed, limit=limit, episodes=episodes,
                             steps=steps, path=path.parent, status=m['status'], manifest=m))
    return runs, notes


def value_at(run, grid, metric):
    if metric == 'total_cost':
        frame, column = run['steps'], 'total_training_cost'
    else:
        frame = run['episodes']
        column = 'cost_smoothed' if metric == 'cost' else metric
    frame = frame.drop_duplicates('environment_steps', keep='last')
    x, y = frame['environment_steps'].to_numpy(), frame[column].to_numpy()
    if metric in ('violations', 'rate', 'excess'):
        return y[np.searchsorted(x, grid, side='right') - 1]
    return np.interp(grid, x, y)


def export_curves(runs, out):
    groups = {}
    for run in runs:
        groups.setdefault((run['env'], run['algo']), []).append(run)
    curves = {}
    for (env, algo), members in sorted(groups.items()):
        # Use only the overlap across all available seeds; never extrapolate a seed.
        low = max(r['episodes']['environment_steps'].iloc[0] for r in members)
        high = min(r['episodes']['environment_steps'].iloc[-1] for r in members)
        if high < low:
            continue
        grid = np.unique(np.concatenate([r['episodes']['environment_steps'].to_numpy() for r in members]))
        grid = grid[(grid >= low) & (grid <= high)]
        data = {'steps': grid}
        for metric, _ in METRICS:
            values = np.array([value_at(r, grid, metric) for r in members])
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=0)
            data[metric] = mean
            data[metric + '_std'] = std
            data[metric + '_lower'], data[metric + '_upper'] = mean - std, mean + std
        frame = pd.DataFrame(data)
        name = f'{env}_{algo}.csv'
        frame.to_csv(out / 'curves' / name, index=False)
        curves[(env, algo)] = dict(frame=frame, name=name, n=len(members), limit=members[0]['limit'])
    return curves


def figures(curves, out, validation, suffix=''):
    if len({env for env, _ in curves}) > 2:
        for family in ('Goal', 'Circle'):
            figures({key: val for key, val in curves.items() if family in key[0]}, out, validation, family)
        return
    envs = sorted(set(env for env, _ in curves))
    if not envs:
        return
    for name, metrics in [('learning_curves', METRICS[:2]), ('safety_metrics', METRICS[2:])]:
        name += suffix
        fig, axes = plt.subplots(len(metrics), len(envs), figsize=(6.0 * len(envs), 2.7 * len(metrics)),
                                 squeeze=False, sharex='col', constrained_layout=True)
        for row, (metric, label) in enumerate(metrics):
            for col, env in enumerate(envs):
                ax = axes[row, col]
                for algo, color in COLORS.items():
                    if (env, algo) not in curves:
                        continue
                    item = curves[(env, algo)]
                    frame = item['frame']
                    x = frame['steps'].to_numpy()
                    y = frame[metric].to_numpy()
                    ax.plot(x, y, color=color, linewidth=1.7, label=f'{algo} (n={item["n"]})')
                    if item['n'] > 1:
                        ax.fill_between(x, frame[metric+'_lower'].to_numpy(), frame[metric+'_upper'].to_numpy(),
                                        color=color, alpha=.16, linewidth=0)
                if metric == 'cost':
                    limit = next(item['limit'] for (task, _), item in curves.items() if task == env)
                    ax.axhline(limit, color='#333333', linestyle='--', linewidth=1, label=f'Cost limit {limit:g}')
                if row == 0:
                    ax.set_title(env.replace('Safety', ''), fontsize=12)
                    ax.legend(fontsize=8, loc='best')
                ax.set_ylabel(label)
                ax.grid(alpha=.25)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                if row == len(metrics)-1:
                    ax.set_xlabel('Training environment steps')
        if validation:
            fig.suptitle('VALIDATION ONLY — short execution checks, not research results', fontsize=11, color='#9b2c2c')
        for extension in ['pdf', 'png', 'svg']:
            fig.savefig(out / 'figures' / f'{name}.{extension}', dpi=180)
        plt.close(fig)
        pgfplots(curves, envs, metrics, out / 'figures' / f'{name}.tex', name)


def pgfplots(curves, envs, metrics, path, figure_name):
    width = '.82' if len(envs) == 1 else '.40'
    height = '.38' if len(envs) == 1 else '.29'
    lines = [r'\begin{tikzpicture}',
        r'\begin{groupplot}[group style={group size=' + f'{len(envs)} by {len(metrics)}' +
        r',horizontal sep=1.35cm,vertical sep=1.5cm},width=' + width + r'\textwidth,height=' + height +
        r'\textwidth,grid=major,grid style={gray!18},tick label style={font=\scriptsize},' +
        r'label style={font=\small},title style={font=\small},scaled x ticks=false,xticklabel style={/pgf/number format/fixed,font=\scriptsize}]']
    for row, (metric, label) in enumerate(metrics):
        for col, env in enumerate(envs):
            options = [f'ylabel={{{label}}}']
            if row == 0:
                options.append('title={' + tex(env.replace('Safety', '')) + '}')
            if row == len(metrics)-1:
                options.append('xlabel={Training environment steps}')
            legend = row == 0 and col == 0
            if legend:
                options.extend([f'legend to name={figure_name}legend', 'legend columns=-1',
                                'legend style={font=\\scriptsize,draw=none}'])
            lines.append(r'\nextgroupplot[' + ','.join(options) + ']')
            for algo in COLORS:
                if (env, algo) not in curves:
                    continue
                item = curves[(env, algo)]
                file = 'curves/' + item['name']
                tag = f'p{row}{col}{algo}'
                if item['n'] > 1:
                    for edge in ['lower', 'upper']:
                        lines.append(r'\addplot[name path=' + tag + edge + r',draw=none,forget plot] table[x=steps,y=' +
                                     metric + '_' + edge + ',col sep=comma]{' + file + '};')
                    lines.append(r'\addplot[' + algo + 'color,fill opacity=0.16,draw=none,forget plot] fill between[of=' +
                                 tag + 'lower and ' + tag + 'upper];')
                lines.append(r'\addplot[' + algo + r'color,thick] table[x=steps,y=' + metric +
                             ',col sep=comma]{' + file + '};')
                if legend:
                    lines.append(r'\addlegendentry{' + algo + f' ($n={item["n"]}$)' + '}')
            if metric == 'cost':
                valid = [v for (task, _), v in curves.items() if task == env]
                low = min(v['frame']['steps'].iloc[0] for v in valid)
                high = max(v['frame']['steps'].iloc[-1] for v in valid)
                limit = valid[0]['limit']
                lines.append(r'\addplot[black,dashed,forget plot] coordinates {(' + str(low) + ',' + str(limit) + ') (' +
                             str(high) + ',' + str(limit) + ')};')
    lines.extend([r'\end{groupplot}', r'\end{tikzpicture}', r'\par\smallskip\ref{' + figure_name + 'legend}'])
    path.write_text('\n'.join(lines) + '\n')


def summary(runs, out, window):
    rows = []
    envs = sorted(set(r['env'] for r in runs))
    for env in envs:
        members = [r for r in runs if r['env'] == env]
        limits = set(r['limit'] for r in members)
        if len(limits) != 1:
            raise ValueError(f'Different cost limits in {env}; use separate reports.')
        horizon = int(min(r['episodes']['environment_steps'].iloc[-1] for r in members))
        for run in members:
            ep = run['episodes'].loc[run['episodes']['environment_steps'] <= horizon]
            steps = run['steps'].loc[run['steps']['environment_steps'] <= horizon]
            if ep.empty or steps.empty:
                continue
            rows.append({'environment': env, 'algorithm': run['algo'], 'seed': run['seed'],
                'status': run['status'], 'comparison_steps': horizon,
                'actual_logged_steps': int(run['steps']['environment_steps'].iloc[-1]),
                'mean_return_last_window': ep['return'].tail(window).mean(),
                'mean_cost_last_window': ep['cost'].tail(window).mean(),
                'total_training_cost': steps['total_training_cost'].iloc[-1],
                'completed_episodes': len(ep), 'violating_episodes': int((ep['cost'] > run['limit']).sum()),
                'violation_fraction': (ep['cost'] > run['limit']).mean(),
                'summed_excess_cost': (ep['cost']-run['limit']).clip(lower=0).sum(),
                'source': str(run['path'])})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'summary_per_seed.csv', index=False)
    lines = []
    if not rows:
        lines.append(r'No training observations are available yet. Run IP3O and CSPO, then refresh this report.')
    else:
        for env in envs:
            lines += [r'\subsection*{' + tex(env) + '}',
                      r'All methods below are evaluated at the same observed training horizon. Values are means across seeds; $\pm$ denotes population standard deviation, not a confidence interval.',
                      r'\begin{center}\small', r'\begin{tabular}{lrrrrr}\toprule',
                      r'Method & Seeds & Steps & Return & Mean cost & Total cost \\ \midrule']
            env_frame = frame[frame.environment == env]
            for algo, group in env_frame.groupby('algorithm'):
                def fmt(column, digits=1):
                    values = group[column].to_numpy()
                    text = f'{values.mean():.{digits}f}'
                    return '$' + text + (r'\pm ' + f'{values.std():.{digits}f}' if len(values)>1 else '') + '$'
                lines.append(f'{algo} & {len(group)} & {group.comparison_steps.iloc[0]:,} & ' +
                             fmt('mean_return_last_window') + ' & ' + fmt('mean_cost_last_window') + ' & ' + fmt('total_training_cost') + r' \\')
            lines += [r'\bottomrule\end{tabular}\end{center}', r'\begin{center}\small',
                      r'\begin{tabular}{lrrr}\toprule', r'Method & Violating episodes & Violation fraction & Excess cost \\ \midrule']
            for algo, group in env_frame.groupby('algorithm'):
                n = len(group)
                values = []
                for column in ['violating_episodes', 'violation_fraction', 'summed_excess_cost']:
                    data = group[column].to_numpy()
                    precision = 3 if column == 'violation_fraction' else 1
                    values.append('$' + f'{data.mean():.{precision}f}' + (r'\pm ' + f'{data.std():.{precision}f}' if n>1 else '') + '$')
                lines.append(algo + ' & ' + ' & '.join(values) + r' \\')
            lines += [r'\bottomrule\end{tabular}\end{center}']
    (out / 'results.tex').write_text('\n'.join(lines) + '\n')


def report(runs, curves, notes, out, window, validation, compile_pdf):
    template = HERE / 'report_template.tex'
    target = out / 'report.tex'
    if not target.exists():
        shutil.copy2(template, target)
    if not (out / 'discussion.tex').exists():
        (out / 'discussion.tex').write_text(r'''\section{Discussion (editable)}
Write your interpretation here after inspecting the curves and matched-budget tables.
Discuss reward together with training cost, violation frequency and severity.
Do not interpret a low final cost as evidence of safe training throughout.
Interpret differences across tasks and seeds at the common interaction budget.
''')
    status = [r'\section{Data status}']
    if validation:
        status.append(r'\textbf{VALIDATION ONLY: these are short execution checks, not scientific experiment results.}')
    if runs:
        status.append(f'This export contains {len(runs)} runs. All statistics describe training, not independent final-policy evaluation.')
        partial = sum(r['status'] != 'completed' for r in runs)
        if partial:
            status.append(rf'\textbf{{Partial report: {partial} runs are unfinished or interrupted.}} Only recorded data are shown.')
        defaults = json.loads((HERE / 'experiment_defaults.json').read_text())
        planned = {(algo, env, seed) for algo in {r['algo'] for r in runs}
                   for env in defaults['envs'] for seed in defaults['seeds']}
        observed = {(r['algo'], r['env'], r['seed']) for r in runs}
        missing = len(planned - observed)
        if missing:
            status.append(rf'\textbf{{Incomplete planned comparison: {missing} configured task--seed runs have no observations in this export yet.}}')
        seeds = sorted(set(r['seed'] for r in runs))
        status.append('Available seed IDs: ' + ', '.join(map(str, seeds)) + '.')
        if any(v['n'] < 2 for v in curves.values()):
            status.append('At least one curve has only one seed; no uncertainty band is drawn for that curve.')
        if 'MICE' not in {r['algo'] for r in runs}:
            status.append(r'\textbf{MICE results are not included.} MICE is the reference method motivating the protocol, not a plotted or measured baseline in this export.')
    else:
        status.append('Training has not been started by this preparation. Results and figures will populate automatically from the selected run folders.')
    status.append(f'Return and episode cost curves use a trailing window of {window} completed episodes within each seed. Cumulative metrics are unsmoothed.')
    (out / 'status.tex').write_text('\n\n'.join(status) + '\n')
    figure_text = []
    for name, caption in [('learning_curves', 'Training return (top) and raw episode cost (bottom). The dashed line is the cost limit, 25. This follows the layout of MICE Figure 3, using our own logged metrics.'),
                          ('safety_metrics', 'Accumulated raw training cost, number of episodes above the cost limit, cumulative violation fraction, and accumulated episode excess cost.')]:
        if curves:
          for suffix in (['Goal', 'Circle'] if len({env for env, _ in curves}) > 2 else ['']):
            figure_text.append(r'\begin{figure}[p]\centering\input{figures/' + name + suffix + r'.tex}\caption{' + caption +
                r' Lines show means across available seeds; shading is $\pm1$ population standard deviation when at least two seeds exist. Each curve stops at the shortest available horizon for its seeds.}\end{figure}')
        else:
            figure_text.append(r'\paragraph{' + name.replace('_', ' ').title() + '} Pending training data.')
    (out / 'generated_figures.tex').write_text('\n'.join(figure_text))
    index = [{'path': str(r['path']), 'algorithm': r['algo'], 'environment': r['env'], 'seed': r['seed'],
              'status': r['status'], 'limit': r['limit'], 'source_commits': r['manifest'].get('sources', {})} for r in runs]
    (out / 'included_runs.json').write_text(json.dumps(index, indent=2))
    (out / 'analysis_notes.json').write_text(json.dumps({'window_episodes': window,
        'band': 'population standard deviation across seeds; not standard error or confidence interval',
        'interpolation': 'linear for smoothed reward/cost and raw cost; previous observation for counts/rates',
        'table_horizon': 'shortest logged completed-episode horizon across included runs for each task',
        'validation_only': validation, 'notes': notes}, indent=2))
    if compile_pdf:
        engine = shutil.which('pdflatex') or '/Library/TeX/texbin/pdflatex'
        if not Path(engine).exists():
            print('LaTeX source ready; pdflatex unavailable, so PDF compilation was skipped.')
            return
        with (out / 'latex-build.log').open('w') as log:
            for _ in range(2):
                result = subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                                        cwd=out, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'LaTeX compilation failed; see {out / "latex-build.log"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='*', default=[])
    p.add_argument('--latest', action='store_true')
    p.add_argument('--out', type=Path, default=HERE / 'comparison')
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--validation', action='store_true', help='Allow old smoke logs; watermark all outputs')
    p.add_argument('--no-compile', action='store_true')
    args = p.parse_args()
    if args.window < 1:
        p.error('window must be positive.')
    roots = latest_roots() if args.latest else args.runs
    out = args.out.resolve()
    for directory in [out, out / 'curves', out / 'figures']:
        directory.mkdir(parents=True, exist_ok=True)
    runs, notes = gather(roots, args.validation, args.window)
    summary(runs, out, args.window)
    from timing import export_timing
    export_timing(runs, out)
    curves = export_curves(runs, out)
    figures(curves, out, args.validation)
    report(runs, curves, notes, out, args.window, args.validation, not args.no_compile)
    print(f'Exported {len(runs)} observed runs to {out}')


if __name__ == '__main__':
    main()
