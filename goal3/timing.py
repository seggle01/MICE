"""Readable summaries of the native trainer's per-epoch timing."""
import pandas as pd


def export_timing(runs, out):
    records = []
    for run in runs:
        progress = next((run['path'] / 'training').rglob('progress.csv'), None)
        if not progress or not progress.stat().st_size:
            continue
        df = pd.read_csv(progress)
        for _, r in df.iterrows():
            records.append(dict(algorithm=run['algo'], environment=run['env'], seed=run['seed'],
                                epoch=int(r['Train/Epoch']) + 1, epoch_seconds=r['Time/Epoch'],
                                rollout_seconds=r['Time/Rollout'], update_seconds=r['Time/Update']))
    if not records:
        return
    df = pd.DataFrame(records).sort_values(['algorithm','environment','seed','epoch'])
    df.to_csv(out / 'epoch_times.csv', index=False, float_format='%.4f')
    lines = ['# Epoch timing', '', 'Measured training seconds per epoch. Rollout and update times are listed separately in epoch_times.csv.', '',
             '| Algorithm | Environment | Seeds | Epochs logged | Mean seconds | Median seconds | Total training minutes |',
             '|---|---|---:|---:|---:|---:|---:|']
    tex = [r'\subsection*{Training time per epoch}', r'\begin{center}\small\begin{tabular}{llrrr}\toprule',
           r'Method & Task & Epochs logged & Mean (s) & Median (s) \\ \midrule']
    for (algo, env), group in df.groupby(['algorithm','environment']):
        t = group.epoch_seconds
        lines.append(f'| {algo} | {env} | {group.seed.nunique()} | {len(t)} | {t.mean():.2f} | {t.median():.2f} | {t.sum()/60:.1f} |')
        tex.append(f'{algo} & {env} & {len(t)} & {t.mean():.2f} & {t.median():.2f}' + r' \\')
    (out / 'TIMING.md').write_text('\n'.join(lines)+'\n')
    (out / 'timing.tex').write_text('\n'.join(tex+[r'\bottomrule\end{tabular}\end{center}']))
