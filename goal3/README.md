# Goal 3 on RenkuLab

Compare the released MICE, IP3O and CSPO implementations on Safety Gymnasium PointGoal1, CarGoal1, PointCircle1 and CarCircle1. Each method uses seeds 0–5 on each task and 7,000,000 environment interactions per seed. This gives 24 runs per method, 72 runs and 504,000,000 interactions overall. The raw episode cost limit is 25. MICE and CSPO use 350 epochs of 20,000 interactions. IP3O uses 1,400 epochs of 5,000 interactions.

## Set up a session

Open a terminal in a Linux x86-64 Renku session with internet access, Git and Python/pip. From the repository root on branch `goal3/renku-7m-6seeds`, run:

```bash
bash goal3/setup.sh
```

In the Renku environment builder, select branch `goal3/renku-7m-6seeds` and rebuild from its latest commit. The root `requirements.txt` installs uv for the builder. Once the session opens, the setup command above creates an isolated Python 3.10 environment, installs CPU PyTorch, and fetches the exact algorithm revisions in `goal3/sources.json`. Use two CPU threads and one simulator per run, as in the earlier comparison. Attach writable persistent storage for the output of long runs. The launcher defaults to `goal3-results/` inside the checkout. To store results on a mounted volume, append `--output /your/mounted/path/goal3-results` to each sweep command.

## Train

Run the commands separately, or in order in one session:

```bash
.goal3-venv/bin/python goal3/sweep.py MICE
.goal3-venv/bin/python goal3/sweep.py IP3O
.goal3-venv/bin/python goal3/sweep.py CSPO
```

Append `--list` to inspect the plan without training. To split the sweep across sessions, assign distinct environment/seed pairs, for example:

```bash
.goal3-venv/bin/python goal3/sweep.py MICE --env SafetyPointGoal1-v0 --seed 0
```

Each run writes its console log and a data directory under `ALGORITHM/ENVIRONMENT/seed-N/`. Read `data/live_status.json` during training and `data/epochs.csv` for per-epoch timing. The same sweep command skips completed runs. After an interruption, move the inactive seed directory outside the results tree and rerun the command to restart that seed from the beginning. Do setup once before starting concurrent runs in the same checkout.

## Export results

For the default output directory:

```bash
.goal3-venv/bin/python goal3/analyze.py --latest --out goal3-results/report --no-compile
```

For a custom results location, set RESULTS to its actual path and pass the run directories explicitly:

```bash
.goal3-venv/bin/python goal3/analyze.py --runs "$RESULTS"/*/*/seed-*/data --out "$RESULTS/report" --no-compile
```

The export includes editable LaTeX, curve data, training reward, accumulated raw cost, counts and fractions of episodes above the limit, and per-epoch times. The input logs and checkpoints remain in the per-seed directories.

## Launch all 72 runs unattended

From the repository root, run `bash run.sh`. Setup, MICE, IP3O, CSPO and the
editable report export run sequentially in the background. Use
`bash run.sh --output /actual/mounted/storage/goal3-results` for persistent outputs.
For a Renku Job use `bash run.sh --foreground` with the same output option and
configure the job time limit for the sweep. The checkout must be the working directory.

Read `batch.log` for setup and scheduling, `batch-status.json` for the current
run and final status, and each seed's console.log and data/epochs.csv for details.
Relaunch with the same output path after an interruption. The launcher reuses
completed seeds, preserves partial runs, and allows up to two separate retry
directories per seed. Each retry restarts that seed from step zero.
Only one all-method launcher can run per output directory. Finish or stop any
independently started manual sweep before launching the combined batch.
