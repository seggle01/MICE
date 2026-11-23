#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then
  python3 -m pip install --user uv
  export PATH="$(python3 -m site --user-base)/bin:$PATH"
fi
uv venv --allow-existing --python 3.10 .goal3-venv
py=.goal3-venv/bin/python
if [[ $(uname) == Linux ]]; then
  uv pip install --python "$py" --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1'
fi
uv pip install --python "$py" -r goal3/requirements.txt
uv pip check --python "$py"
"$py" goal3/prepare.py
