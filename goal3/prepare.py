"""Prepare IP3O's documented OmniSafe integration without changing source clones."""
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / '.goal3-sources'
RUNTIME = ROOT / 'work' / 'runtimes' / 'ip3o'


def prepare():
    ROOT.mkdir(exist_ok=True)
    import yaml
    sources = json.loads((HERE / 'sources.json').read_text())
    for name, source in sources.items():
        repo = ROOT / name
        if not repo.exists():
            subprocess.run(['git', 'clone', source['url'], str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'checkout', source['commit']], check=True)
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != source['commit']:
            raise RuntimeError(f'{name}: expected {source["commit"]}, got {actual}. Update sources deliberately.')
    # IP3O ships only an algorithm and config, requiring an OmniSafe host.
    # MICE's host shares identical PPO/policy-gradient/rollout files with CSPO.
    RUNTIME.mkdir(parents=True, exist_ok=True)
    dest = RUNTIME / 'omnisafe'
    shutil.copytree(ROOT / 'MICE' / 'omnisafe', dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    relative = Path('algorithms/on_policy/penalty_function')
    shutil.copy2(ROOT / 'IP3O/omnisafe' / relative / 'ip3o.py', dest / relative / 'ip3o.py')
    init = dest / relative / '__init__.py'
    init.write_text(init.read_text() + '\nfrom .ip3o import IP3O\n__all__.append("IP3O")\n')
    config = yaml.safe_load((ROOT / 'IP3O/omnisafe/configs/on-policy/IP3O.yaml').read_text())
    # Required by the source but absent from its YAML. Paper §5.2 uses 0.1.
    config['defaults']['algo_cfgs']['alpha'] = 0.1
    (dest / 'configs/on-policy/IP3O.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Prepared IP3O runtime: {RUNTIME}')


if __name__ == '__main__':
    prepare()
