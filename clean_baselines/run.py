import os
import argparse
import pathlib
current = pathlib.Path(__file__).parent.resolve()
import sys
sys.path.insert(0, os.path.join(current, 'gym'))  # prevent gym imports from breaking


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-n', type=int, default=2)
parser.add_argument('-config', type=str, default='baseline_configs.yml')
parsed = parser.parse_args()


cmd = f'python -u run_baselines.py --seed {parsed.seed} -config {parsed.config}'
for thread in range(parsed.n - 1):
    cmd += f' & python -u run_baselines.py --seed {parsed.seed + thread + 1} -config {parsed.config}'
print(cmd, flush=True)
os.system(cmd)
