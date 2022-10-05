import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('-n', type=int, default=2)
parser.add_argument('-config', type=str, default='configs.yml')
parsed = parser.parse_args()

cmd = f'python -u ddpg_agent.py --seed {parsed.seed} -config {parsed.config}'
for thread in range(parsed.n - 1):
    cmd += f' & python -u ddpg_agent.py --seed {parsed.seed + thread + 1} -config {parsed.config}'
print(cmd, flush=True)
os.system(cmd)
    
