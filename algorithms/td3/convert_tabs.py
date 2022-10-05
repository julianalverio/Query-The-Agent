import os

with open('/storage/jalverio/hrl/Meta-SAC/TD3/main.py', 'r') as f:
    lines = f.readlines()
    new_lines = list()
    for line in lines:
        new_line = line.replace('\t', '  ')
        new_lines.append(new_line)
with open('/storage/jalverio/hrl/Meta-SAC/TD3/main.py', 'w+') as f:
    for line in new_lines:
        f.write(line)

