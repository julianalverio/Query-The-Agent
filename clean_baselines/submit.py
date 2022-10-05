import tempfile, shutil, os
import numpy as np
import time
from os.path import expanduser


temp_dir = expanduser("~")
salt = np.random.randint(low=0, high=1000000000)
temp_dir = os.path.join(temp_dir, str(salt))
shutil.rmtree(temp_dir, ignore_errors=True)
os.mkdir(temp_dir)

def temporary_copy(filename):
    new_path = os.path.join(temp_dir, filename)
    shutil.copy2(filename, new_path)
    return new_path

configs_path = temporary_copy('baseline_configs.yml')
slurm_path = temporary_copy('slurm_baselines.sh')
new_slurm_path = os.path.join(temp_dir, 'updated_slurm_supercloud.sh')
with open(slurm_path, 'r') as f:
    lines = f.readlines()
print('Copied!')
with open(new_slurm_path, 'w+') as f:
    for line in lines:
        if 'run.py' in line:
            line = line.replace('\n', ' -config %s\n' % configs_path)
        f.write(line)

os.system('sbatch %s' % new_slurm_path)
