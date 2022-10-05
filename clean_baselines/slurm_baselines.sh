#!/bin/bash
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --time UNLIMITED
#SBATCH -a 0,3
#SBATCH -o /home/gridsan/jalverio/output%j.txt


eval "$(conda shell.bash hook)"
source /etc/profile
#module load anaconda/2021a
conda activate baselines
python -u /home/gridsan/jalverio/hrl/clean_baselines/run.py --seed $SLURM_ARRAY_TASK_ID -n 3
