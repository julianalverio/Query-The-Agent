#!/bin/bash
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --time UNLIMITED
#SBATCH -a 0-9
##SBATCH -a 10-19
#SBATCH -o /home/gridsan/jalverio/output%j.txt


eval "$(conda shell.bash hook)"
source /etc/profile
module load anaconda/2021a
####conda activate new_torch
python /home/gridsan/jalverio/hrl/algorithms/main.py --config /home/gridsan/jalverio/hrl/algorithms/self_play.yml --seed $SLURM_ARRAY_TASK_ID --log_dir bob_v0
