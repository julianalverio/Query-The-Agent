import argparse
import sys
import os
import pathlib
import json

current = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, os.path.join(current, '..'))
sys.path.insert(0, os.path.join(current, '../hindsight_experience_replay'))
sys.path.insert(0, os.path.join(current, '../vds'))
from utils import ArgsHolder
# import gym
# from envs.maze_manager import generate_maze
# from ddpg_agent import ArgsHolder
# from vds.baselines.ve_run import train
import yaml
import subprocess


def main():
    current = pathlib.Path(__file__).parent.resolve()
    args = get_args_and_env()
    json_save_dir = os.path.join(current, args.experiment_path)
    if not os.path.exists(json_save_dir):
        os.mkdir(json_save_dir)
    json_save_path = os.path.join(current, args.experiment_path, 'baseline_configs.json')
    file = open(json_save_path, 'w+')
    json.dump(args.to_json(), file)
    file.close()
    

    log_root = os.path.join(current, 'logs')
    # now call the script that will run the job you want
    if args.alg == 'entropy':
        env = 'Maze2D-v1'
        max_steps = args.max_steps
        replay_size = args.max_steps
        seed = args.seed
        env_max_step = args.env_max_step
        experiment_name = args.experiment_name

        os.chdir(os.path.join(os.getcwd(), "entropy_maximization"))
        cmd = "python experiments/mega/train_mega.py --env={} --max_steps={} --seed={} --replay_size={} --env_max_step={} --parent_folder={} --ag_curiosity minkde --num_envs {} --epoch_len {}".format(env, max_steps, seed, replay_size, env_max_step, args.experiment_path, args.num_envs, args.epoch_length)
        os.system(cmd)

    if args.alg == 'vds':
        os.chdir(os.path.join(os.getcwd(), "vds"))
        cmd = f'python -m baselines.ve_run --env={args.env_name} --alg={args.vds_alg} --num_timesteps={args.num_timesteps} --size_ensemble={args.size_ensemble} --log_path={args.experiment_path} --seed={args.seed} --save_interval={args.save_interval} --T={args.rollout_length} --n_test_rollouts={args.n_test_rollouts}'
        os.system(cmd)

def get_args_and_env():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('-config', type=str, default='baseline_configs.yml')
    parsed = parser.parse_args()
    configs = yaml.safe_load(open(parsed.config, 'r'))
    print('Loaded configs!')
    args = ArgsHolder(configs)
    if parsed.seed is not None:
        args.seed = parsed.seed

    if args.maze_type == 'e':
        experiment_name = f'{args.alg}_{args.maze_height}x5E'
    elif args.maze_type == 'w':
        experiment_name = f'{args.alg}_{args.maze_num_u}U5'
    else:
        raise TypeError(f'Invalid maze type {args.maze_type}')

    current = pathlib.Path(__file__).parent.resolve()

    args.experiment_name = experiment_name
    args.experiment_path = os.path.join(current, f'logs/{experiment_name}_seed{args.seed}')
    return args

if __name__ == '__main__':
    main()
