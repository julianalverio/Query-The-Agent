import os
import re
from ddpg_agent import Actor, ArgsHolder
import pickle
import json
import gym
from gym.envs.maze_test.maze_manager import generate_maze
from normalizer import normalizer
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


# use case: cd into the dir of a particular seed/run and call this. Leave a video in /storage/jalverio/test.mp4
class VideoGenerator(object):
    def __init__(self, use_normalizer):
        self.use_normalizer = use_normalizer
        root = os.getcwd()
        configs_path = os.path.join(root, 'configs.json')
        with open(configs_path, 'r') as f:
            configs = json.load(f)
            args = ArgsHolder(configs)
        if 'maze' not in args.env_name:
            env = gym.make(args.env_name)
        else:
            env = generate_maze(args)
        try:
            env.render(mode='human')
        except:
            pass
        all_frames = self.generate_all_videos(root, env)
        mp4_path = '/storage/jalverio/test.mp4'
        self.write_mp4(all_frames, mp4_path, fps=7)

    def write_mp4(self, frames, path, fps=2, switch=True):
        if switch:
            for frame in frames:
                red = frame[:, :, 0].copy()
                frame[:, :, 0] = frame[:, :, 2]
                frame[:, :, 2] = red

        shape = (frames[0].shape[1], frames[0].shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(path, fourcc, fps, shape)
        for frame in frames:
            # frame = frame[:shape[1]]
            writer.write(frame)
        writer.release()
        
    def generate_all_videos(self, dir_path, env):
        pattern = 'actor(?P<iteration>\d+)\.torch'
        actor_paths = [filename for filename in os.listdir(dir_path) if filename.startswith('actor')]
        actor_paths = sorted(actor_paths, key=lambda x:int(re.match(pattern, x).groups()[0]))
        all_frames = list()
        for actor_path in actor_paths:
            match = re.match(pattern, actor_path)
            iteration = int(match.groups()[0])
            print(iteration)
            full_actor_path = os.path.join(dir_path, actor_path)
            actor = torch.load(full_actor_path)
            g_norm_path = f'g_norm{iteration}.pkl'
            o_norm_path = f'o_norm{iteration}.pkl'
            with open(g_norm_path, 'rb') as f:
                g_norm = pickle.load(f)
            with open(o_norm_path, 'rb') as f:
                o_norm = pickle.load(f)
            all_frames.extend(self.generate_video(actor, env, g_norm, o_norm, iteration))
        return all_frames

    def generate_video(self, actor, env, g_norm, o_norm, iteration):
        observation = env.reset()
        obs = observation['observation']
        goal = observation['desired_goal']
        frames = list()
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frame = self.process_image(frame, iteration)
        frames.append(np.array(frame))
        for timestep in range(50):
            with torch.no_grad():
                input_tensor = self.preproc_inputs(obs, goal, o_norm, g_norm)
                action = actor(input_tensor).detach().cpu().numpy().squeeze()
            observation = env.step(action)[0]
            obs = observation['observation']
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            frame = self.process_image(frame, iteration)
            frames.append(frame)
        return frames

    def preproc_inputs(self, obs, g, o_norm, g_norm):
        if len(obs.shape) == len(g.shape) == 1:
            obs = obs[None, :]
            g = g[None, :]
        if self.use_normalizer:
            obs_norm = o_norm.normalize(obs)
            g_norm = g_norm.normalize(g)
        else:
            obs_norm = obs
            g_norm = g
        return torch.FloatTensor(np.concatenate([obs_norm, g_norm], axis=1))

    def process_image(self, image, iteration):
        image = self.pad_image(image)
        image = self.add_text(image, iteration)
        return np.array(image)

    def pad_image(self, image, padding=150):
        width, height = image.size
        new_height = height + 150
        larger_image = Image.new(image.mode, (width, new_height), 'white')
        larger_image.paste(image, (0, 0))
        return larger_image

    def add_text(self, image, iteration):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/storage/jalverio/arial.ttf", 40)
        width, height = image.size
        draw.text((100, height - 100), f'Iteration: {iteration}', (0,0,0), font=font)
        return image


VideoGenerator(use_normalizer=False)
