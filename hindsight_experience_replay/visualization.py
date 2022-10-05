import sys
import scipy
import copy
import cv2
import re
import torch
import pickle
import os
import yaml
import numpy as np

from ddpg_agent import ArgsHolder
import pathlib
current = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, os.path.join(current, '../clean_baselines/gym/gym/envs/maze_test'))
from maze_manager import generate_maze


from PIL import Image, ImageDraw, ImageFont
from models import Actor, EnsembleCritic

def pad_image(image, padding=150):
    width, height = image.size
    new_height = height + 150
    larger_image = Image.new(image.mode, (width, new_height), 'white')
    larger_image.paste(image, (0, 0))
    return larger_image

def add_text(image, iteration, upper_lower_bound=None):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/storage/jalverio/arial.ttf", 20)
    width, height = image.size
    if upper_lower_bound is None:
        draw.text((0, height - 100), f'Iteration: {iteration}', (0,0,0), font=font)
    else:
        lower_bound = str(round(upper_lower_bound[0], 3))
        upper_bound = str(round(upper_lower_bound[1], 3))
        text = f'Iteration: {iteration} bounds: {lower_bound} {upper_bound}'
        draw.text((0, height - 100), text, (0,0,0), font=font)
    return image
    
def write_mp4(frames, path, fps=2, switch=True):
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


class Visualizer(object):
    def __init__(self):
        configs = yaml.safe_load(open('visualization_configs.yml', 'r'))
        self.args = ArgsHolder(configs)
        self.env = generate_maze(self.args)
        self.env_image = self.env.render(show_agent=False, show_colors=False)

        # root = '/storage/jalverio/hrl/hindsight_experience_replay/vds_logs/exp555/exp555_seed3'
        # root = '/storage/jalverio/hrl/hindsight_experience_replay/vds_logs/exp1018/exp1018_seed1'
        # root = '/storage/jalverio/hrl/hindsight_experience_replay/vds_logs/EMPTYMODELS/EMPTYMODELS_seed0'
        root = '/storage/jalverio/hrl/hindsight_experience_replay/vds_logs/exp1185/exp1185_seed1'
        critic_paths = [path for path in os.listdir(root) if path.startswith('critic')]
        regex = 'critic(?P<iteration>\d+).torch'
        # increasing
        iterations = sorted([int(re.match(regex, critic_path).groups()[0]) for critic_path in critic_paths])

        # HACK, REMOVE THIS
        iterations = iterations[:100]

        if self.args.target_range_lower == 'None' or self.args.target_range_upper == 'None':
            self.args.target_range = None
        else:
            self.args.target_range = (self.args.target_range_lower, self.args.target_range_upper)

        # global norm
        if self.args.heatmap_type != 'forward_uncertainty':
            self.args.global_norm = True
        else:
            self.args.global_norm = False
            
        images = self.generate_heatmaps(root, iterations)

        write_mp4(images, '/storage/jalverio/test.mp4')

    def generate_heatmaps(self, path, iterations):
        # sampled_loc, length = self.get_locations()
        sampled_loc, length = self.env.get_locations()
        all_values = list()
        for iteration in iterations:
            # GRAB HEATMAP VALUES. Higher uncertainty is a larger number
            if self.args.heatmap_type == 'value_distance':
                values = self.get_values(path, iteration, sampled_loc)  # all negative numbers
                values = np.load('/storage/jalverio/hrl/hindsight_experience_replay/optimal_q_values/12x5E.npy')
                if self.args.compute_value_error:
                    values = self.compute_value_error(values)
                else:
                    values *= -1  # larger values must be more uncertain
            elif self.args.heatmap_type == 'forward_uncertainty':
                values = self.get_forward_uncertainty(path, iteration, sampled_loc)
            elif self.args.heatmap_type == 'sample_probability':
                values = self.get_sampling_probability(path, iteration, sampled_loc)
            else:
                raise ValueError('Invalid heatmap type passed in')
            all_values.append(values)

        # NORMALIZE
        all_values = np.stack(all_values, axis=0)  # shape: (n_frames x m_sampled_loc_locations)
        if not self.args.global_norm:
            mins = all_values.min(axis=1)
            all_values -= mins[:, None]
            maxs = all_values.max(axis=1)
            with np.errstate(under='ignore'):
                all_values /= maxs[:, None]
            upper_lower_bounds = [(minimum, maximum) for minimum, maximum in zip(mins, maxs)]
        else:
            lower_bound = all_values.min()
            all_values -= lower_bound
            upper_bound = all_values.max()
            with np.errstate(under='ignore'):
                all_values /= upper_bound
            upper_lower_bounds = [(lower_bound, upper_bound) for _ in range(len(iterations))]

        # GENERATE IMAGES
        all_images = list()
        for idx, (values, iteration) in enumerate(zip(all_values, iterations)):
            if self.args.draw_top_n:
                torch_values = torch.FloatTensor(values)
                top_idxs = torch_values.topk(self.args.draw_top_n)[1]
                image_sampled_loc = sampled_loc[top_idxs]
                values = values[top_idxs]
            else:
                image_sampled_loc = sampled_loc
            image = self.env.draw_heatmap(image_sampled_loc, values, length, copy.deepcopy(self.env_image))
            image = pad_image(image)
            image = add_text(image, iteration, upper_lower_bounds[idx])
            all_images.append(np.array(image))
        return all_images

    # GENERATE THE ACTUAL HEATMAP VALUES

    def get_sampling_probability(self, path, iteration, sampled_loc):
        forward_uncertainty = self.get_forward_uncertainty(path, iteration, sampled_loc)
        scaled_uncertainty = (forward_uncertainty - forward_uncertainty.min())
        scaled_uncertainty /= scaled_uncertainty.max()

        linear_f_slope = 626.58
        linear_f_intercept = -591.18

        likelihoods = linear_f_slope + linear_f_intercept * scaled_uncertainty
        likelihoods = np.clip(likelihoods, a_min=0, a_max=np.inf)
        likelihoods /= likelihoods.sum()

        # rescale likelihoods for contrast
        likelihoods -= likelihoods.min()
        likelihoods /= likelihoods.max()
        likelihoods *= -1
        
        return likelihoods

        # sigma = 0.05
        # with np.errstate(under='ignore'):
        #     probs = scipy.stats.norm.pdf(scaled_uncertainty, loc=1, scale=sigma)
        # return probs


    def get_forward_uncertainty(self, path, iteration, sampled_loc):
        critic_path = os.path.join(path, f'critic{iteration}.torch')
        critic_network = torch.load(critic_path).cpu()
        o_norm_path = os.path.join(path, f'o_norm{iteration}.pkl')
        with open(o_norm_path, 'rb') as f:
            o_norm = pickle.load(f)
        g_norm_path = os.path.join(path, f'g_norm{iteration}.pkl')
        with open(g_norm_path, 'rb') as f:
            g_norm = pickle.load(f)


        n_random_actions = 8
        all_stds = list()
        for action_idx in range(n_random_actions):
            action = self.env.action_space.sample()[None, :]
            action = np.repeat(action, sampled_loc.shape[0], axis=0)
            action = torch.FloatTensor(action)
            obs_norm = o_norm.normalize(sampled_loc)  # assume you're at the loc, taking an action
            goal_norm = g_norm.normalize(np.zeros_like(obs_norm))
            all_input_start_states = torch.FloatTensor(np.concatenate([obs_norm, goal_norm], axis=1))
            multiple_values = critic_network.predict_forward(all_input_start_states, action)
            stds = multiple_values.std(axis=0)
            all_stds.append(stds)

        mean_disagreement = torch.stack(all_stds, axis=0).mean(axis=0).mean(axis=1)
        mean_disagreement = np.array(mean_disagreement.cpu().detach())
        return mean_disagreement

    def get_values(self, path, iteration, sampled_loc):
        critic_path = os.path.join(path, f'critic{iteration}.torch')
        critic_network = torch.load(critic_path).cpu()
        o_norm_path = os.path.join(path, f'o_norm{iteration}.pkl')
        with open(o_norm_path, 'rb') as f:
            o_norm = pickle.load(f)
        g_norm_path = os.path.join(path, f'g_norm{iteration}.pkl')
        with open(g_norm_path, 'rb') as f:
            g_norm = pickle.load(f)

        obs = self.env.get_obs()['observation'][None, :]
        obs = np.repeat(obs, sampled_loc.shape[0], axis=0)
        actions = torch.zeros((sampled_loc.shape[0], 2))
        obs_norm = o_norm.normalize(obs)
        g_norm = g_norm.normalize(sampled_loc)
        all_input_start_states = torch.FloatTensor(np.concatenate([obs_norm, g_norm], axis=1))
        nonmean_values = critic_network(all_input_start_states, actions)
        nonmean_values = torch.clamp(nonmean_values, min=-50.)
        mean_values = np.array(nonmean_values.mean(axis=0).cpu().detach())
        nonmean_values = np.array(nonmean_values.cpu().detach())
        return mean_values

    # MISC. HELPERS

    def get_locations(self):
        num_blocks = 10
        length = 1./(num_blocks)
        sampled_loc = list()
        for block in self.env.free_positions:
            for row_idx in range(num_blocks):
                for col_idx in range(num_blocks):
                    location = block + np.array([row_idx, col_idx]) * length + length/2
                    sampled_loc.append(location)
        return np.array(sampled_loc), length

    def compute_value_error(self, values):
        optimal_values = np.load('/storage/jalverio/hrl/hindsight_experience_replay/optimal_q_values/12x5E.npy')
        error = abs(optimal_values[:, None] - values)
        return error

    # currently unused!
    def find_value_distance(self, sampled_loc, values, image):
        upper_bound = -self.args.target_range[0]
        lower_bound = -self.args.target_range[1]
        values_below_upper = values[:, 0] < upper_bound
        values_above_lower = values[:, 0] > lower_bound
        mask = np.logical_and(values_below_upper, values_above_lower)
        target_points = sampled_loc[mask]
        if target_points.shape[0] != 0:
            image = self.env.draw_points(target_points, image, color='blue')
        return image



if __name__ == '__main__':
    Visualizer()

