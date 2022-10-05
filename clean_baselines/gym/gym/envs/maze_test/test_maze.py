import time
import sys
sys.path.insert(0, '/storage/rsonecha/hrl/clean_baselines/gym/gym/envs/maze_test')
sys.path.insert(0, '/storage/rsonecha/hrl/clean_baselines/gym/')
import numpy as np


from maze_generators.obstacle_maze import generate_obstacle_maze
from maze_generators.prim_maze import generate_prim_maze
from maze_generators.w_maze import generate_w_maze
from maze_generators.pitchfork_maze import generate_pitchfork_maze
from maze_generators.block_maze import generate_block_maze

from maze_manager_2d import MazeEnv2D
from maze_manager_3d import MazeEnv3D

from PIL import Image, ImageDraw
import cv2
import mujoco_py

class Holder(object):
    def __init__(self):
        self.agent_type = 'humanoid' #'ant'
        self.maze_size_scaling = 8

# def viewer_setup(env, mode, camera_name=None):
#     no_camera_specified = camera_name is None
#     if no_camera_specified:
#         camera_name = "track"
#     if env.viewer is None:
#         if mode == "human":
#             env.viewer = mujoco_py.MjViewer(env.sim)
#         elif mode == "rgb_array" or mode == "depth_array":
#             env.viewer = mujoco_py.MjRenderContextOffscreen(env.sim, -1)
#     env.viewer_setup()

# maze = generate_pitchfork_maze(10, 8)
# maze = generate_block_maze(10, 10, 1)
maze = generate_w_maze(num_u=1, length_lower=3, length_upper=3, bridge_length_lower=1, bridge_length_upper=1)
# maze = generate_block_maze(20, 20, 2)
# maze = generate_prim_maze(20, 20)

env = MazeEnv2D(maze)
env3d = MazeEnv3D(env, args = Holder())

env3d.reset()
import pdb; pdb.set_trace()
image2 = Image.fromarray(env3d.render(mode='rgb_array'))
image2.save("/storage/rsonecha/test_images/3d_maze_start.png")
import pdb; pdb.set_trace()

def process_image(image, cycle, pos):
    image = pad_image(image)
    image = add_text(image, cycle, pos)
    return np.array(image)

def pad_image(image, padding=150):
    width, height = image.size
    new_height = height + 150
    larger_image = Image.new(image.mode, (width, new_height), 'white')
    larger_image.paste(image, (0, 0))
    return larger_image

def add_text(image, cycle, pos):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.text((100, height - 100), f'Cycle: {cycle+1} Position: {pos}', (0,0,0))
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

# cycles = 20
# steps = 10
# frames = []
# for i in range(cycles):
#     env3d.reset()
#     import pdb; pdb.set_trace()
#     image = Image.fromarray(env3d.render(mode='rgb_array'))
#     pos = env3d.get_xy()
#     frame = process_image(image, i, pos)
#     frames.append(frame)
#     for j in range(steps):
#         env3d.step(np.random.rand(17)*10)
#         pos = env3d.get_xy()
#         image = Image.fromarray(env3d.render(mode='rgb_array'))
#         frame = process_image(image, i, pos)
#         frames.append(frame)
# write_mp4(frames, "/storage/rsonecha/test_images/humanoid_test.mp4", fps=10)
