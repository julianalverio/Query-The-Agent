import mujoco_py
import time
import sys
sys.path.insert(0, '/storage/jalverio/hrl/clean_baselines/gym/gym/envs/maze_test')
sys.path.insert(0, '/storage/jalverio/hrl/clean_baselines/gym/')
import numpy as np
from PIL import Image


from maze_generators.obstacle_maze import generate_obstacle_maze
from maze_generators.prim_maze import generate_prim_maze
from maze_generators.w_maze import generate_w_maze
from maze_generators.pitchfork_maze import generate_pitchfork_maze
from maze_generators.block_maze import generate_block_maze
from maze_generators.e_maze import generate_e_maze

from maze_manager_2d import MazeEnv2D
from maze_manager_3d import MazeEnv3D


class Holder(object):
    def __init__(self):
        self.agent_type = 'humanoid'
        self.maze_size_scaling = 8
        self.dense_reward = False
        

def render_time(duration):
    start = time.time()
    while time.time() - start < duration:
        env3d.render(mode='human')

def save():
    Image.fromarray(env3d.render(mode='rgb_array')).save('/storage/jalverio/test.png')

maze = generate_block_maze(10, 10, 1)
maze = generate_w_maze(num_u=5, length_lower=5, length_upper=5, bridge_length_lower=1, bridge_length_upper=1)
# maze = generate_block_maze(20, 20, 2)
maze = generate_e_maze(13, 5, 1, 1)

# maze = generate_obstacle_maze(10, 10)
# maze = generate_prim_maze(15, 15)
env = MazeEnv2D(maze)
image = env.render(show_agent=False)

env.compute_optimal_q_function()
import pdb; pdb.set_trace()
env3d = MazeEnv3D(env, args = Holder())
env3d.reset()
env3d.wrapped_env.set_xy(np.array([48., 32.]))

# for swimmer
# for _ in range(5):
#     env3d.step([-1, 1])
# env3d.wrapped_env.set_z(0.5) # for swimmer

human_viewer = mujoco_py.MjViewer(env3d.wrapped_env.sim)
rgb_array_viewer = mujoco_py.MjRenderContextOffscreen(env3d.wrapped_env.sim, -1)
# env3d.set_viewer(human_viewer, 'human')
env3d.set_viewer(rgb_array_viewer, 'rgb_array')

# env3d.render(mode='human')
env3d.render(mode='rgb_array')


# breakpoint()
env3d.viewer.cam.lookat[0] = 30.
env3d.viewer.cam.lookat[1] = 0.
env3d.viewer.cam.lookat[2] = 0.5
env3d.viewer.cam.azimuth = 270
env3d.viewer.cam.elevation = -45.
env3d.viewer.cam.distance = 260.
# azimuth, elevation, distance

save()
import os; os.system('stty sane')
import sys; sys.exit()
breakpoint()
render_time(100)
import pdb; pdb.set_trace()

