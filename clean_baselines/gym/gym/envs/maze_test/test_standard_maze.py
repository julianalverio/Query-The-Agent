import sys
sys.path.insert(0, '/storage/rsonecha/hrl/clean_baselines/gym/gym/envs/maze_test')
sys.path.insert(0, '/storage/rsonecha/hrl/clean_baselines/gym/')
import numpy as np
from PIL import Image, ImageDraw
import cv2
import mujoco_py
import yaml

from maze_manager import generate_maze
from maze_manager_2d import MazeEnv2D
from maze_manager_3d import MazeEnv3D


class ArgsHolder(object):
    def __init__(self, args):
        self.keys = list()
        for key, value in dict(args).items():
            setattr(self, key, value)
            self.keys.append(key)

    def merge(self, args):
        for key, value in dict(args).items():
            if not hasattr(self, key):
                setattr(self, key, value)
                self.keys.append(key)
            else:
                raise KeyError(f'Config attribute {key} already exists')

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

def viewer_setup(env, mode, camera_name=None):
    no_camera_specified = camera_name is None
    if no_camera_specified:
        camera_name = "track"
    if env.viewer is None:
        if mode == "human":
            env.set_viewer(mujoco_py.MjViewer(env.wrapped_env.sim), mode)
        elif mode == "rgb_array" or mode == "depth_array":
            env.set_viewer(mujoco_py.MjRenderContextOffscreen(env.wrapped_env.sim, -1), mode)

def setup_env():
    configs = yaml.safe_load(open('test_configs.yml', 'r'))
    args = ArgsHolder(configs)
    assert args.agent_type in ['ant', 'humanoid', 'ball', 'half_cheetah', 'hopper', 'swimmer']
    env = generate_maze(args)
    viewer_setup(env, 'rgb_array')
    env.viewer.cam.trackbodyid = -1
    # env.viewer.cam.azimuth = 35
    # env.viewer.cam.lookat[0] += 0.5         # x,y,z offset from the object (works if trackbodyid=-1)
    # env.viewer.cam.lookat[1] += 0.5
    # env.viewer.cam.lookat[2] += 0.5
    # env.viewer.cam.distance = env.model.stat.extent * 0.7
    # env.viewer.cam.elevation = -55
    env.viewer.cam.azimuth = args.azimuth
    env.viewer.cam.lookat[0] += args.lookat_0         # x,y,z offset from the object (works if trackbodyid=-1)
    env.viewer.cam.lookat[1] += args.lookat_1
    env.viewer.cam.lookat[2] += args.lookat_2
    env.viewer.cam.distance = env.model.stat.extent * args.distance_ratio
    env.viewer.cam.elevation = args.elevation
    return env, args.agent_type

def save_image(env, path="/storage/rsonecha/test_images/test.png"):
    image = Image.fromarray(env.render(mode='rgb_array', camera_name='track'))
    image.save(path)

def create_video(env, action_space, path, fps=2, cycles=20, steps=20):
    frames = []
    for i in range(cycles):
        env.reset()
        image = Image.fromarray(env.render(mode='rgb_array'))
        pos = env.get_xy()
        frame = process_image(image, i, pos)
        frames.append(frame)
        for j in range(steps):
            env.step(np.random.rand(action_space)*10)
            pos = env.get_xy()
            image = Image.fromarray(env.render(mode='rgb_array'))
            frame = process_image(image, i, pos)
            frames.append(frame)
    write_mp4(frames, path, fps=10)

if __name__ == "__main__":
    env, agent_type = setup_env()
    img_path = "/storage/rsonecha/test_images/{}_test.png".format(agent_type)
    save_image(env, img_path)
    if agent_type == 'ant':
        action_space = 8
    if agent_type == 'humanoid':
        action_space = 17
    if agent_type == 'ball':
        action_space = 2
    if agent_type == 'half_cheetah':
        action_space = 6
    if agent_type == 'hopper':
        action_space = 3
    if agent_type == 'swimmer':
        action_space = 2
    mp4_path = "/storage/rsonecha/test_videos/{}_test.mp4".format(agent_type)
    create_video(env, action_space, mp4_path, fps=10)
