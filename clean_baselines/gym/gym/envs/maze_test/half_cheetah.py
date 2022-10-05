import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env

'''
Action space: (6,)
Qpos:
0. x coordinate
1. y coordinate
2. z coordinate
3. Rotation about Y-axis
4. Back thigh angle
5. Back shin angle
6. Back foot angle
7. Front thigh angle
8. Front shin angle
9. Front foot angle

Observation space:
- qpos (0-9)
- qvel (10-16)
'''
class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file_path):
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
          return self.sim
        else:
          return self.model

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        reward = 0
        done = False
        info = dict()
        return ob, reward, done, dict()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    def get_achieved_goal(self):
        return self.sim.data.qpos.flat.copy()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_z(self, z):
        self.physics.data.set_joint_qpos('rootz', z)

    def set_xy(self, xy):
        self.physics.data.set_joint_qpos('rootx', xy[0])
        self.physics.data.set_joint_qpos('rooty_pos', xy[1])

    def get_xy(self):
        x = self.physics.data.get_joint_qpos('rootx')
        y = self.physics.data.get_joint_qpos('rooty_pos')
        return np.array([x, y])


