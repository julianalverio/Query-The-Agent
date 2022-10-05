import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env

# Action space shape: (2,)
# Observation:
'''
Action space shape: (2,)
Qpos:
0: x coordinate
1: y coordinate
2: z coordinate
3: Rotation about Z axis (rot0)
4: First joint angle (rot2)
5: Second joint angle (rot3)

Observation:
0: x coordinate
1: y coordinate
2: Rotation about Z axis (rot0)
3: First joint angle (rot2)
4: Second joint angle (rot3)
5-10: qvel
'''
class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file_path):
        mujoco_env.MujocoEnv.__init__(self, file_path, 4)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model

    def _step(self, a):
        self.step(a)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = 0
        done = False
        info = dict()
        return obs, reward, done, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[:2], qpos.flat[3:], qvel.flat])

    def get_achieved_goal(self):
        qpos = self.sim.data.qpos
        return np.concatenate([qpos.flat[:2], qpos.flat[3:]])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()

    def set_z(self, z):
        self.physics.data.set_joint_qpos('slider3', z)

    def set_xy(self, xy):
        self.physics.data.set_joint_qpos('slider1', xy[0])
        self.physics.data.set_joint_qpos('slider2', xy[1])

    def get_xy(self):
        x = self.physics.data.get_joint_qpos('slider1')
        y = self.physics.data.get_joint_qpos('slider2')
        return np.array([x, y])
