import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env

'''
Action space: (2,)
Observation space: 6-dim
0: x coordinate
1: y coordinate
2: rotation
3-5: qvel
'''
class BallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
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
        ob = self._get_obs()
        done = False
        info = dict()
        reward = 0
        return ob, reward, done, info

    def get_achieved_goal(self):
        return self.sim.data.qpos.flat.copy()

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()

    def set_z(self, z):
        pass

    def set_xy(self, xy):
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return self.physics.data.qpos[:2]
