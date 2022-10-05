import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env

'''
Action space: (6,)
qpos: 10-dim
qvel: 10-dim
'''
class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
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
        posafter, height, ang = self.sim.data.qpos[0:3]
        height = self.sim.data.qpos[2]
        angle = self.sim.data.qpos[3]
        done = not (height > 0.8 and height < 2.0 and angle > -1.0 and angle < 1.0)
        reward = 0
        info = dict()
        ob = self._get_obs()
        return ob, reward, done, info

    def get_achieved_goal(self):
        return self.sim.data.qpos[:2].flat.copy()

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_z(self, z):
        self.physics.data.set_joint_qpos('rootz', z)

    def set_xy(self, xy):
        self.physics.data.set_joint_qpos('rootx', xy[0])
        self.physics.data.set_joint_qpos('rooty_pos', xy[1])

    def get_xy(self):
        x = self.physics.data.get_joint_qpos('rootx')
        y = self.physics.data.get_joint_qpos('rooty_pos')
        return np.array([x, y])
