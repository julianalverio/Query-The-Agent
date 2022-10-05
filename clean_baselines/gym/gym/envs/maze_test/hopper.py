import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env

'''
Action space: (3,)
Observation:
15-dim
qpos: 7-dim
qvel: 7
'''
class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
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
        return self.step(a)
        
    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        height = self.sim.data.qpos[2]
        angle = self.sim.data.qpos[3]
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[3:]) < 100).all()
            and (height > 0.7)
            and (abs(angle) < 0.2)
        )
        ob = self._get_obs()
        info = dict()
        reward = 0
        return ob, reward, done, info

    def get_achieved_goal(self):
        return self.sim.data.qpos.flat.copy()

    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat, np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_z(self, z):
        self.physics.data.set_joint_qpos('rootz', z)

    def set_xy(self, xy):
        # qpos = np.copy(self.physics.data.qpos)
        # qpos[0] = xy[0]
        # qpos[1] = xy[1]

        # qvel = self.physics.data.qvel
        # self.set_state(qpos, qvel)

        self.physics.data.set_joint_qpos('rootx', xy[0])
        self.physics.data.set_joint_qpos('rooty_pos', xy[1])

    def get_xy(self):
        x = self.physics.data.get_joint_qpos('rootx')
        y = self.physics.data.get_joint_qpos('rooty_pos')
        return np.array([x, y])
        # return self.physics.data.qpos[:2]

