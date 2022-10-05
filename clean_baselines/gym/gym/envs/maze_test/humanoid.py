import numpy as np
import mujoco_py
from gym.envs.mujoco import mujoco_env
from gym import utils


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

'''
Action space: (17,)
qpos: 24-dim
qvel: 23-dim
obs: 378-dim
'''
class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file_path):
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat,
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def _step(self, a):
        return self.step(a)

    def get_achieved_goal(self):
        return self.sim.data.qpos.flat.copy()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        qpos = self.sim.data.qpos
        reward = 0
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        info = dict()
        return (self._get_obs(), reward, done, info)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        # self.viewer.cam.lookat[2] = 2.0
        # self.viewer.cam.elevation = -20

    def set_z(self, z):
        qpos = np.copy(self.physics.data.qpos)
        qpos[2] = z

        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def set_xy(self, xy):
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return self.physics.data.qpos[:2]
