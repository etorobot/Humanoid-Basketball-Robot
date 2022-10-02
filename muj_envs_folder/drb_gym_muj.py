#this file has list of classes for environments
import numpy as np, os
from gym import utils, error, spaces
from gym.envs.mujoco import mujoco_env

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class MujKClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        # FILE_PATH = 'humanoid.xml' # Absolute path to your .xml MuJoCo scene file 
                        # OR.
        # FILE_PATH = mujoco_py.load_model_from_path('/home/admin/mujokaleido/KSTL.xml')
        FILE_PATH = os.path.join('/home/admin/mujokaleido', 'KSTL.xml')
        frame_skip = 1
        mujoco_env.MujocoEnv.__init__(self, FILE_PATH, frame_skip)

    def _get_obs(self):
      # Observation of environment fed to agent. This should never be called
      # directly but should be returned through reset_model and step    
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, action):
        # Carry out one step 
        # Don't forget to do self.do_simulation(a, self.frame_skip)
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        alive_bonus = 5.0
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos

        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
      
        done = bool(False)
        info = dict(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost, reward_impact=-quad_impact_cost)

        return self._get_obs(), reward, done, info

    def reset_model(self):
        # Reset model to original state. 
        # This is called in the overall env.reset method
        # do not call this method directly.
        c = 0.082
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        # Position the camera
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
