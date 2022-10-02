from collections import OrderedDict
import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
from . import my_mjviewer

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        # print ("DICT", observation.items())
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        za_pie = np.full(observation.shape, 3.1416)
        metres = np.full(observation.shape, 3)
        toInfinityAndBeyond = np.full(observation.shape, float('inf'))
        # print ("NDA NDARRAY")
        # if observation.shape == (2,) and np.amin(observation) >= -0.5236: #TORSO JOINT POSs
        #     # print ("za obs", observation)
        #     chest_lo = np.full(observation.shape, 0.5236)
        #     chest_hi = np.full(observation.shape, 1.2218)
        #     # space = spaces.Box(-za_pie, za_pie, dtype=observation.dtype)
        #     space = spaces.Box(-chest_lo, chest_hi, dtype=observation.dtype)
        # elif observation.shape == (4,) and np.amin(observation) >= -1 and np.amax(observation) <= 1: #FOR THE QUATERNIONS
        #     # print ("QUAT OBS DICT", observation)
        #     quat_one = np.full(observation.shape, 1)
        #     space = spaces.Box(-quat_one, quat_one, dtype=observation.dtype)
        # elif observation.shape == (3,) and np.amax(observation) < 1.1 and np.amin(observation) >= -1.1: #size 3 for XYZ
        #     # print ("XYZ DICT", observation)
        #     space = spaces.Box(-metres, metres, dtype=observation.dtype)
        # elif observation.shape == (3,) and np.amin(observation) < -3.1 and np.amax(observation) > 1.1: #size 3 for SENSaz
        #     # print ("SENSAZ DICT 3 > 1.2", observation)
        #     space = spaces.Box(-toInfinityAndBeyond, toInfinityAndBeyond, dtype=observation.dtype)                        
        # elif observation.shape == (11,3) or observation.shape == (9,3): #ARM LEG XYZ POSITIONS
        #     space = spaces.Box(-metres, metres, dtype=observation.dtype)
        # elif observation.shape == (16,) or observation.shape == (8,): #ARM J POSITIONS
        #     space = spaces.Box(-za_pie, za_pie, dtype=observation.dtype)        
        # elif observation.shape == (12,) or observation.shape == (6,): #LEG J POSITIONS
        #     leg_low = np.full(observation.shape, 1.92)
        #     leg_high = np.full(observation.shape, 2.444)
        #     # space = spaces.Box(-za_pie, za_pie, dtype=observation.dtype)        
        #     space = spaces.Box(-leg_low, leg_high, dtype=observation.dtype)        
        # elif observation.shape == (32,) and np.amax(observation) < 3.15: #32 JOINT POS
        #     space = spaces.Box(-za_pie, za_pie, dtype=observation.dtype)
        # elif observation.shape == (32,) and np.amax(observation) > 3.15 or np.amin(observation) < -3.15: #32 J VEL
        #     vel = np.full(observation.shape, 12)
        #     space = spaces.Box(-vel, vel, dtype=observation.dtype) 
        # elif isinstance(observation, bool):
        #     space = spaces.Box(False, True, dtype=observation.dtype)
        # else:
        space = spaces.Box(-toInfinityAndBeyond, toInfinityAndBeyond, dtype=observation.dtype)

    else:
        raise NotImplementedError(type(observation), observation)

    return space

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        self.mCounter = 0
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        # print ("1", dir(self.sim.data))
        # print ("MODEL", dir(self.sim.data))
        # print ("ssetStatTE", dir(self.sim.set_state))
        # print ("setCONSTANTS", dir(self.sim.set_constants))
        # print ("ssetStatTE", dir(self.sim.set_state))
        # print ("STEP", dir(self.sim.step))
        self.data = self.sim.data
        self.contact = self.data.contact
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        bounds = np.deg2rad(self.model.actuator_ctrlrange.copy())#; print(bounds)
        # bounds = (self.model.actuator_ctrlrange.copy()/180.0); print(bounds)
        low, high = bounds.T
        # low, high = np.ones([2], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        # print ("OLD STATE", old_state[4])
        self.sim.set_state(new_state)
        self.sim.forward()

    def mid_reset(self):
        for _ in range(1):
            old_state = self.sim.get_state()
            # print ("JOIN", self.sim.data.qpos[:41])      
            # join = np.concatenate(self.sim.data.qpos[:41], [0,0,0,0,0])
            # print ("JOINEDDD", join)  
            # lis1 = [0, 1]; l2 = [9,0,6]
            # lis1.extend(l2)
            # print ("EXTEND", lis1)
            bot = self.sim.data.qpos[:37]; velReset = self.sim.data.qvel[:36]
            botlist = bot.tolist(); velList = velReset.tolist()
            botlist.extend([.4, 0, 1, 0,0,0,0]); velList.extend([0,0,0,0,0,0])
            self.sim.set_state(mujoco_py.MjSimState(old_state.time,
                                botlist, 
                                [x*1 for x in velList], #[1]*len(velList), 
                                old_state.act, old_state.udd_state))
            # print ("JOINEDDD",((len(self.sim.data.qvel))))
            # self.sim.set_state(old_state)
            # self.sim.forward()
            # self.sim.step()        
            self.sim.data.cinert[48] = 0,0,0,0,0,0,0,0,0,0
            self.sim.data.cvel[48] = 0,0,0,0,0,0
            self.sim.data.xfrc_applied[48] = 0,0,0,0,0,0        # print (len(self.sim.data.qvel))f

        # return 0
    def ballStuck(self):
        for _ in range(1):
            old_state = self.sim.get_state()
            bot = self.sim.data.qpos[:37]; velReset = self.sim.data.qvel[:36]
            botlist = bot.tolist(); velList = velReset.tolist()
            ballXYZ = self.sim.data.qpos[37:]
            botlist.extend(ballXYZ); velList.extend([0,0,0,0,0,0])
            self.sim.set_state(mujoco_py.MjSimState(old_state.time,
                                botlist, 
                                [x*1 for x in velList], #[1]*len(velList), 
                                old_state.act, old_state.udd_state))
            # print ("JOINEDDD",((len(self.sim.data.qvel))))
            # self.sim.set_state(old_state)
            # self.sim.forward()
            # self.sim.step()
            self.sim.data.cinert[48] = 0,0,0,0,0,0,0,0,0,0
            self.sim.data.cvel[48] = 0,0,0,0,0,0
            self.sim.data.xfrc_applied[48] = 0,0,0,0,0,0 # print (len(self.sim.data.qvel))f

        # return 0
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        # print ("mCOUNT % 1", self.mCounter % 11)
        if self.mCounter>=0 :
            self.sim.data.ctrl[:] = ctrl
        # elif self.mCounter % 50 ==0:
        #     # self.sim.data.qpos[15] = 1.5
        #     # self.sim.data.qvel[14] = 0
        #     self.sim.data.ctrl[:] = ctrl
        # else: self.sim.data.ctrl[:] = ctrl
        # self.sim.data.xfrc_applied[48] = [0,0,6.0822,0,0,0]
        # self.sim.data.cinert[48] = 0,0,0,0,0,0,0,0,0,0
        # self.sim.data.cvel[48] = [0,0,0,0,0,0]
        for _ in range(n_frames):
            self.sim.step()
        self.mCounter +=1
    def do_force(self, ctrl, n_frames):
        self.sim.data.xfrc_applied[48] = ctrl
        self.sim.data.cinert[48] = 0,0,0,0,0,0,0,0,0,0
        self.sim.data.cvel[48] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()
            # self.viewer.add_marker(pos=[[self.sim.data.body_xpos[48][0], self.sim.data.body_xpos[48][1], (self.sim.data.body_xpos[48][2]+0.3)]],
            #                         type=5,
            #                         size=[.01, .03, .05],
            #                         rgba=[1, 0, 0, 0.1],
            #                         label=str("BALL_MARKER"))

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = my_mjviewer.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
    
    # -----------------------------
