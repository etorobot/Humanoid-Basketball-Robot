
class KBallRandomClass(KHumanoidv3Class):
    def __init__(self): KHumanoidv3Class.__init__()
    def step(self): KHumanoidv3Class.step()
    # def reset_model(self): KHumanoidv3Class.reset_model()
    def reset_model(self): KHumanoidv3Class.step()
        # print ("REEEEEEEEEEEEEEEEEEEEESEEEEEEEEEEEEEET")

class KHumanoidStandupClass(globalClass):

    def __init__(self): self.global_init()

    def step(self, action):
        actiondeg = np.rad2deg(action)
        # print ("step:", action)
        
        self.do_simulation(actiondeg, self.frame_skip)
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1
        if stepCount >= 10000:
            done = bool(True)
        else: done = bool(False)

        return (
            self.global_get_obs(), reward, done,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        global stepCount; stepCount = 0
        c = 0.01
        bed_pose = np.array([0, 0, 0.18,
                             0, .866, 0, .866, 
                             0, 0, 0, 0 #TORSO
                            ,0, 0, 0
                            ,0, 0, 0, 0, 0
                            ,0, 0, 0
                            ,0, 0, 0, 0, 0
                            ,0, 0, 0, 0
                            ,0, 0
                            ,0, 0, 0, 0
                            ,0, 0
                            ,.6, 0, 1 #BALL XYZ
                            ,randy(self,-1,1), randy(self,-1,1), randy(self,-1,1), randy(self, -1,1) #BALL QUATERNION
                    ])
        # print (len(bed_pose))
        self.set_state(
            bed_pose,
            # self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,),
        )
        return self.global_get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 3.0
        self.viewer.cam.lookat[2] = 0.1
        self.viewer.cam.azimuth = 200
        self.viewer.cam.elevation = -22
        # for key, value in DEFAULT_CAMERA_CONFIG.items():
        #     if isinstance(value, np.ndarray):
        #         getattr(self.viewer.cam, key)[:] = value
        #     else: setattr(self.viewer.cam, key, value)        

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.50,
    "lookat": np.array((0.0, 0.0, .5)),
    "elevation": 0 #-12.0,
    ,"azimuth": 235
}
def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class KHumanoidClass(my_mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        my_mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        done = bool(False)

        return (
            self.global_get_obs(),
            reward,
            done,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,),
        )
        return self.global_get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
