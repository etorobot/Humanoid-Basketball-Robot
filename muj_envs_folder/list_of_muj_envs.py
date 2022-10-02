#this file has list of classes for environments
# from gym.envs.mujoco import mujoco_env
from gym import utils
from . import my_mujoco_env
# from mujoco_py import cymj
# import matplotlib.pyplot as plt
import numpy as np, os #, json, mujoco_py
# import xml.etree.ElementTree as elementTree
# from ../randomized_locomotion import RandomizedLocomotionEnv
# the_xml = os.path.join('/home/jovyan/private/mujokaleido', 'KSTL.xml')
def quatToRPY(self, quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_Y = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_X = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.rad2deg(roll_Y), np.rad2deg(pitch_X), np.rad2deg(yaw_z)    
def unifNoise(self, real, range=None):
    if range==None: range = 0.1048 #6 degrees of NOIZZZ
    gauss = np.random.normal(real, 0.0523)
    noise = np.random.uniform(-range, (range*1.001))
    noisyObs = real + noise
    # return noisyObs
    return gauss
def save_obs(self, currentObs, no_frames):
    for i in no_frames:
        self.sim
    return None
def randy(self, low, high):
    return np.random.uniform(low, high)
def poseDR(self, DRrange, value):
    lower = 1 - DRrange
    higher = 1 + DRrange
    return np.random.uniform(lower*value, higher*value)

class globalClass(my_mujoco_env.MujocoEnv, utils.EzPickle):
    # global stepCount; stepCount = 0
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 3
        self.viewer.cam.distance = 4#self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = .8
        self.viewer.cam.elevation = -5
        self.viewer.cam.azimuth = 240
    def global_init(self, **kwargs):
        self.poseCount = 'khi'
        
        if   self.poseCount =='srv':    the_xml = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'KSrv.xml')
        elif self.poseCount =='spk':    the_xml = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'KSpike.xml')
        elif self.poseCount =='drb':    the_xml = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'KDrb.xml')
        elif self.poseCount =='khi':    the_xml = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'KSTL.xml')
        elif self.poseCount =='hold' or 'catch':    the_xml = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'KHold.xml')
        
        frameSkip = 1; self.epNo, self.stepper, self.longStepper, self.Curriculum, self.poseEps, self.number_of_bounces = 0,0,0,0,0,0
        self.no_oneHand_touches,  self.touchesReward = 0, 0
        
        self.resetTorso, self.resetLeftFoot, self.resetRightFoot, self.allVEL = [],[],[],[]
        self.currCounter, self.difficultyLength = 0, 500_000   
        self.terminate = 0; self.HeadQuatVel, self.HeadQuatVelb4, self.BallBeenTouched = 0, 0, 0
        self.act2 = [0]*32; self.allPOS = []
        self.act_KHI = [
            0,                      #CP
            -.0873, -.5236, 0, 0, 0    #LSP_LSR_LSY_LEP
            ,.0873, -.5236, 0, 0, 0    #RSP_RSR_RSY_REP
            ,0 , 0, -.2967             #LCR LCP
            # , .6283     #LKP
            ,0 , 0, -.2967             #RCR RCP
            # , .6283     #RKP
        ]
        self.act_hold =[
            0,
            -.2618, -.35, 0
            ,-1.31, 0  #L_EP
            , -.2618, .35, 0
            , -1.31, 0  #REP

            , 0, 0        #LCR
            ,-.2967     #LCP
            , 0, 0      #RCR
            , -.2967    #RCP
        ]
        self.act_catch =[
            0,
            -.2618, .0873, 0
            ,-1.31, 0  #L_EP
            , -.2618, -.0873, 0
            , -1.31, 0  #REP

            , 0, 0      #LCR
            ,-.2967     #LCP
            , 0, 0         #RCR
            , -.2967    #RCP
        ]
        self.act_drb =[                 #DRB DRB DRB DRB DRB DRB DRB
            0             
            ,-.2618     #LSP 
            , -.1   #LSY 
            ,-.52   #LEP
            ,-.52   #LWP
            , .0    #RSP
            # , .3    #RSY
            ,-.8    #REP
            , 0, 0        #LCY_LCR
            , -.2967    #LCP
            , 0, 0         #RCR
            , -.2967    #RCP
        ]
        self.act_srv =[                 #SERVE SERVE SERVE SERVE SERV
            0,
            .5236   #LSP
            , -1.13 
            ,0      #LWP
            ,.5236
            , -1.13     #REP
            , 0     #RWP
            , 0, -.2967,
            0, -.2967
        ]
        self.act_SPK =[                 #SPIKE SPIKE SPIKE SPIKE SPIKE
            0    #CP
            # -0.9    #LSP
            ,-0.6   #LEP
            ,0      #LWP
            , -1.9, 0    #RSP
            # , .174     #RSR
            , -1.6     #REP
            , 0    #RWP
            , 0, 0, -.2967, #LCR #LCP
            0, 0, -.2967   #RCR #RCP
        ]        

        self.KHI_pose_fixed = [         #POSE_KHI_STAND
            0, 0, .96, 0, 0, 0, 0 #ROOT
            , 0, 0, 0, 0 #chest Y P Head Y P

            , 0, .0873, -.0873 #LSP_LSR_LSY
            # , -.2618, 0.0873, -.0 #left x12

            ,-.5236, 0          #LEP_LEY
            # , 0, 0, 0 #left WP WY hand
            , 0, 0 #left WP KLAW

            , 0, -.0873, .0873 #right Shoulder P R Y
            ,-.5236, 0 #right elbow P Y
            # , 0, 0, 0 #right WP WY hand
            , 0, 0 #right WP R_KLAW

            , 0, 0, -.2967 #left CY CR CP 
            , .6283 #left KP
            , 0, -.3316 # L ankle R LAP
                                    
            , 0, 0, -.2967  #RCY RCR RCP
            , .6283        #RKP 
            , 0, -.3316 # RAR RAP 

            ,-1.2, -.8, .13  #BALL FLOOR
            ,1, 1, 0, 1 #BALL QUATERNION
            ]
        self.hold_pose = [              #POSE_HOLD_BALL_POSE
            0, 0, .95,      #root X Y Z
            0, 0, 0, 0      #root quat
            , 0, 0, 0, 0    #CY CP HY HP
            
            , .2618, 0.0873, -.310 #left x12
            ,-1.75236, 0       #LEP_LEY
            , 0, 0          #LWP_LKLAW

            , .2618, -0.0873, .310 #right Shoulder P R Y
            # , 0, 0, .35 #right Shoulder P R Y
            ,-1.75236, 0           #REP_REY
            , 0, 0              #RWP_RKLAW

            , 0, 0, -.0 #left CY CR CP 
            , 0 #LKP
            , 0, 0 #LAR LAP
                                    
            , 0, 0, -.0 #right CY CR CP
            , 0             # R_KPP 
            , 0, 0 # RAR RAP 

            , .35, 0, 1.0812  #BALL xyz
            ,   1, 1, 0, randy(self, -1,1) #BALL QUATERNION
            ]
        self.dribble_pose = [           #POSE_DRIVVLE_POSE
            0, 0, .95, #root X Y Z
            0, 0, 0, 0 #root quat
            , 0, 0, 0, 0        #CY_CP_HY_HP

            , .0, .0, -.1     #LSP_LR_LSY
            ,-.52, 0             #LEP_LEY
            , -.52, 0           #LWP_LKLAW

            , -.0, 0, 0.95    #RSP_RSR_RSY
            ,-.8, 0         #REP_REY
            , 0, 0          #RWP_RKLAW

            , 0, 0, -.0 #left CY CR CP 
            , 0#.6283       #left KP
            , 0, 0#-.3316 # L ankle R LAP
                                    
            , 0, 0, -.0 #RCY_RCR_RCP
            , 0#.6283       # R KNEE P 
            , 0, 0# -.3316 # RAR RAP 

            ,randy(self,0.2,.6), .25, randy(self,.16,.3) #BALL xyz
            ,0, 0, randy(self,-1,1), 1 #BALL QUATERNION
            ]
        self.catch_pose = [             #POSE_CATCH_POSE
            0, 0, .95, 0, 0, 0, 0 #root
            , 0, 0, 0, 0 #CY CP HY HP
            
            , -.2618, 0.0873, -.0 #left x12
            ,-1.31, 0   #LEP_LEY
            , 0, 0      #LWP L_KLAW

            , -.2618, -0.0873, .0 #right Shoulder P R Y
            ,-1.31, 0   #REP_REY
            , 0, 0      #R_WP R_KLAW

            , 0, 0, -.0 #left CY CR CP 
            , 0.0 #left KP
            , 0, 0#-.3316 # L ankle R LAP
                                    
            , 0, 0, -.0 #right CY CR CP
            , 0.0 # R KNEE P 
            , 0, 0#-.3316 # RAR RAP 

            ,1.95, 0, 1.6  #BALL xyz
            ,0, 1, randy(self,-1,1), 1 #BALL QUATERNION
            ]
        self.srv_pose = [               #POSE DIG DIG DIG
            0, 0, .95, 0, 0, 0, 0 #root
            , 0, 0.5, 0, 0        #CY_CP_HY_HP
            
            , -.52360, 0, -0.27        #LSP_LSR_LSY
            # , 0, 0            #LEP_LEY
            ,-1, 0              #LEP_LEY
            , 0, 0              #LWP_KLAW

            , -.52360, 0, .27    #RSP_RSR_RSY
            # ,0, 0             #REP_REY
            ,-1, 0              #REP_REY
            , 0, 0              #RWP_RKLAW

            , 0, 0, -.0      #LCY_LCR_LCP 
            , 0#.6283           #LKP
            , 0, 0#-.3316       #LAR_LAP
                                    
            , 0, 0, -.0      #RCY_RCR_RCP
            , 0#.6283           #RKP 
            , 0, 0# -.3316      #RAR_RAP 

            ,1.9, 0, 1.8  #BALL xyz
            # ,.9, 0, 1.8  #BALL xyz
            ,1, 1, 1, 1 #BALL QUATERNION
            ]
        self.spk_pose = [               #POSE_SPIKE_SPIKE_SPK
            0, 0, .95, 0, 0, 0, 0 #root
            , 0, 0, 0, 0 #CY_CP_HY_HP
            
            , -.9, -.174, 0     #LSP SR SY
            # , 0, 0        #LEP_LEY
            , -.6, 0        #LEP_LEY
            , 0, 0          #LWP_KLAW

            , -1.9, 0, .0873  #RSP RSR RSY
            # , -2, .1740, 0  #RSP RSR RSY
            , -1.6, 0       #REP_REY
            , 0, 0          #RWP_RKLAW

            , 0, 0, -.0 #left CY CR CP 
            , 0#.6283       #LKP
            , 0, 0#-.3316 # L ankle R LAP
                                    
            , 0, 0, -.0 #RCY_RCR CP
            , 0#.6283       # R_KP 
            , 0, 0# -.3316 # RAR RAP 

            ,.73, .155, 1.35  #BALL xyz
            # ,.69, .155, 1.44  #BALL xyz
            ,1, 1, randy(self,-1,1), 0 #BALL QUATERNION
            ]       
        
        self._terminate_when_unhealthy  =   1
        self._forward_reward_weight     =   1.25
        self._ctrl_cost_weight          =   .5
        self._contact_cost_weight       =   5e-7
        self._contact_cost_range        =   (-np.inf, 10.0)
        self._healthy_reward            =   200
        self._healthy_z_range           =   (.65, 1.2)
        self.heathy_angle               =   75
        self._reset_noise_scale         =   .1
        self._exclude_current_positions_from_observation=True
        
        my_mujoco_env.MujocoEnv.__init__(self, the_xml, frameSkip)
        utils.EzPickle.__init__(**locals()) # utils.EzPickle.__init__(self)
        # cymj.set_pid_control(self.sim.model, self.sim.data)

    def global_get_obs(self):
        # root_XYZ        = self.sim.data.qpos.flat[0:3]
        # root_Quat = self.sim.data.qpos.flat[3:7]
        # waistZQuat      = self.sim.data.qpos.flat[2:7]
        # position        = self.sim.data.qpos.flat.copy()
        # jPOS       = self.sim.data.qpos[7:39] #size is 32
        jPOS       = self.sim.data.qpos[7:37] #30 because KLAW
        # velocity        = self.sim.data.qvel.flat.copy()
        # jVEL       = self.sim.data.qvel[6:38] #size is 32
        jVEL       = self.sim.data.qvel[6:36] #30 cuz KLAW
        # effort          = self.sim.data.qfrc_actuator.flat.copy()
        jEFF         = self.sim.data.qfrc_actuator.flat[6:36] #size is 32
        effort_scaled   = self.sim.data.actuator_force.flat.copy()

        # com_inertia     = self.sim.data.cinert.flat.copy()
        # com_velocity    = self.sim.data.cvel.flat.copy()

        # external_contact_forces = self.sim.data.cfrc_ext.flat.copy()
        # every_body_XYZ          = self.sim.data.body_xpos.flat.copy()
        # botBodies_xyz   = self.sim.data.body_xpos[1:50]
        botBodies_xyz   = self.sim.data.body_xpos[1:48] #KLAW
        Torso_xyz    = self.sim.data.body_xpos[1:10]
        Torso_Z      = Torso_xyz[2::3]
        rel_Waist_xyz = Torso_xyz[0] - [0, 0, 0.95]
        Neck_xyz    = self.sim.data.body_xpos[8]; rel_neck_xyz = Neck_xyz - Torso_xyz[0]
        
        # Arms_xyz     = self.sim.data.body_xpos[10:28]
        Arms_xyz     = self.sim.data.body_xpos[10:27] #KLAW
        # LeftArm_xyz  = self.sim.data.body_xpos[10:19]
        LeftArm_xyz  = self.sim.data.body_xpos[10:18] #KLAW
        # RightArm_xyz = self.sim.data.body_xpos[19:28]
        RightArm_xyz = self.sim.data.body_xpos[18:26] #KLAW
        Larm_Loc = LeftArm_xyz - Torso_xyz[0]; Rarm_Loc = RightArm_xyz - Torso_xyz[0]
        # LEP_xyz    = self.sim.data.body_xpos[14]; REP_xyz = self.sim.data.body_xpos[23]
        LSP_xyz    = self.sim.data.body_xpos[10]; RSP_glo = self.sim.data.body_xpos[18] #shoulder
        LSP_local = LSP_xyz - Torso_xyz[0]; RSP_local = RSP_glo - Torso_xyz[0]
        LEP_xyz    = self.sim.data.body_xpos[14]; REP_xyz = self.sim.data.body_xpos[22] #KLAW
        LEP_relXYZ = LEP_xyz - Torso_xyz[0]; REP_relXYZ = REP_xyz - Torso_xyz[0]
        # self.LHand_xyz    = self.sim.data.body_xpos[18]; self.RHand_xyz = self.sim.data.body_xpos[27]
        self.LHand_xyz    = self.sim.data.body_xpos[17]; self.RHand_xyz = self.sim.data.body_xpos[25]
        LH_local = self.LHand_xyz - Torso_xyz[0]; RH_local = self.RHand_xyz - Torso_xyz[0]
        
        # Legs_xyz    = self.sim.data.body_xpos[28:50].flat
        Legs_xyz    = self.sim.data.body_xpos[26:48].flat
        # LeftLeg_xyz  = self.sim.data.body_xpos[28:39]
        LeftLeg_xyz  = self.sim.data.body_xpos[26:37]
        # L_CKA_X, R_CKA_X   = self.sim.data.body_xpos[32], self.sim.data.body_xquat[43]
        # RightLeg_xyz  = self.sim.data.body_xpos[39:50]
        RightLeg_xyz    = self.sim.data.body_xpos[37:48]
        LLeg_Loc = LeftLeg_xyz - Torso_xyz[0]; RLeg_Loc = RightLeg_xyz - Torso_xyz[0]
        # self.LAP_xyz     = self.sim.data.body_xpos[38]; self.RAP_xyz = self.sim.data.body_xpos[49]
        self.LAP_xyz    = self.sim.data.body_xpos[36]; self.RAP_xyz = self.sim.data.body_xpos[47]
        LAP_rel_xyz     = self.LAP_xyz - Torso_xyz[0]; RAP_rel_xyz = self.RAP_xyz - Torso_xyz[0] 
        LKP_GLO = self.sim.data.body_xpos[32]
        RKP_GLO = self.sim.data.body_xpos[43]

        self.ball_XYZ                = self.sim.data.body_xpos[48]
        self.ball_relXYZ = self.ball_XYZ  - Torso_xyz[0]
        self.ballVel = self.sim.data.cvel[48][3:]
        bodies_Z_noFloor        = self.sim.data.body_xpos.flat[5:250:3]

        every_body_Quat = self.sim.data.body_xquat.flat.copy()
        # botBodies_Quat  = self.sim.data.body_xquat[1:50]
        botBodies_Quat  = self.sim.data.body_xquat[1:48]
        Torso_Quat  = self.sim.data.body_xquat[1:10]
        waist_Quat  = self.sim.data.body_xquat[1]
        neck_Q  = self.sim.data.body_xquat[8]
        LEP_Q   = self.sim.data.body_xquat[14]; REP_Q   = self.sim.data.body_xquat[22]
        LSP_Q   = self.sim.data.body_xquat[10]; RSP_Q = self.sim.data.body_xquat[18]
        LWP_Q, RWP_Q  = self.sim.data.body_xquat[17], self.sim.data.body_xquat[25]
        LAP_Quat, RAP_Quat   = self.sim.data.body_xquat[36], self.sim.data.body_xquat[47]
        LKP_Q, RKP_Q   = self.sim.data.body_xquat[32], self.sim.data.body_xquat[43]
        LCP_Q, RCP_Q   = self.sim.data.body_xquat[28], self.sim.data.body_xquat[39]

        self.waistRPY = quatToRPY(self, waist_Quat); #waist_RPY_rad = np.deg2rad(waistRPY)
        self.neck_RPY = quatToRPY(self, neck_Q); #neck_rpy_rad = np.deg2rad(neck_RPY)
        LHand_RPY =     quatToRPY(self, LWP_Q); #LHand_RPY_rad = np.deg2rad(LHand_RPY)
        RHand_RPY =     quatToRPY(self, RWP_Q); #RHand_RPY_rad = np.deg2rad(RHand_RPY)
        self.Lap_RPY = quatToRPY(self, LAP_Quat); #LAP_rpy_rad = np.deg2rad(Lap_RPY)
        self.Rap_RPY = quatToRPY(self, RAP_Quat); #RAP_rpy_rad = np.deg2rad(Rap_RPY)

        # geom_friction = self.sim.data.efc_frictionloss//
        # print (geom_friction)
        # geom_xpos               = self.sim.data.geom_xpos.flat.copy()
        sensor_data = self.sim.data.sensordata#get_body_com(1)
        self.accel = sensor_data[0:3];  acceleNoise = unifNoise(self, self.accel, 0.)
        self.gyro = sensor_data[2:5];   gyroNoise = unifNoise(self, self.gyro)
        force_LF = sensor_data[6:9];    mom_LF = sensor_data[9:12]
        force_RF = sensor_data[12:15];  mom_RF = sensor_data[15:18]
        self.touch_L, self.touch_R = sensor_data[18:24], sensor_data[24:30]
        # if self._exclude_current_positions_from_observation:
        #     position = position[2:]
        j_Chest = jPOS[:2];    j_chest_noise = unifNoise(self, j_Chest)
        j_Head = jPOS[2:4];    j_head_noise = unifNoise(self, j_Head)
        # j_LArm =  jPOS[4:12];  j_LArm_noise = unifNoise(self, j_LArm)
        j_LArm =  jPOS[4:11];  j_LArm_noise = unifNoise(self, j_LArm)
        j_LShEP =  jPOS[4:8];  j_LShEP_noise = unifNoise(self, j_LShEP)
        # j_RArm = jPOS[12:20];  j_RArm_noise = unifNoise(self, j_RArm)
        j_RArm = jPOS[11:18];  j_RArm_noise = unifNoise(self, j_RArm)
        # j_LLeg = jPOS[20:26];  j_LLeg_noise = unifNoise(self, j_LLeg)
        j_LLeg = jPOS[18:24];  j_LLeg_noise = unifNoise(self, j_LLeg)
        # j_RLeg = jPOS[26:];    j_RLeg_noise = unifNoise(self, j_RLeg)
        j_RLeg = jPOS[24:];    j_RLeg_noise = unifNoise(self, j_RLeg)
        pChP = jPOS[1]; vChP = jVEL[1]
        pLSP, pLSR, pLSY, pLEP, pLEY, pLWP, pLKLAW, pLCY, pLCR, pLCP, pLKP, pLAR, pLAP = jPOS[4], jPOS[5], jPOS[6], jPOS[7], jPOS[8], jPOS[9], jPOS[10], jPOS[18], jPOS[19], jPOS[20], jPOS[21], jPOS[22], jPOS[23]
        vLSP, vLSR, vLSY, vLEP, vLEY, vLWP, vLKLAW, vLCY, vLCR, vLCP, vLKP, vLAR, vLAP = jVEL[4], jVEL[5], jVEL[6], jVEL[7], jVEL[8], jVEL[9], jVEL[10], jVEL[18], jVEL[19], jVEL[20], jVEL[21], jVEL[22], jVEL[23]
        eLSP, eLSR, eLSY, eLEP, eLEY, eLWP, eLKLAW = jEFF[4], jEFF[5], jEFF[6], jEFF[7], jEFF[8], jEFF[9], jEFF[10]
        pRSP, pRSR, pRSY, pREP, pREY, pRWP, pRKLAW, pRCY, pRCR, pRCP, pRKP, pRAR, pRAP = jPOS[11], jPOS[12], jPOS[13], jPOS[14], jPOS[15], jPOS[16], jPOS[17], jPOS[24], jPOS[25], jPOS[26], jPOS[27], jPOS[28], jPOS[29]
        vRSP, vRSR, vRSY, vREP, vREY, vRWP, vRKLAW, vRCY, vRCR, vRCP, vRKP, vRAR, vRAP = jVEL[11], jVEL[12], jVEL[13], jVEL[14], jVEL[15], jVEL[16], jVEL[17], jVEL[24], jVEL[25], jVEL[26], jVEL[27], jVEL[28], jVEL[29]
        eRSP, eRSR, eRSY, eREP, eREY, eRWP, eRKLAW = jEFF[11], jEFF[12], jEFF[13], jEFF[14], jEFF[15], jEFF[16], jEFF[17]

        pLeft_arm    = pLSY, pLEP, pLWP ; pRight_arm   = pRSY, pREP, pRWP
        vLeft_arm    = vLSY, vLEP, vLWP ; vRight_arm   = vRSY, vREP, vRWP
        # pos_left_arm, pos_right_ARM = np.array([pLeft_arm]), np.array([pRight_arm])
        # vel_left_arm, vel_right_ARM = np.array([vLeft_arm]), np.array([vRight_arm])
        # pos_l_leg = jPOS[19:21]; pos_r_leg = jPOS[25:27]
        # vel_l_leg = jVEL[19:21]; vel_r_leg = jVEL[25:27]     

# DRB RSY CR CP
        if self.poseCount=='drb':
            self.allPOS = np.array([pChP, pLSP, pLSY, pLEP, pLWP, pRSR, pRSY, pREP, pLCY, pLCR, pLCP, pRCY, pRCR, pRCP])
            self.allVEL = np.array([vChP, vLSP, vLSY, vLEP, vLWP, vRSR, vRSY, vREP, vLCY, vLCR, vLCP, vRCY, vRCR, vRCP])
# SRV SP EP WP CR CP
        elif self.poseCount=='srv':
            self.allPOS = np.array([pChP, pLSP, pLSY, pLEP, pLWP, pRSP, pRSY, pREP, pRWP, pLCY, pLCR, pLCP, pRCY, pRCR, pRCP])
            self.allVEL = np.array([vChP, vLSP, vLSY, vLEP, vLWP, vRSP, vRSY, vREP, vRWP, vLCY, vLCR, vLCP, vRCY, vRCR, vRCP])
  # SPK
        elif self.poseCount=='spk':
            self.allPOS = np.array([pChP, pLSP, pLSR, pLEP, pLWP, pRSP, pRSR, pRSY, pREP, pRWP, pLCY, pLCR, pLCP, pRCY, pRCR, pRCP])
            self.allVEL = np.array([vChP, vLSP, vLSR, vLEP, vLWP, vRSP, vRSR, vRSY, vREP, vRWP, vLCY, vLCR, vLCP, vRCY, vRCR, vRCP])
  # HOLD SP EP WP CR CP
        elif self.poseCount=='hold' or 'catch' or 'khi':
            self.allPOS = np.array([pChP, pLSP, pLSR, pLSY, pLEP, pLWP, pRSP, pRSR, pRSY, pREP, pRWP, pLCY, pLCR, pLCP, pRCY, pRCR, pRCP])
            self.allVEL = np.array([vChP, vLSP, vLSR, vLSY, vLEP, vLWP, vRSP, vRSR, vRSY, vREP, vRWP, vLCY, vLCR, vLCP, vRCY, vRCR, vRCP])
            # self.allVEL = np.array([vChP, vLSP, vLSR, vLEP, vLWP, vRSP, vRSR, vREP, vRWP, vLCY, vLCR, vLCP, vRCY, vRCR, pRCP])
        # self.allEFF = np.array([eLSP, eLSY, eLEP, eLWP, eRSP, eRSY, eREP, eRWP, jEFF[18], jEFF[24]])

        LandRTouch = np.array([self.bounceCount()[1], self.bounceCount()[2]])
        # print ((self.sim.data.cfrc_ext[17]))
        imuBoth = np.array([self.accel, self.gyro])
        
        if self.longStepper > 500_000:
            pvP = np.random.normal(self.allPOS, .0523); pVV =  np.random.normal(self.allVEL, .01)
        else: pvP, pVV = self.allPOS, self.allVEL

        za_states = {
            # "GYROGRYO": self.gyro, "ACCELERO": self.accel,
            # "ball_ALL": np.array([self.ball_XYZ, self.ballVel]),#, np.array([self.ball_touch_sens])]),
            # "BALL_TOUCH": np.array([self.ball_touch_sens]),
            
            "IMU": imuBoth,
            "TORSO_Q": np.array([waist_Quat, neck_Q]),
            "TORSO_GLO": np.array([Torso_xyz[0], Neck_xyz]),
            "EP_GLO": np.array([LEP_xyz, REP_xyz]),
            "EP_Q": np.array([LEP_Q, REP_Q]),
            "WP_GLO": np.array([self.LHand_xyz, self.RHand_xyz]),
            "WP_Q": np.array([LWP_Q, RWP_Q]),
            "AP_GLO": np.array([self.LAP_xyz, self.RAP_xyz]),
            "AP_Q": np.array([LAP_Quat, RAP_Quat]),
            "MOMS": np.array([mom_LF, mom_RF]),            
            
            # "LH_Q": LWP_Q, "LH_GLOBAL": self.LHand_xyz,
            # "R_HAND_Q": RWP_Q, "RH_GLOBAL": self.RHand_xyz,
            # "LAQ": LAP_Quat, "LAP_GLOBAL": self.LAP_xyz, "LF_MOM": mom_LF,
            # "RAQ": RAP_Quat, "RAP_GLOBAL": self.RAP_xyz, "RF_MOM": mom_RF,
            
            # "WAIST_Q": waist_Quat, "NECK_Q": neck_Q,
            # "WAIST_GLOBAL": Torso_xyz[0], #"WAIST_X": rel_Waist_xyz,
            # "NECK_GLOBAL": Neck_xyz, # "NECK_X": rel_neck_xyz,
            
            # "LH_Q": LWP_Q,
            # "L_ARM_Q": np.array([LEP_Q, LWP_Q]),
            # "L_ARM_GLO": np.array([LEP_xyz, self.LHand_xyz]),
            # "LEP_X": LEP_xyz, 
            # "LH_GLOBAL": self.LHand_xyz,
            # "LSP_Q": LSP_Q, 
            # "LEP_Q": LEP_Q, "LH_Q": LWP_Q, 
            # "LSP_X": LSP_xyz, 
            # "LH_X": LH_local,
            # "L_ARM": LeftArm_xyz,
            # "L_ARM_LOC": Larm_Loc,
            # "L_TOUCH_YES": np.insert(self.touch_L, 0, khi_class.ball_L),
            # "L_TOUCH": self.touch_L,
            # "L_TOUCH_BALL": np.array(self.touch_L),

            # "R_HAND_Q": RWP_Q,
            # "R_ARM_Q": np.array([REP_Q, RWP_Q]),
            # "R_ARM_GLO": np.array([REP_xyz, self.RHand_xyz]),
            # "REP_X": REP_xyz, 
            # "RH_GLOBAL": self.RHand_xyz,
            # "RSP_Q": RSP_Q, 
            # "REP_Q": REP_Q, "RH_Q": RWP_Q,
            # "RSP_X": RSP_glo, 
            # "RH_X": RH_local,
            # "R_ARM": RightArm_xyz,
            # "R_ARM_LOC": Rarm_Loc,
            # "R_TOUCH_YES": np.insert(self.self.touch_R, 0, khi_class.ball_R),
            # "R_TOUCH": self.touch_R,
            # "R_TOUCH_BALL": np.array(self.touch_R),

            # "LAP_QX": np.array([LAP_Quat, self.LAP_xyz]),
            # "LCQ": LCP_Q, "LKQ": LKP_Q, 
            # "LAQ": LAP_Quat, 
            # "L_LEG_Q": np.array([LCP_Q, LKP_Q, (LAP_Quat)]),
            # "L_LEG_Q": np.array([LKP_Q, LAP_Quat]),
            # "L_LEG_X": LeftLeg_xyz,
            # "L_LEG_LOC": LLeg_Loc,
            # "LKP_LAP_GLO": np.array([LKP_GLO, self.LAP_xyz]),
            # "LAP_GLOBAL": self.LAP_xyz,
            # "LA_X": LAP_rel_xyz,
            # "LF_FORCE": force_LF,
            # "LF_MOM": mom_LF,
            # "LF": np.array([force_LF, mom_LF]),

            # "RAP_QX": np.array([RAP_Quat, self.RAP_xyz]),
            # "RCQ": RCP_Q, "RKQ": RKP_Q,
            # "RAQ": RAP_Quat,
            # "R_LEG_Q": np.array([RKP_Q, RAP_Quat]),
            # "R_LEG_X": RightLeg_xyz,
            # "R_LEG_LOC": RLeg_Loc,
            # "RAP_GLOBAL": self.RAP_xyz,
            # "RKP_RAP_GLO": np.array([RKP_GLO, self.RAP_xyz]),
            # "RA_X": RAP_rel_xyz,
            # "RF_FORCE": force_RF, 
            # "RF_MOM": mom_RF,
            # "RF": np.array([force_RF, mom_RF]),
            # "LLRR_MOM": np.array([mom_LF, mom_RF]),

            ##################           Joint_POSSSSSSSSSsition       ############: {
            # "position": jPOS,
            # "pos_CHEST":    j_Chest,
            # "pos_HEAD":     j_Head,
            # "pos_Torso":    jPOS[:4],
            # "pos_Arms": jPOS[4:20],
            # "pos_L_Arm":    j_LArm,
            # "pos_R_Arm":    j_RArm,
            # "pos_Legs": jPOS[20:],
            # "pos_L_Leg":    j_LLeg, "pos_R_Leg":    j_RLeg,
            
            ##################           Joint VELLLLLL             ############: {
            # "velocity":jVEL,
            # "vel_Chest":    jVEL[:2],
            # "vel_HEAD":     jVEL[2:4],
            # "vel_Torso":    jVEL[:4],
            # # "vel_ARMS": jVEL[4:20],
            # "vel_L_ARM":    jVEL[4:11],
            # "vel_R_ARM":    jVEL[11:18],
            # # "vel_LEGZ": jVEL[20:],
            # "vel_L_LEG":    jVEL[18:24],
            # "vel_R_LEG":    jVEL[24:],

            # "eff_Chest":    jeffort[:2],
            # "eff_HEAD":     jeffort[2:4],
            # "eff_L_ARM":    jeffort[4:11],
            # "eff_R_ARM":    jeffort[11:18],
            # "eff_L_LEG":    jeffort[18:24],
            # "eff_R_LEG":    jeffort[24:],
            # "pos_L_Arm":    pos_left_arm,   "pos_R_Arm":    pos_right_ARM,
            # "pos_L_Leg":    pos_l_leg,      "pos_R_Leg":    pos_r_leg,
            # "vel_L_Arm":    vel_left_arm,   "vel_R_Arm":    vel_right_ARM,
            # "vel_L_Leg":    vel_l_leg,      "vel_R_Leg":    vel_r_leg,
            # "previous": np.array(self.act2),
            # "pos": np.random.normal(self.allPOS, .0523), "vel": np.random.normal(self.allVEL, .01),
            
            # "pos": self.allPOS, "vel": self.allVEL,
            "pos": jPOS, "vel": jVEL,
            # "pos":pvP, "vel": pVV,

            # "eff": np.tanh(self.allEFF),
            "previous": np.array(self.act2),

            # "GLO_BAL": self.ball_XYZ,
            # "LOC_BAL": self.ball_relXYZ,
            # "BALL_VEL": self.ballVel,
            # "ball_ALL": np.array([self.ball_XYZ, self.ballVel]),#, np.array([self.ball_touch_sens])]),
            # "BALL_TOUCH": np.array([self.ball_touch_sens]),

        # ,"effort": jeffort,
        }
        # print (self.stepper, self.ballVel)
        # print ("time",self.stepper,"previous", self.touch_L)# np.array(self.act2))
        # print ("step", self.stepper, "ballFFF", self.ball_touch_sens, "ball stats", self.ballVel)
        # print ("Touch Sens", np.array([self.ball_touch_sens]), (self.bounceCount()[3]), self.bounceCount()[4])
        # print ("JPOS L",  np.array([(LSP_Q), (LEP_Q), (LWP_Q)]))
        # print ("effort",((self.allPOS[:])))
        # print ("act",((self.sim.data.actuator_force[:])))
        return za_states

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.sim.data.ctrl))
        return control_cost

    def vel_cost(self):
        vel_cost = 0.1 * np.sum(abs(self.sim.data.qvel[6:38]))
        return vel_cost

    def ballDistance(self):
        touch, leftClose, rightCLose = False, False, False
        ll, rr = 0, 0
        leftGap = self.ball_XYZ - self.LHand_xyz; rightGap = self.ball_XYZ - self.RHand_xyz
        sq_L_XYZ = np.sqrt(np.sum(np.square(leftGap)))
        sQLZ = np.sqrt(np.sum(np.square(leftGap[2])))
        sqLXY = np.sqrt(np.sum(np.square([leftGap[0], leftGap[1]])))
        sqRXY = np.sqrt(np.sum(np.square([rightGap[0], rightGap[1]])))
        drb_RX = np.sqrt(np.sum(np.square([rightGap[0]])))
        drb_sqLXZ = np.sqrt(np.sum(np.square([leftGap[0], leftGap[2]])))
        spk_sqRYZ = np.sqrt(np.sum(np.square([rightGap[1], rightGap[2]])))
        spk_sqRYYY = np.sqrt(np.sum(np.square([rightGap[1]])))
        srv_XY = sqLXY + sqRXY
        sQLXZ = np.sqrt(np.sum(np.square([leftGap[0], leftGap[2]])))

        sq_R_XYZ = np.sqrt(np.sum(np.square(rightGap)))
        sqRXZ = np.sqrt(np.sum(np.square([rightGap[0], rightGap[2]])))
        sQRZ = np.sqrt(np.sum(np.square(rightGap[2])))
        sumXYZ = sq_L_XYZ+sq_R_XYZ; sumZZZ = sQLZ + sQRZ; sumXZ = sqRXZ + sQLXZ
        # if sum <= 0.3: ll, rr = sq_L_XYZ, sq_R_XYZ
        return sumXYZ, ll, (sumZZZ), (sumXZ), sq_R_XYZ, srv_XY, spk_sqRYZ, sqLXY, drb_sqLXZ, sq_L_XYZ, spk_sqRYYY, drb_RX

    def feetOnGround(self):
        left = self.LAP_xyz[2]; right = self.RAP_xyz[2]
        # print ("left", left, "right", right)
        return 100*((left + right)/2)

    def handsFront(self):
        xLeft, xR, yLeft, yRight = 0,0,0,0
        wX = self.sim.data.body_xpos[1][0]
        lHand = self.LHand_xyz; rHand = self.RHand_xyz
        lf = self.LAP_xyz; rf = self.RAP_xyz
        leftXY = lf[:2] - lHand[:2]; rightXY = rf[:2] - rHand[:2]
        lll = np.sqrt(np.sum(np.square(leftXY)))
        rrr = np.sqrt(np.sum(np.square(rightXY)))
        leftX = wX - lHand[0]; rightX = wX - rHand[0]
        leftHY = lHand[1] - self.LAP_xyz[1]; rightHY = rf[1] - rHand[1]
        if wX > lHand[0]: 
            xLeft = leftX
            # print ("LEFT HAND BEHIND", (xLeft))
        if wX > rHand[0]:
            xR = rightX
            # print ("RIGHT BEHIND", (xR))
        if lHand[1] > lf[1]:
            yLeft =  leftHY
        if rHand[1] < rf[1]:
            yRight =  rightHY
        # else: xLeft, xR, yRight, yLeft = 0, 0, 0, 0
        forLeft = np.sqrt(np.sum(np.square(xLeft + yLeft)))
        forRight = np.sqrt(np.sum(np.square(xR + yRight)))
        # print ("LEFFF OUTSIDE", int(yLeft), "RIGHT OUTSIDE", int(yRight))
        # print ("LEFT BACK", int(xLeft), "RIGHT BACK", int(xR))
        # print ("left", forLeft, "right", forRight)
        return forLeft + forRight

    def waistCurrent(self):
        waistZ = self.sim.data.body_xpos[1][2]
        # return waistZ/.95 if waistZ <=.95 else 1, waistZ
        return (waistZ-.8)/.15 if waistZ <=.95 else 1, waistZ
    def imuCost(self):
        gCost = np.sqrt(np.sum(np.square(self.gyro)))
        aCoustic = np.sqrt(np.sum(np.square(self.accel)))
        # print ("ACC", int(aCoustic), "GYR", int(gCost))
        return aCoustic, gCost, aCoustic + gCost
    def torsoCost(self):
        waist2D = np.sqrt(np.sum(np.square(self.sim.data.body_xpos[1][:2])))
        waist3D = np.sqrt(np.sum(np.square(self.sim.data.body_xpos[1] - [0,0,.95])))
        wQuat = np.sqrt(np.sum(np.square(self.sim.data.body_xquat[1] - [1,0,0,0])))
        hQuat = np.sqrt(np.sum(np.square(self.sim.data.body_xquat[8] - [1,0,0,0])))
        head2D = np.sqrt(np.sum(np.square(self.sim.data.body_xpos[8][:2])))
        head3D = np.sqrt(np.sum(np.square(self.sim.data.body_xpos[8] - [0,0,1.55])))
        return waist3D, wQuat, head3D, hQuat, head2D, waist2D
    def footRotation(self):
        left =  np.sqrt(np.sum(np.square((self.sim.data.body_xquat[36] - [1, 0, 0, 0]))))
        right = np.sqrt(np.sum(np.square((self.sim.data.body_xquat[47] - [1, 0, 0, 0]))))
        # right = np.sqrt(np.sum(np.square((self.sim.data.body_xquat[47] - [0.98628664, 0, -0.16504141, 0]))))
        # return abs(left) + abs(right)
        # print (self.stepper, left, self.sim.data.body_xquat[36])
        return .5* abs(left) + abs(right)
    def ballForce(self):
        left =  np.sqrt(np.sum(np.square(self.touch_L)))
        right = np.sqrt(np.sum(np.square((self.touch_R))))
        return np.tanh((left + right))
    def slowBall(self):
        return np.sqrt(np.sum(np.square(self.ballVel)))

    def waist_bend_cost(self):
        pitch_cost = abs(self.waistRPY[1] / 1.0)
        roll_cost  = abs(self.waistRPY[0] / 1.0)
        yaw_cost  = abs(self.waistRPY[2] / 1.0)
        return pitch_cost + roll_cost + yaw_cost
    def neck_bend_cost(self):
        pitch_cost = abs(self.neck_RPY[1] / 1.0)
        roll_cost  = abs(self.neck_RPY[0] / 1.0)
        yaw_cost  = abs(self.neck_RPY[2] / 1.0)
        return pitch_cost + roll_cost + yaw_cost
    def foot_bend(self):
        bodiesQuatNoFloor = self.sim.data.body_xquat
        lRoll, lPitch, lYaw = self.Lap_RPY[0] ,self.Lap_RPY[1], self.Lap_RPY[2]
        rRoll, rPitch, rYaw = self.Rap_RPY[0] ,self.Rap_RPY[1], self.Rap_RPY[2]
        # print ("L_P", "{:.2f}".format(lPitch), "L_R", "{:.2f}".format(lRoll), "yaw:", "{:.2f}".format(lYaw))
        # print ("R_P", "{:.2f}".format(rPitch), "R_R", "{:.2f}".format(right_ARPY[0]), "yaw:", "{:.2f}".format(right_ARPY[2]))
        feet_pitch_cost = abs(lPitch / 1.0) + abs(rPitch / 1.0)
        feet_roll_cost  = abs(lRoll / 1.0) + abs(rRoll / 1.0)
        # print ("pitch cost", feet_pitch_cost, "roll", feet_roll_cost) 
        # feet_pitch = abs(wRPY[1] / 3.0)
        # feet_roll  = abs(wRPY[0] / 3.0)
        return (feet_pitch_cost + feet_roll_cost)/2

    def bounceCount(self):
        leftReward, rightReward = 0, 0
        BallRightFore, BallLeftFore, BallLeftH, BallRightH, BallChest, self.contactFloor, BallHead, leftHrightH = False, False, False, False, False, False, False, False
        self.touch_L, self.touch_R, self.ball_touch_sens, self.handClap = [0,0,0,0,0,0], [0,0,0,0,0,0], 0, 0
        self.BallTooFar = abs(self.ball_XYZ[1]) > 1 or abs(self.ball_XYZ[0]) > 1 or (self.ball_XYZ[2] > 2)
        self.ballContactIFS = (BallLeftH or BallRightH) or (BallChest and BallLeftH) or (BallChest and BallRightH)        
        # if self.poseCount=='khi':

        # if (self.sim.data.time % 1.5 < .1):
        # if self.number_of_bounces == 70: 
        #     self.mid_reset()
        #     self.number_of_bounces = 0
        # print (self.ball_XYZ)

        for contactIndex in range (self.data.ncon):
            contactID = self.contact[contactIndex]
            if (contactID.geom1 == 23 and contactID.geom2 == 32) or (contactID.geom1 == 32 and contactID.geom2 == 23): leftHrightH = True
            if (contactID.geom1 == 23 and contactID.geom2 == 73) or (contactID.geom1 == 73 and contactID.geom2 == 23): BallLeftH = True
            if (contactID.geom1 == 21 and contactID.geom2 == 73) or (contactID.geom1 == 73 and contactID.geom2 == 21): BallLeftFore = True
            if (contactID.geom1 == 32 and contactID.geom2 == 73) or (contactID.geom1 == 73 and contactID.geom2 == 32): BallRightH = True
            if (contactID.geom1 == 30 and contactID.geom2 == 73) or (contactID.geom1 == 73 and contactID.geom2 == 30): BallRightFore = True
            if (contactID.geom1 == 8 and contactID.geom2 == 73) or (contactID.geom1 == 73 and contactID.geom2 == 8): BallChest = True
            if (contactID.geom1 == 14 and contactID.geom2 == 73) or (contactID.geom1 == 73 and contactID.geom2 == 14): BallHead = True             
            if (contactID.geom1 == 0 and contactID.geom2 == 73) or (contactID.geom1 == 73 and contactID.geom2 == 0):
                self.number_of_bounces += 1; self.contactFloor = True
                # print ("TOTAL Bounces = ", self.number_of_bounces)
                # for i in  range((self.sim.data.time)):
            # if (contactID.geom1 == 23) and contactID.geom2 == 73:
        # if BallChest: print ("ep:",self.epNo, self.stepper, "BALL CHEST")
        # if BallHead: print ("ep:",self.epNo, self.stepper, "Ball HEAD")
        # if BallRightH: print ("ep:",self.epNo, self.stepper, "Ball Right Hand")
        # if BallLeftH: print ("ep:",self.epNo, self.stepper, "Ball Left Hand")

        if self.poseCount=='srv': 
            self.ballContactIFS = (BallLeftH or BallRightH) or (BallLeftFore or BallRightFore)
            self.BallTooFar = (abs(self.ball_relXYZ[1]) > 1 ) or (self.ball_relXYZ[0] < -1)
        elif self.poseCount=='spk':
            self.ballContactIFS = BallRightH
            self.BallTooFar = (abs(self.ball_relXYZ[1]) > 1 ) or (self.ball_relXYZ[0] < -1)
        elif self.poseCount=='drb':
            self.ballContactIFS = BallLeftH #or BallRightH
            self.BallTooFar = abs(self.ball_relXYZ[1]) > 1 or abs(self.ball_relXYZ[0]) > 1
        elif self.poseCount=='hold' or 'catch':
            self.ballContactIFS = (BallLeftH and BallRightH) or (BallChest and BallLeftH) or (BallChest and BallRightH) or (BallChest and BallRightFore) or (BallChest and BallLeftFore) or (BallLeftFore and BallRightFore) or (BallLeftFore and BallRightH) or (BallRightFore and BallLeftH)
            if BallLeftH or BallRightH or BallLeftFore or BallRightFore:
                self.BallTouching = 0.5
            # self.ballContactIFS = (BallLeftH or BallRightH or BallChest)
            self.BallTooFar = abs(self.ball_relXYZ[1]) > 1 or abs(self.ball_relXYZ[0]) > 1
        # print (self.ballContactIFS)        
        if self.ballContactIFS:
            self.no_oneHand_touches +=1
            self.BallTouching = 1
            self.ball_touch_sens = (self.sim.data.sensordata[30])
        if self.BallTooFar:
            # self.mid_reset()
            # self.ballStuck()
            self.number_of_bounces = 0

        if (leftHrightH):
            # print ("ep:", self.epNo, self.stepper, "hands TOGETHER")
            self.handClap = 1
        if BallLeftH or BallRightH:
            self.ball_touch_sens = (self.sim.data.sensordata[30])
            # self.ball_touch_sens = (1*self.ball_touch_sens)
            # self.ball_touch_sens = min(2, .01*self.ball_touch_sens)
            self.touch_L, self.touch_R = self.sim.data.sensordata[18:24], self.sim.data.sensordata[24:30]
            self.leftFINGER = np.count_nonzero([self.touch_L]); self.rightFINGER = np.count_nonzero([self.touch_R])
            # print ("step", self.stepper, "BALL TOUCH", '%.2f' %self.ball_touch_sens, "L fingers", self.leftFINGER, "right FING", self.rightFINGER)
            # print ("step", self.stepper, "BALL TOUCH", '%.2f' %self.ball_touch_sens, "L fingers", sum(self.touch_L), "right FING", sum(self.touch_R), "Chest", '%.2f' %self.sim.data.sensordata[31])
            # print ("ep:",self.epNo,"step",self.stepper, "BALL TOUCH", '%.2f' %self.ball_touch_sens, "L fingers", self.leftFINGER, "right FING", self.rightFINGER, "Chest", '%.2f' %self.sim.data.sensordata[31])
            # self.touch_L = (self.sim.data.cfrc_ext[17])
            # self.touch_R = (self.sim.data.cfrc_ext[23])

        return self.number_of_bounces, leftReward, rightReward, self.touch_L, self.touch_R, self.handClap

    @property
    def healthy_reward(self):
        return ( float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward )
    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost
    @property
    def stillAlive(self):
        min_z, max_z = self._healthy_z_range
        safeBend = self.heathy_angle; maxBounces = 0
        overTilting, drowning = False, False
        nflDown = False; waistLOW = False; wallDown = False
        tooMuchBounce, ballOut = False, False
        if -safeBend < self.waistRPY[0] < safeBend and (-safeBend < self.waistRPY[1] < safeBend) and -safeBend < self.neck_RPY[0] < safeBend and (-safeBend < self.neck_RPY[1] < safeBend): overTilting = False
        else: overTilting = True
        
        if min_z < self.sim.data.body_xpos[1][2] < max_z: waistLOW = False
        else: waistLOW = True
        floorcont, wallCont, bounceCounter = 0, 0, 0

        for contactIndex in range (self.data.ncon):
            contactID = self.contact[contactIndex]
            # print (dir(contactID))
            # print ("Contact", contactIndex,":", (contactID.geom1 ), "and", contactID.geom2 )
            if contactID.geom1 == 0:
                floorcont +=1
                # print ("floor cont", floorcont)
                # print ("floor and ", contactID.geom2)# and contactID.geom2 > 50: nflDown=False
                if floorcont <=2:#nflDown = False
                    if contactID.geom2 == 52 or contactID.geom2 == 72 or contactID.geom2 == 73:
                        nflDown = False
                        # below was previously elif 6 < (changed on o.i thursday)
                    elif 5 < contactID.geom2 < 52 or 52 < contactID.geom2 <72:
                        nflDown=True
                    else: nflDown = True

            if 0 < contactID.geom1 <= 5:
                wallCont +=1
                # print ("WALL ", contactID.geom1, "and ", contactID.geom2)# and contactID.geom2 > 50: nflDown=False
                if wallCont <=1: #nflDown = False
                    if contactID.geom2 == 73: wallDown = False
                    elif contactID.geom2 < 73: wallDown = True
                else: wallDown = True
        
            if contactID.geom1 == 0 and contactID.geom2 == 73:
                bounceCounter +=1
                if bounceCounter > maxBounces: tooMuchBounce = True

            if 1.4 < self.sim.data.body_xpos[8][2] < 1.65: drowning = False
            else: drowning = True
        
        if self.BallTooFar: ballOut = True

        height_tilt = overTilting==False and waistLOW==False
        DroppedBall = tooMuchBounce==False
        LostBall = ballOut==False
        nfl = nflDown == False
        # is_healthy = nflDown==False #and wallDown==False #and self.bounceCount()[0] < 60 #and overTilting==False
        return height_tilt, DroppedBall, LostBall, nfl
    @property
    def done(self):
        done = (not self.stillAlive) if self._terminate_when_unhealthy else False
        # done = True if self.is_healthy <= 0 else False
        return done

class khi_class(globalClass):
    def __init__(self):  self.global_init()
    def step(self, action):
        new =[]; limit = 0.261799 #0.785398 #45 deg
        touches_B4, handReward, self.ballContact, ballGapAfter, ballGapB4, ballPot, XZpot, ballZZZPot, ballBouncedCost, newEpisode, jPos_1, jPos_2, vel1, jVel_2 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
        fallen = 1; self.BallTouching, ballPositionB4, ballPositionAFTER =0,[0,0,0],[0,0,0]
        ballUPUP = 0
        # self.touchesReward = 0
        for i in range(len(action)):
            # if abs(actDiff[i]) > 0.785398: 
#FULL LOWER KHI
            if self.poseCount=='khi':
                # action[1] = action[6]
                # action[2] = -action[7]
                # action[3] = -action[8]
                # action[4] = action[9]
                # action[5] = action[10]
                
                action[11] = action[17] = 0
                action[12] = action[18] = 0
                # action[13] = action[19] = -.2967
                # action[14] = action[20] = .6283
                action[15] = action[21] = 0
                action[16] = action[22] = 0

# CATCH A RIIIIIIIDE CR CP AR
            # if self.poseCount=='catch' or 'hold' or 'khi':
            #     action[6] =   action[1] #SHD P
            #     action[7] =  -action[2] #SHD R
            #     action[8] =  -action[3] #SY
            #     action[9] =   action[4] #EP
            #     action[10] =   action[5] #WP
                # action[14] = -action[11] #CY
                # action[16] = action[13] #CP
                # action[15] = -action[12] #CR
# DRB CR CP
            # if self.poseCount=='drb':
            #     action[13] = action[10] #LEP REP with CR_CP_KP 
#SRV SP EP CR CP
            if self.poseCount=='srv':
                action[5] = action[1] #SP
                action[6] = -action[2] #SY
                action[7] = action[3] #EP
                action[8] = action[4] #WP
#SPK SP EP CR CP
            # if self.poseCount=='spk':
            #     action[15] = action[12] #CP_CP

            list1 = np.clip(action[i], self.act2[i] - limit, self.act2[i] + limit)
            # list1 = np.clip(action[i], self.act2[i]-(1+limit2), self.act2[i]+(1+limit2))
            # if actDiff[i] <  0.785398: self.clipped = np.clip(action[i], self.act2[i] - 0.785398, action[i])
            new.append(list1)
            # self.act2.append(self.clipped)
        # if self.stepper< 2: self.act2 = self.act2 #WAIT X STEPS
        # else: 
        self.act2 = new
        # actDiff = (action - self.act2)
        # print ("time:", self.stepper, "raww", actDiff)
        # print ("time:", self.stepper, "act2", self.act2)
        # print ("DIFFERENCE", actDiff)
        actiondeg = np.rad2deg(self.act2) #; actionrad = np.deg2rad(action)
        # actiondeg = np.rad2deg(action)
        # print("action:", action, "in deg:", actiondeg)
        jPOS_B4 = self.sim.data.qpos.flat[7:37]
        # jPOS_B4 = self.sim.data.qpos.flat[7:37] ###VEL COST ONLY FOR ACTIVE JOINTS ##########
        waistB4 = self.waistCurrent()[1]
        wTilt1 = self.torsoCost()[1]
        head_Quat_b4 = self.torsoCost()[3]
        waist_XYZ_b4 = self.torsoCost()[0]
        waist_XY_b4 = self.torsoCost()[5]
        hCart1, head2D1 = self.torsoCost()[2],self.torsoCost()[4]

        if self.stepper >0:
            ballGapB4 = self.ballDistance()[0]
            ballXZB4 = self.ballDistance()[3]
            vel1 = self.allVEL
            self.HeadQuatVelb4 = self.HeadQuatVel
            jPos_1 = self.allPOS
            touches_B4 = self.no_oneHand_touches
            ballVelB4 = self.ballVel
            ballPositionB4 = self.ball_relXYZ

        # if self.stepper % 1 == 0:
        # if self.epNo==1 and self.stepper ==0:
        #     print("begin begin", self.epNo, "over. Enter 'pique' to continue")
        #     fg = input()
        #     if fg=="k": None        
        self.do_simulation(actiondeg, self.frame_skip)
        # self.sim.step()
        stepObs = self.global_get_obs()
        ######       AFTER OBS #######
        bodyFall, ballDrop, ballLost, nflFall = not self.stillAlive[0], not self.stillAlive[1], not self.stillAlive[2], not self.stillAlive[3]
        touchesAfter = self.no_oneHand_touches
        ballVelAFTER = self.ballVel
        ballPositionAFTER = self.ball_relXYZ

        if self.sim.data.cfrc_ext[48][:5].all() != 0: self.BallBeenTouched +=1
        if self.poseCount =='khi':  self.terminate=nflFall

        elif self.poseCount =='hold':
            self.terminate=bodyFall+ballDrop
            if (self.BallBeenTouched==0) and self.stepper <200 :
                self.sim.data.xfrc_applied[48] = [0,0,6.083,0,0,0]
            else: self.sim.data.xfrc_applied[48] = [0,0,0,0,0,0]
            if self.number_of_bounces>0: ballBouncedCost = 1
            if self.BallTouching==0:# and self.stepper <100 :
                handReward = self.ballDistance()[0] #L_R_XYZ

        elif self.poseCount =='catch':
            self.terminate=bodyFall+ballDrop#+ballLost
            if self.stepper<10: self.sim.data.xfrc_applied[48] = [-13,0,20,0,0,0]
            else: self.sim.data.xfrc_applied[48] = [0,0,0,0,0,0]
            if self.number_of_bounces>0: ballBouncedCost = 1  
            # if self.no_oneHand_touches <1 :
            if (self.BallBeenTouched>0) or self.BallTouching==0:# and self.stepper <100 :
                handReward = self.ballDistance()[0] #* self.stepper/50 #L_R_XYZ
            # ballUPUP = min(1, self.ball_relXYZ[2]) #BALL Z BEFORE HIT
            # print ("catch", handReward)

        elif self.poseCount =='drb':
            self.terminate=bodyFall+ballLost
            if (self.sim.data.sensordata[30]==0 and self.stepper<3):
                self.sim.data.xfrc_applied[48] = [0,0,90,0,0,0]
            else: self.sim.data.xfrc_applied[48] = [0,0,0,0,0,0]
            if (self.BallTouching==0 and self.ballVel[2]>0):# and self.stepper <100 :
                # handReward = self.ballDistance()[7]     #LEFT_XY
                # handReward = self.ballDistance()[7] + self.ballDistance()[4]     #LEFT_XY AND R_XYZ
                handReward = self.ballDistance()[9]     #L_XYZ
                # handReward = self.ballDistance()[9] + (self.ballDistance()[11] )#*self.Curriculum)    #LXYZ_RX
                # handReward = self.ballDistance()[9] + self.ballDistance()[4]    #LEFT_XYZ_AND_R_XYZ
            if self.no_oneHand_touches<1:
                if self.number_of_bounces>0:
                    ballBouncedCost = 1
                if self.terminate:
                    ballBouncedCost = 10
            if self.no_oneHand_touches >1:
                # if self.ballVel[2]==0:
                if (ballPositionB4[2] == ballPositionAFTER[2]):
                    self.terminate = True
                    ballUPUP = -1
                # ballUPUP = np.tanh(.5* self.ballVel[2]) if (self.ballVel[2]>0 and not self.BallTooFar) else 0
            # if self.BallTooFar: ballBouncedCost = 1

        elif self.poseCount =='srv':
            self.terminate=bodyFall +ballDrop #+ballLost
            if self.stepper<5: self.sim.data.xfrc_applied[48] = [-15,0,8,0,0,0]
            # if self.stepper<5: self.sim.data.xfrc_applied[48] = [-6,0,4,0,0,0]
            else: self.sim.data.xfrc_applied[48] = [0,0,0,0,0,0]
            if self.no_oneHand_touches<1:
                handReward = self.ballDistance()[0] #L_R_XYZ
                if self.number_of_bounces>0:
                    self.terminate = True
                    ballBouncedCost = 1
            if self.no_oneHand_touches >1:
                if (ballPositionB4[2] < ballPositionAFTER[2]):
                    ballUPUP = min(3, self.ball_relXYZ[2])
                if self.contactFloor:
                    ballBouncedCost = -2
            # if not self.BallTooFar and self.stepper<=50:
            #     handReward = self.ballDistance()[5] * self.stepper/50 #L_R_XY
            # else: handReward = self.ballDistance()[5]

        elif self.poseCount =='spk':
            self.terminate=bodyFall#+ballDrop #+ballLost
            if self.no_oneHand_touches<1:
                handReward = self.ballDistance()[4] #RIGHT_XYZ
                # handReward = (self.ballDistance()[6]) #RIGHT_YZ
                # handReward = (self.ballDistance()[10]) #RIGHT_YYYY
                if self.contactFloor:
                    self.terminate=True
                    ballBouncedCost = 10
                if (ballPositionB4[2] < ballPositionAFTER[2]):
                    ballUPUP = min(1, self.ball_relXYZ[2]) #BALL Z BEFORE HIT
            if self.no_oneHand_touches>0:
                ballUPUP = min(3, self.ball_relXYZ[0]) #BALL X AFTER HIT
                if self.contactFloor:# and (ballPositionB4[0] - ballPositionAFTER[0]) <0.01:
                    # print ("DONE", self.ball_relXYZ[0],"metres")
                    ballBouncedCost = -2
                    self.terminate=True

        headXY_2 = self.torsoCost()[4]
        waist2D2 = self.torsoCost()[5]
        head_3D, headQUAT_2 = self.torsoCost()[2], self.torsoCost()[3]
        wCart2, wQuat2 = self.torsoCost()[0], self.torsoCost()[1]
        waistAft = self.waistCurrent()[1]
        jVel_2 = self.allVEL; jPos_2 = self.allPOS
        jPOS_AFTER = self.sim.data.qpos.flat[7:37] #size is now 30
        if self.stepper >0: 
            ballXYZGapAfter = self.ballDistance()[0]
            ballXZAfter = self.ballDistance()[3]; ballZZAfter = self.ballDistance()[2]
            ballPot = (ballXYZGapAfter - ballGapB4)
            if self.ballForce() ==0: 
                XZpot = (ballXZAfter - ballXZB4)
                ballZZZPot = (ballZZAfter - ballXZB4)
            # if ballPot > 0: ballPot = ballPot
            # else: ballPot=0
            # print ("ball Dropping", 1 * (ballXZAfter), "ballXYZ POT", ballPot)
        if self.waistCurrent()[1] < .95:
            wZZZPot = 100 * (waistAft - waistB4)
        else: wZZZPot = 0
        self.HeadQuatVel =-100* (headQUAT_2 - head_Quat_b4)
        wTiltPot =-100* (wQuat2 - wTilt1)
        waist3DPot = -100* (wCart2 - waist_XYZ_b4)
        head3DPot = -100* (head_3D - hCart1)
        head2DPot = -100* (headXY_2 - head2D1)
        waist2DPot = -100* (waist2D2 - waist_XY_b4)
        HeadQuatAcc = (self.HeadQuatVel - self.HeadQuatVelb4)
        # ballTouchSense = .01 * self.ball_touch_sens
        # if self.ball_touch_sens >0:
        #     self.ballContact = min(2, .01 * self.ball_touch_sens)
        if touches_B4 < touchesAfter: self.touchesReward += 1
        elif touches_B4==touchesAfter: self.touchesReward =0
        jointAcceleration = -.1 * np.sqrt(np.sum(np.square(jVel_2-vel1)))
        j_vel_cost = sum(abs(jPos_2 - jPos_1) / self.dt)
        # print (self.stepper, "joint vel", j_vel_cost)
        if self.longStepper > 5_00_000: suddenly =1
        else: suddenly=1
        if self.longStepper > 1_000_000: suddenly2 =1
        else: suddenly2=0
        # print ("episode", self.epNo, "ep LENGTH", self.stepper, self.ball_touch_sens)
        if self.terminate: 
            # print ("ep", self.epNo, "length", self.stepper)
            fallen = 2; newEpisode = 1
            # print ("episode", self.epNo, "ep LENGTH", self.stepper, self.ball_touch_sens)
            # f = open('/home/admin/Downloads/length.csv','a') # Prints Action Values
            # # for s in goodz:
            # f.write(str(self.stepper))
            # f.write(" ")
            # f.write("\n")
            # f.close() 
        if self.stepper< 100: epReward = self.stepper/100
        else: epReward = 1
        goodz = [
                # self.waistCurrent()[0]
                epReward,  #S_100
                 -1*headQUAT_2 #* self.Curriculum
                ,  -1*(self.footRotation()) #* self.Curriculum
                ,-1 * np.tanh(handReward) #* suddenly
                # ,-1 * np.tanh(handReward) * self.Curriculum
                # , wZZZPot
                # , self.HeadQuatVel
                # , HeadQuatAcc
                # , wTiltPot
                # , waist2DPot
                # , waist3DPot
                # , head3DPot
                # , head2DPot
                # , 1 * self.ballDistance()[0]
                # , ballPot *suddenly #* self.Curriculum
                # ,-1* ballZZZPot
                # ,( self.bounceCount()[1] + self.bounceCount()[2])
                # forward_reward = self._forward_reward_weight * x_velocity
                # , 1 * min(2,.01*ballTouch) #*suddenly #* self.Curriculum
               , 1 * self.BallTouching #* suddenly
            #    , 1 * self.BallTouching * self.Curriculum
                # , 1 * min(1, self.touchesReward / 5) * self.Curriculum
                , -1 * min(1, self.touchesReward / 5) # * suddenly
                # , -1 *ballBouncedCost * self.Curriculum
                , -1 *ballBouncedCost #* suddenly
                # ,epReward * (self.ballForce()) #*suddenly #* self.Curriculum
                , ballUPUP
                # , -1* wQuat2
                # , -1 * head_3D #head3D

                , 1 *fallen
                ]
        # badz = [
                # j_vel_cost * .01 * self.Curriculum,
                # self.torsoCost()[0], #waist3D
                # self.torsoCost()[1] #* self.Curriculum, #waist Quat
                # .01 * self.imuCost(),
                # (1 * self.feetOnGround()),
                # self.neck_bend_cost(),
                # (1 * self.waist_bend_cost()),
                # self.foot_bend() * self.Curriculum,
                # self.handsFront() * self.Curriculum,
                # (10*self.slowBall()),
                # ]

        self.reward = sum(goodz) #- sum(badz)
        info = {
            # "reward_alive":  self.healthy_reward,
            # "reward_linvel": forward_reward,
            # "reward_quadctrl": -ctrl_cost,
            # "reward_impact": -contact_cost,
            # "x_position": xy_position_after[0],
            # "y_position": xy_position_after[1],
            # "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            # "x_velocity": x_velocity,
            # "y_velocity": y_velocity,
            # "forward_reward": forward_reward,
            "WAIST DROP:":  ('%.2f' %(self.waistCurrent()[0] )),
            "BALL POT: ":   ('%.2f' %(ballPot)),
            "FEET RAISED":  ('%.2f' %self.feetOnGround()),
            "J VEL":        ('%.2f' %j_vel_cost),
            "WAISTBEND":    ('%.2f' %self.waist_bend_cost()),
            # "NECKBEND":     ('%.2f' %self.neck_bend_cost()),
            "FOOTBEND":     ('%.2f' %self.foot_bend()),
            "BALL DISTANCE":    ('%.2f' %self.ballDistance()[0]),
            "BALL TOUCH L":     ('%.2f' %self.bounceCount()[1]),
            "BALL TOUCH R":     ('%.2f' %self.bounceCount()[2]),
            }
        # print (self.stepper, goodz)
        # f = open('/home/admin/Downloads/plother.txt','a') # Prints Action Values
        # for s in goodz:
        #     f.write(str(s))
        #     f.write(" ")
        # f.write("\n")
        # f.close()
        
       
        # print ("headTILT", '%.2f' %headQUAT_2, "TiltPOT", '%.2f' %self.HeadQuatVel, "3DPOT", '%.2f' %head3DPot)
        # print ("headCART", '%.2f' %head_3D)#, "TiltPOT", '%.2f' %self.HeadQuatVel, "3DPOT", '%.2f' %head3DPot)
        # print ("step", self.stepper, "wQuat", '%.2f' %(wTiltPot), "hQuat", '%.2f' %self.HeadQuatVel)
        # print ("step", self.stepper, "waist3D", '%.2f' %(waist3DPot),"wZZZ", '%.2f' %(wZZZPot),"head3D", '%.2f' %head3DPot)
        # print ("step", self.stepper, "waist2D", '%.2f' %(waist2DPot),"head 2D", '%.2f' %(head2DPot))
        # print ("headQ", self.torsoCost()[3])
        # print ("ep", self.epNo, "REWARD:", '%.2f' %self.reward, "good", goodz, "CURRIC", '%.5f' %self.Curriculum)
        # print ("ballFRC", self.sim.data.xfrc_applied[48])
        if self.longStepper > 5_00_000:
            if self.currCounter < (self.difficultyLength): self.currCounter +=1
        self.stepper += 1; self.longStepper += 1
        # print ("episode", self.epNo, "curric", self.Curriculum, "bouncedCost", ballBouncedCost)
        # if self.terminate:
        #     print("episode", self.epNo, "over. Enter 'p' to continue")
        #     fg = input()
        #     if fg=="p": None
        return stepObs, self.reward, self.terminate, info

    def reset_model(self): 
        self.number_of_bounces, self.no_oneHand_touches, self.touchesReward, self.BallBeenTouched, self.BallTouching = 0,0, 0, 0, 0
        # if self.poseEps <= 9:       self.poseCount = 1
        # if 9 < self.poseEps < 19:   self.poseCount = 2
        # if self.poseEps ==20: self.poseEps =0
        # qpos #+ self.np_random.uniform( low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel #+ self.np_random.uniform( low=noise_low, high=noise_high, size=self.model.nv)
        if self.poseCount =='khi':
            self.set_state(np.array(self.KHI_pose_fixed), qvel)
            # self.act2 = self.act_KHI
            # print ("episode", self.epNo, "KHI KHI KHI")
        if self.poseCount =='hold':
            randdd = self.hold_pose[:37]
            randdd.extend([randy(self, 0.3, 0.6), 0, randy(self, 1.1, 1.4), 0,0,0,0])
            if self.longStepper >4_000_000:
                self.set_state(np.array(randdd), qvel)
            else: self.set_state(np.array(self.hold_pose), qvel)            
            # self.act2 = self.act_hold
            # print ("episode", self.epNo, "HOLD POSE")
        if self.poseCount == 'drb':
            randdd = self.dribble_pose[:37]
            randdd.extend([randy(self, 0.2, 0.6), randy(self, .2, 0.4), randy(self, 0.15, 0.3), 0,0,0,0])
            if self.longStepper >500_000:
                self.set_state(np.array(randdd), qvel)
            else: self.set_state(np.array(self.dribble_pose), qvel)
            # self.act2 = self.act_drb
            # print ("episode", self.epNo, "Dribble Pose")
        if self.poseCount == 'catch':
            self.set_state(np.array(self.catch_pose), qvel)
            # self.act2 = self.act_catch
            # print ("episode", self.epNo, "i can catch")
        if self.poseCount == 'srv':
            self.set_state(np.array(self.srv_pose), qvel)
            # self.act2 = self.act_srv
        if self.poseCount == 'spk':
            self.set_state(np.array(self.spk_pose), qvel)
            # self.act2 = self.act_SPK            
            # print ("episode", self.epNo, "Volleyball")
        reset_obs = self.global_get_obs()
        self.epNo += 1 #; self.poseEps += 1
        self.Curriculum = self.currCounter / self.difficultyLength 
        # print ("ep", self.epNo, "rew", ('%.2f' %self.reward), "length:", self.stepper, "timestep", self.longStepper, "curric", self.Curriculum)
        self.stepper = 0
        return reset_obs
        # self.global_reset()
