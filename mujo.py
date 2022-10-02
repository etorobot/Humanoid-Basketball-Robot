# pip3 install -U 'mujoco-py<2.2,>=2.1'
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/admin/.mujoco/mujoco200/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
from simmod.wrappers import UDRMujocoWrapper, ADRMujocoWrapper
from simmod.modification.mujoco import MujocoJointModifier, MujocoOptionModifier, MujocoBodyModifier
from simmod.algorithms import UniformDomainRandomization
from simmod import load_yaml

import mujoco_py, gym, os, muj_envs_folder
import numpy as np, math, json
mj_path = mujoco_py.utils.discover_mujoco()
model = mujoco_py.load_model_from_path('/home/admin/mujokaleido/KSTL.xml')
# model = mujoco_py.load_model_from_path(xml)
# xml = os.path.join(mj_path, 'model', 'humanoid100.xml')
# xml=os.path.join('/home/admin/mujokaleido', 'KSTL.xml'),

sim    = mujoco_py.MjSim(model)
# mujoco_py.cymj.set_pid_control(sim.model, sim.data)
# viewer = mujoco_py.MjViewer(sim)
sim_state = sim.get_state()

t = 0
# while True:
#     # MjModel.joint_name2id returns the index of a joint in
#     # MjData.qpos
#     L_CP = sim.model.get_joint_qpos_addr("L_CROTCH_P")
#     L_KP = sim.model.get_joint_qpos_addr("L_KNEE_P")
#     sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
#     sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
#     t += 1
#     x, y = math.cos(t), math.sin(-.5*t)
#     viewer.add_marker( pos=np.array([x, y, 1]), label=str(t) )
#     for i in range(400):
#         sim.set_state(sim_state)
#         if i < 50: sim.data.ctrl[:] = 0
#         if i < 150: sim.data.ctrl[:] = 2
#         elif i < 250: sim.data.ctrl[:] = 0.5
#         elif i < 350: sim.data.ctrl[:] = -1
#         else: sim.data.ctrl[:] = -.5    

#         sim.forward()
#         sim.step()
#         # sim.render()
#         viewer.render()
#     if t > 10 and os.getenv('TESTING') is not None:
#         break    

# env = gym.make('KaleidoBed-v0')
# env = gym.make('gym.envs.mujoco:HumanoidStandup-v2')
# env = gym.make('KHumanoid-v0')
env = gym.make('KaleidoKHI-v0')
# tex_mod = MujocoTextureModifier(sim=env.sim)
# mat_mod = MujocoMaterialModifier(sim=env.sim)
# env = UDRMujocoWrapper(env, mat_mod)

observation = env.reset()
info = env.reset()
act_dim = env.action_space.shape[0]
obs_dim = env.observation_space

print("Observation space:", (obs_dim))
# print ("type", type(obs_dim))
print("Action space:", env.action_space)
env.reset()
yamlJoint = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_joints.yaml'))
yamlBodies = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies.yaml'))
yamlOption = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_option.yaml'))
yamlBodiesFriction = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_friction.yaml'))
yamlBodiesMass = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_mass.yaml'))

modify_bodiez_friction = MujocoBodyModifier(sim=env.sim, config = yamlBodiesFriction)
modify_bodiez_mass = MujocoBodyModifier(sim=env.sim, config = yamlBodiesMass)
modify_bodiez = MujocoBodyModifier(sim=env.sim, config = yamlBodies)
modify_joints = MujocoJointModifier(sim=env.sim, config = yamlJoint)
modify_option = MujocoOptionModifier(sim=env.sim, config = yamlOption)
# BODYID = modify_body.model.body_name2id('BODY')
# env = ADRMujocoWrapper(env, None, None, None), , modify_joints, modify_option, modify_bodiez_friction, modify_bodiez_mass, None)
env = UDRMujocoWrapper(env, modify_bodiez_friction, modify_bodiez_mass, modify_joints)
d = env.unwrapped.data
i = 3
# for i in range(4000):
while True:
    # doctor_ALGO = UniformDomainRandomization(modify_body)
    # sim.step() 
    # viewer.render()
    action = np.random.randn(act_dim)
    # action = action.reshape(1, -1)
    observation, reward, done, info = env.step(action)

    # tensorboard --logdir='./tensorB'
    # ./simulate ~/mujokaleido/KSTL.xml 
    # for coni in range(d.ncon):
    #     print('  Contact %d:' % (coni,))
        # con = d.contact[coni] 
        # print('    dist     = %0.3f' % (con.dist,))
        # # print('    pos      = %s' % (str_mj_arr(con.pos),))
        # # print('    frame    = %s' % (str_mj_arr(con.frame),))
        # # print('    friction = %s' % (str_mj_arr(con.friction),))
        # print('    dim      = %d' % (con.dim,))
        # print('    geom1    = %d' % (con.geom1,))
        # print('    geom2    = %d' % (con.geom2,))
    # observation, reward, done, info = env.step(0); env.render()
    # observation, reward, done, info = env.step(0); env.render()
    # observation, reward, done, info = env.step(0); env.render()
    # observation, reward, done, info = env.step(0); env.render()
    # observation, reward, done, info = env.step(-2); env.render() 
    # observation, reward, done, info = env.step(-2); env.render()
    # observation, reward, done, info = env.step(-2); env.render()
    # observation, reward, done, info = env.step(-2); env.render()    

    # observation, reward, done, info = env.step(-1); env.render()
    # observation, reward, done, info = env.step(-1); env.render()
    # observation, reward, done, info = env.step(-1); env.render()
    # observation, reward, done, info = env.step(-1); env.render()


   
    # print("action:", action,"deg:", np.rad2deg(action))
    # print(sim.data.qvel )
    # modify_body.model.body_mass[3]
    # print ("le mass", model.body_mass)
    noise = np.clip(np.random.normal(0, 0.00), -.009, .009)
    # print (noise)
    # print ("TOTAL KG:", (env.sim.model.body_mass[1]))
    # print ("gravity,", env.sim.model.opt.gravity)
    # print ("row,", env.sim.model.opt.viscosity)
    # viewer.add_marker( pos=np.array([x, y, 1]), label=str(t) )    
    env.render()
    # env.step(action)
    if done:
            observation = env.reset()
            info = env.reset()
    # env.reset()
    # doctor_ALGO.step()

env.close()

#python3 -m tensorboard.main --logdir='./tensorB'


