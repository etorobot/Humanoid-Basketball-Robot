import gym, os
from stable_baselines3 import SAC
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
from simmod.wrappers import UDRMujocoWrapper, ADRMujocoWrapper
from simmod.modification.mujoco import MujocoJointModifier, MujocoOptionModifier, MujocoBodyModifier
from simmod.algorithms import UniformDomainRandomization
from simmod import load_yaml
yamlJoint = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_joints.yaml'))
yamlBodies = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies.yaml'))
yamlOption = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_option.yaml'))
yamlBodiesFriction = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_friction.yaml'))
yamlBodiesMass = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_mass.yaml'))

# Create log dir
log_dir = "just_xyz/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make("muj_envs_folder:KaleidoKHI-v0")
modify_bodiez_friction = MujocoBodyModifier(sim=env.sim, config = yamlBodiesFriction)
modify_bodiez_mass = MujocoBodyModifier(sim=env.sim, config = yamlBodiesMass)
modify_bodiez = MujocoBodyModifier(sim=env.sim, config = yamlBodies)
modify_joints = MujocoJointModifier(sim=env.sim, config = yamlJoint)
modify_option = MujocoOptionModifier(sim=env.sim, config = yamlOption)
# env = UDRMujocoWrapper(env, 
#                         modify_bodiez_friction,
#                         modify_bodiez_mass,
#                         modify_joints)

# model = SAC("MultiInputPolicy", env, learning_starts=10)

# model = SAC.load("/home/admin/mujokaleido/mon2_drb_life1_CY_LSP/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/monday3_drb_life1_f10_CYLSP_2h_500Hz/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/monday_drb_life1_f10_CYLSP/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/fr11_SPK_CURR_500k_live0_F2_fixed/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/mondayTWO_drb_life1_f10_CYLSP_2h_500Hz/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/mon4B_SPK_life1_f10/best_model.zip")
model = SAC.load("/home/admin/mujokaleido/fr1_STAND_500M/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/fr1_HOLD_1Msudden_F2/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/TH3_HOLD_1Msudden_F2/best_model.zip")
# model = SAC.load("/home/admin/mujokaleido/TH3_HOLD_1Msudden_F2/best_model.zip")



# mean_reward, std_reward = evaluate_policy(model, env)k
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# for i in range(1000):
# timesteps = 3_000_000
# plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "K__Stand")
# plt.show()
obs = env.reset()
while True:    
    action, states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done: obs = env.reset()

# env.close()