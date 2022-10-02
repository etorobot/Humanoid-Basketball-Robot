import gym, os, numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from simmod.wrappers import UDRMujocoWrapper
from simmod.modification.mujoco import MujocoJointModifier, MujocoOptionModifier, MujocoBodyModifier
from simmod import load_yaml
yamlJoint = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_joints.yaml'))
yamlBodies = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies.yaml'))
yamlOption = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_option.yaml'))
yamlBodiesFriction = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_friction.yaml'))
yamlBodiesMass = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_mass.yaml'))

# Create log dir
log_dir = "venEnv/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
KHI_env = "muj_envs_folder:KaleidoKHI-v0"
env = gym.make('gym.envs.mujoco:HumanoidStandup-v2');env = Monitor(env, log_dir)

# modify_bodiez_friction = MujocoBodyModifier(sim=env.sim, config = yamlBodiesFriction)
# modify_bodiez_mass = MujocoBodyModifier(sim=env.sim, config = yamlBodiesMass)
# modify_bodiez = MujocoBodyModifier(sim=env.sim, config = yamlBodies)
# modify_joints = MujocoJointModifier(sim=env.sim, config = yamlJoint)
# modify_option = MujocoOptionModifier(sim=env.sim, config = yamlOption)

# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq:
#     :param log_dir: Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: Verbosity level.
#     """
#     def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.save_path = os.path.join(log_dir, 'best_model')
#         self.best_mean_reward = -np.inf

#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:

#           # Retrieve training reward
#           x, y = ts2xy(load_results(self.log_dir), 'timesteps')
#           if len(x) > 0:
#               # Mean training reward over the last 100 episodes
#               mean_reward = np.mean(y[-100:])
#               if self.verbose > 0:
#                 print(f"Num timesteps: {self.num_timesteps}")
#                 print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

#               # New best model, you could save the agent here
#               if mean_reward > self.best_mean_reward:
#                   self.best_mean_reward = mean_reward
#                   # Example for saving best model
#                   if self.verbose > 0:
#                     print(f"Saving new best model to {self.save_path}")
#                   self.model.save(self.save_path)

#         return True

# # path = "/home/admin/mujokaleido/mjk_results"
# # ospath = os.path.join("/home/admin/mujokaleido/")
# model = SAC("MultiInputPolicy", env, verbose=2, tensorboard_log="tensorB/"+log_dir)
# timesteps = 2000000; timeInterval = 100
# # Create the callback: check every 1000 steps
# callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
# # Train the agent
# model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=20)

# #RANDOM FIRST RANDOM
# env = UDRMujocoWrapper(env, modify_bodiez_friction)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=20, tb_log_name="Random1_FRICTION", reset_num_timesteps=False)
# #DEFAULT DEFAULT DEFAULT
# # env = gym.make(KHI_env); env = Monitor(env, log_dir)
# # model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=20, tb_log_name="Default_2"+str(timesteps), reset_num_timesteps=False)

# #RANDOM SECOND RANDOM
# env = UDRMujocoWrapper(env, modify_bodiez_friction, modify_option)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=20, tb_log_name="Random2_+GRAVITY", reset_num_timesteps=False)

# #RANDOM3
# env = UDRMujocoWrapper(env, modify_bodiez_friction, modify_option, modify_bodiez_mass)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=20, tb_log_name="Random3_+MASS", reset_num_timesteps=False)

# #RANDOM4
# env = UDRMujocoWrapper(env, modify_bodiez_friction, modify_option, modify_bodiez_mass, modify_joints)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=20, tb_log_name="Random4_+JOINTS", reset_num_timesteps=False)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# model.save("fu") #no xyz

class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv):
        # self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs[self.key]

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs[self.key], reward, done, info

env = DummyVecEnv([lambda: env])
# Wrap the VecEnv
env = VecExtractDictObs(env)
