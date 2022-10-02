import gym, os, numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from simmod.wrappers import UDRMujocoWrapper
from simmod.modification.mujoco import MujocoJointModifier, MujocoOptionModifier, MujocoBodyModifier
from simmod import load_yaml
yamlJoint = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_joints.yaml'))
yamlBodies = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies.yaml'))
yamlOption = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_option.yaml'))
yamlBodiesFriction = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_friction.yaml'))
yamlBodiesMass = load_yaml(os.path.join((os.path.dirname(__file__)), './muj_envs_folder/DR_bodies_mass.yaml'))

# Create log dir
log_dir = "mon4_freeArm96_STAND_500M"
os.makedirs(log_dir, exist_ok=True)
#tensorboard --logdir='./tensorB'

# Create and wrap the environment
KHI_env = "muj_envs_folder:KaleidoKHI-v0"
env = gym.make(KHI_env); env = Monitor(env, log_dir)

modify_bodiez_friction = MujocoBodyModifier(sim=env.sim, config = yamlBodiesFriction)
modify_bodiez_mass = MujocoBodyModifier(sim=env.sim, config = yamlBodiesMass)
# modify_bodiez = MujocoBodyModifier(sim=env.sim, config = yamlBodies)
modify_joints = MujocoJointModifier(sim=env.sim, config = yamlJoint)
modify_option = MujocoOptionModifier(sim=env.sim, config = yamlOption)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

# path = "/home/admin/mujokaleido/mjk_results"
# ospath = os.path.join("/home/admin/mujokaleido/")
# model = SAC.load(, env, verbose=2, tensorboard_log="tensorB/"+log_dir)
model = SAC(
        "MultiInputPolicy",
        # "/home/admin/mujokaleido/tue21_Asymm_CP/best_model.zip",
        # "/home/admin/mujokaleido/tue20_Symm_all/best_model.zip",
        env,
        # verbose=1,
        train_freq=1,
        buffer_size=1_000_000,
        learning_rate=3e-4,
        tensorboard_log="tensorB/"+log_dir,
        # ent_coef=5
        )
timesteps = 5_00_000; timeInterval = 100; trainTest = 1_000
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
# Train the agent
# for i in range(int(timesteps/trainTest)):
# env.render()
model.learn(total_timesteps=int(100*timesteps), callback=callback, log_interval=20)
  # model.save("mo")
  # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
  # model = SAC.load("mo")

# # RANDOM JOINT JOINT
# env = UDRMujocoWrapper(env, modify_joints)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(1*timesteps), callback=callback, log_interval=20, tb_log_name="JOINTS", reset_num_timesteps=False)

# # # #RANDOM KILO KILO KILO
# env = UDRMujocoWrapper(env, modify_bodiez_mass, modify_joints)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(1 *timesteps), callback=callback, log_interval=20, tb_log_name="+MASS", reset_num_timesteps=False)

# # # #RANDOM FRICTION 
# env = UDRMujocoWrapper(env, modify_bodiez_friction, modify_bodiez_mass, modify_joints)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(1 *timesteps), callback=callback, log_interval=20, tb_log_name="+FRICTION", reset_num_timesteps=False)

# # #RANDOM GRAVITY 
# env = UDRMujocoWrapper(env, modify_option, modify_bodiez_friction, modify_bodiez_mass, modify_joints)
# env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(3 *timesteps), callback=callback, log_interval=20, tb_log_name="+FRICTION", reset_num_timesteps=False)

#DEFAULT DEFAULT DEFAULT
# env = gym.make(KHI_env); env = Monitor(env, log_dir)
# model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=20, tb_log_name="Default_2"+str(timesteps), reset_num_timesteps=False)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
