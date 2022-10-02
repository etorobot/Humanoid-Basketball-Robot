import gym, os
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

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

# Create log dir
log_dir = "just_xyz/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make("muj_envs_folder:KaleidoKHI-v0")
env = Monitor(env, log_dir)

# path = "/home/admin/mujokaleido/mjk_results"
# ospath = os.path.join("/home/admin/mujokaleido/")
model = SAC("MultiInputPolicy", env, verbose=2)

# Create the callback: check every 1000 steps
# callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
# Train the agent
timesteps = 3_000_000
# model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=50)
# model.learn(total_timesteps=1_000_000, log_interval=1)

# plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "K__Stand")
# plt.show()

# model.save("fu") #no xyz
# model.save("26_10_5M") 
# model.save("50_25_5M")
# model.save("test")
model = SAC.load("best_model")
# mean_reward, std_reward = evaluate_policy(model, env)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
obs = env.reset()
# for i in range(1000):
while True:    
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()


# env.close()