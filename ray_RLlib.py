# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.sac import SACTrainer
env = "muj_envs_folder:KaleidoKHI-v0"
# env = "muj_envs_folder:KaleidoBed-v0"

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": env,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "policy_model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": None,
        "custom_model": None,  # Use this to define a custom policy model.
        "custom_model_config": {},
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}

# Create our RLlib Trainer.
trainer = SACTrainer(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for i in range(2):
    # print(trainer.train())
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
    print ("slim shaaady")
# trainer.save("/home/admin/mujokaleido/ray/")

# obs = env.reset()
# done = False
# total_R = 0.0
# while not done:
#     action = trainer.compute_single_action(obs)
#     obs, reward, done, info = env.step(action)
#     total_R += reward
# print ("Total reward", total_R)
# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# trainer.evaluate()
