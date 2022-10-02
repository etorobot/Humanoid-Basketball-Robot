"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
"""
import argparse, functools, logging, sys
from distutils.version import LooseVersion

import gym
import gym.wrappers
from gym.utils import seeding
import numpy as np, torch
from torch import distributions, nn, true_divide

import pfrl
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="mjk_results", help=("Directory path to save output files. If it does not exist, it will be created."),)
    
    parser.add_argument("--env",    type=str,
                        # default="muj_envs_folder:KStandupEnv-v0",
                        # default="muj_envs_folder:KHumanoidEnv-v0", 
                        default="muj_envs_folder:KHumanoid-v3",   
                        # default="gym.envs.mujoco:Ant-v2", 
                        # default="drb_envs:K22Env-v0", 
                             
                        help="OpenAI Gym MuJoCo env to perform algorithm on.",)

    parser.add_argument("--load",   type=str, 
                        # default="/home/admin/dribble_repo/results/e6d4944dc0c55a21dd9de76aee81dbc0949d9aa0-00000000-61732a5d/16913_except"
                        # default="/home/admin/mujokaleido/",
                        default="", 
                        help="Directory to load agent from.")

    parser.add_argument("--demo",
                        action="store_false",
                        # action="store_true",
                        help="Just run evaluation, not training.")
    
    parser.add_argument("--steps",                  type=int,   default=3000000,        help="Total number of timesteps to train the agent.",)
    parser.add_argument("--num-envs",               type=int,   default=1,              help="Number of envs run in parallel.")
    parser.add_argument("--seed",                   type=int,   default=0,              help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu",                    type=int,   default=0,              help="GPU to use, set to -1 if no GPU.")
    parser.add_argument("--eval-n-runs",            type=int,   default=20,             help="Number of episodes run for each evaluation.",)
    parser.add_argument("--eval-interval",          type=int,   default=5000,           help="Interval in timesteps between evaluations.",)
    parser.add_argument("--replay-start-size",      type=int,   default=10000,          help="Minimum replay buffer size before performing gradient updates.",)
    parser.add_argument("--update-interval",        type=int,   default=1,              help="Interval in timesteps between model updates.",)    
    parser.add_argument("--batch-size",             type=int,   default=256,            help="Minibatch size")
    parser.add_argument("--render",                 action="store_true",                help="Render env states in a GUI window.")
    parser.add_argument("--load-pretrained",        action="store_true",                default=False)
    parser.add_argument("--monitor",                action="store_true",                help="Wrap env with gym.wrappers.Monitor.")
    parser.add_argument("--pretrained-type",        type=str,   default="best",         choices=["best", "final"])
    parser.add_argument("--log-interval",           type=int,   default=1000,           help="Interval in timesteps between outputting log messages during training",)
    parser.add_argument("--log-level",              type=int,   default=logging.INFO,   help="Level of the root logger.")
    parser.add_argument("--policy-output-scale",    type=float, default=1.0,            help="Weight initialization scale of policy output.",)
    
    parser.add_argument("--n-hidden-channels",      type=int,   default=1024,           help="Number of hidden channels of NN models.", )
    parser.add_argument("--discount",               type=float, default=0.98,           help="Discount factor.")
    parser.add_argument("--n-step-return",          type=int,   default=3,              help="N-step return.")
    parser.add_argument("--lr",                     type=float, default=3e-4,           help="Learning rate.")
    parser.add_argument("--adam-eps",               type=float, default=1e-1,           help="Adam eps.") 
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(args, process_idx, test):
        env = gym.make(args.env)
        # Unwrap TimiLimit wrapper
        # assert isinstance(env, gym.wrappers.TimeLimit) #comment because AssertionError
        # env = env.env #above
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        # env.verbose=True
        # env.reset()
        # env.seed(env_seed)
        env.seed(int(env_seed))
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        # Normalize action space to [-1, 1]^n
        env = pfrl.wrappers.NormalizeActionSpace(env)
        # env = pfrl.wrappers.Monitor(env, args.outdir, force=True, video_callable=lambda _: True)
        # env = pfrl.wrappers.Render(env, mode="human")
        if args.monitor:
            # env = gym.wrappers.Monitor(env, args.outdir)
            env = pfrl.wrappers.Monitor(env, args.outdir, force=True, video_callable=lambda _: True)
        if args.render:
            # env = pfrl.wrappers.Render(env)
            env = pfrl.wrappers.Render(env, mode="human")
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test) #original
                # functools.partial(make_env, args, process_seeds[idx], test) #KSTAND
                # functools.partial(make_env, process_seeds[idx], test)
                for idx, env in enumerate(range(args.num_envs))
            ])

    sample_env = make_env(args, process_seeds[0], test=False) #original
    # sample_env = make_env(args, process_seeds[0], test=False) #KSTAND
    timestep_limit  = sample_env.spec.max_episode_steps
    obs_space    = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)
    del sample_env

    obs_size    = obs_space.low.size
    action_size = action_space.low.size

    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent( distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1 )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
                base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
                )

    policy = nn.Sequential(
            nn.Linear(obs_space.low.size, args.n_hidden_channels),
            nn.ReLU(),
            nn.Linear(args.n_hidden_channels, args.n_hidden_channels),
            nn.ReLU(),
            nn.Linear(args.n_hidden_channels, action_size * 2),
            Lambda(squashed_diagonal_gaussian_head),
                        )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    # torch.nn.init.xavier_uniform_(policy[4].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight, gain=args.policy_output_scale)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=args.adam_eps)

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_space.low.size + action_size, args.n_hidden_channels),
            nn.ReLU(),
            nn.Linear(args.n_hidden_channels, args.n_hidden_channels),
            nn.ReLU(),
            nn.Linear(args.n_hidden_channels, 1),            
            # nn.Linear(obs_size + action_size, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        # q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=args.lr, eps=args.adam_eps)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10 ** 6, num_steps=args.n_step_return)
    # rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=args.discount,
        update_interval=args.update_interval,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=args.lr,
    )

    if len(args.load) > 0 or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not len(args.load) > 0 or not args.load_pretrained
        if len(args.load) > 0:
            agent.load(args.load)
        else:
            agent.load(utils.download_model("SAC", args.env, model_type=args.pretrained_type)[0])

    if args.demo:
        eval_env1 = make_env(args, process_idx=0, test=True)
        eval_env = make_batch_env(test=True),
        eval_env1.reset() #this and while Keeps from closing after 10 secs
        while True:
            eval_stats = experiments.eval_performance(
                env=eval_env,
                agent=agent,
                n_steps=None,
                n_episodes=args.eval_n_runs,
                max_episode_len=timestep_limit
            )
            print(
                "n_runs: {} mean: {} median: {} stdev {}".format(
                    args.eval_n_runs,
                    eval_stats["mean"],
                    eval_stats["median"],
                    eval_stats["stdev"],
                )
            )

            # import json
            # import os

            # with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
            #     json.dump(eval_stats, f)
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            use_tensorboard=0
            )
        make_env(args, seed=0, test=True).render()


if __name__ == "__main__":
    main()
