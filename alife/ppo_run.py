import argparse

import gym
from baselines import ppo1

from env import Env

import argparse
from baselines import bench, logger

def train(args):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    num_timesteps = args.num_timesteps
    seed = args.seed
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = Env()
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    # Set model parameters
    policy = MlpPolicy
    nsteps = 2048
    nminibatches = 32
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5

    if args.action == 'train':
        ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=nminibatches,
            lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
            ent_coef=ent_coef,
            lr=3e-4,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            cliprange=0.2,
            total_timesteps=num_timesteps,
            save_interval=10)
    elif args.action == 'run':
        trained_model = ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
        trained_model.load('/tmp/openai-2018-01-08-23-40-30-695356/checkpoints/00040')
        env2 = Env()
        for i in range(1):
            obs, done = env2.reset(), False
            episode_rew = 0
            while not done:
                env2.render()
                obs, rew, done, _ = env2.step(trained_model.step(obs[None])[0])
                episode_rew += rew

            print('Episode reward', episode_rew)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e5))
    parser.add_argument('--train', dest='action', action='store_const',
                        const='train', default='run',
                        help='train a model (default: run a pre-trained model)')
    args = parser.parse_args()
    logger.configure()
    train(args)


if __name__ == '__main__':
    main()

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Bring an agent into life')
    parser.add_argument('--train', dest='action', action='store_const',
                        const='train', default='run',
                        help='train a model (default: run a pre-trained model)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    env = Env()
    if args.action == 'train':
        model = deepq.models.mlp([64])
        act = deepq.learn(
            env,
            q_func=model,
            lr=1e-3,
            max_timesteps=50000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            print_freq=10,
            callback=None
        )

        print('Saving model to alife.pkl')
        act.save('alife.pkl')
    elif args.action == 'run':
        act = deepq.load('alife.pkl')

        for i in range(1):
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                env.render()
                obs, rew, done, _ = env.step(act(obs[None])[0])
                episode_rew += rew

            print('Episode reward', episode_rew)

"""
