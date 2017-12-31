import argparse

import gym
from baselines import deepq

from env import Env



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
            max_timesteps=1000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=10,
            callback=None
        )

        print('Saving model to alife.pkl')
        act.save('alife.pkl')
    elif args.action == 'run':
        act = deepq.load('alife.pkl')

        while True:
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                env.render()
                obs, rew, done, _ = env.step(act(obs[None])[0])
                episode_rew += rew

            print('Episode reward', episode_rew)
