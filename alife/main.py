import argparse

from baselines.deepq.experiments.train_cartpole import main as train_cartpole_main
from baselines.deepq.experiments.enjoy_cartpole import main as enjoy_cartpole_main

def parse_args():
    parser = argparse.ArgumentParser(description='Bring an agent into life')
    parser.add_argument('--train', dest='action', action='store_const',
                        const='train', default='run',
                        help='train a model (default: run a pre-trained model)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.action == 'train':
        train_cartpole_main()
    elif args.action == 'run':
        enjoy_cartpole_main()
