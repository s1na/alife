import sys

import gym
from gym.spaces import Discrete, Box
from six import StringIO

from sim import Simulator
from owndiscrete import Owndiscrete



class Env(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.sim = Simulator()
        # {0: noop, 1: up, 2: left, 3: down, 4: right}
        self.action_space = Owndiscrete(5)
        # {up: [blank, wall, food, danger], ...}
        self.observation_space = Box(low=-1, high=1, shape=(4,))

    def _step(self, a):
        obs, reward, done = self.sim.act(a)

        return obs, reward, done, {}

    def _reset(self):
        return self.sim.reset()

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(str(self.sim))
        return outfile

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
