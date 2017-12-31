import unittest

import numpy as np
from numpy.testing import assert_array_equal

from sim import Simulator



class SimulatorTest(unittest.TestCase):

    def test_new_pos(self):
        cases = [
            {'pos': np.array([2, 2]), 'a': 0, 'new_pos': [2, 2], 'ok': True},
            {'pos': np.array([2, 2]), 'a': 1, 'new_pos': [1, 2], 'ok': True},
            {'pos': np.array([2, 2]), 'a': 2, 'new_pos': [2, 1], 'ok': True},
            {'pos': np.array([2, 2]), 'a': 3, 'new_pos': [3, 2], 'ok': True},
            {'pos': np.array([2, 2]), 'a': 4, 'new_pos': [2, 3], 'ok': True},
            {'pos': np.array([0, 2]), 'a': 1, 'new_pos': [0, 2], 'ok': False},
            {'pos': np.array([2, 0]), 'a': 2, 'new_pos': [2, 0], 'ok': False},
            {'pos': np.array([4, 2]), 'a': 3, 'new_pos': [4, 2], 'ok': False},
            {'pos': np.array([2, 4]), 'a': 4, 'new_pos': [2, 4], 'ok': False},
        ]

        sim = self.setup_sim()

        for case in cases:
            new_pos, ok = sim._new_pos(case['pos'], case['a'])
            self.assertEqual(ok, case['ok'])
            self.assertListEqual(list(new_pos), case['new_pos'])

    def test_calculate_obs(self):
        cases = [
            {'pos': np.array([2, 2]), 'obs': [1, 0, 1, 1]},
            {'pos': np.array([0, 0]), 'obs': [0, 0, 0, 1]},
            {'pos': np.array([4, 3]), 'obs': [0, 0, 0, -1]},
            {'pos': np.array([3, 1]), 'obs': [0, -1, 1, 1]},
        ]

        sim = self.setup_sim()

        for case in cases:
            sim.agent_pos = case['pos']
            obs = sim._calculate_obs()
            self.assertListEqual(list(obs), case['obs'])

    def setup_sim(self):
        sim = Simulator()
        sim.grid = np.array([[0, 2, 0, 0, 3], [0, 0, 2, 2, 0], [0, 0, 0, 2, 0], [3, 0, 2, 0, 0], [0, 2, 0, 0, 3]])

        return sim



if __name__ == '__main__':
    unittest.main()
