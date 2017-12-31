import numpy as np


class Simulator(object):

    def __init__(self):
        self.grid = self.init_grid()
        self.agent_pos = np.array([2, 2])

    def act(self, a):
        obs = 0
        reward = 0
        done = False

        # Move agent if possible
        new_pos, ok = self._new_pos(self.agent_pos, a)
        if ok:
            self.agent_pos = new_pos

        reward = self._get_reward()

        # Apply special actions, e.g. eating food
        if self.grid[tuple(self.agent_pos)] == 2:
            self.grid[tuple(self.agent_pos)] = 0

        obs = self._calculate_obs()
        done = self.is_done()

        return obs, reward, done

    def reset(self):
        self.grid = self.init_grid()
        self.agent_pos = np.array([2, 2])

        return self._calculate_obs()

    def init_grid(self):
        return np.array([[0, 0, 3], [0, 0, 0], [2, 0, 0]])

    def is_done(self):
        return 2 not in self.grid

    def _get_reward(self):
        reward = 0

        if self.grid[tuple(self.agent_pos)] == 2:
            reward += 1
        elif self.grid[tuple(self.agent_pos)] == 3:
            reward -= 1

        return reward

    def _new_pos(self, cur_pos, a):
        is_valid = True
        diff = np.array([a - 2 if a % 2 == 1  else 0, a - 3 if a != 0 and a % 2 == 0 else 0])
        new_pos = self.agent_pos + diff
        if (new_pos[0] < 0 or new_pos[0] >= len(self.grid)) or\
           (new_pos[1] < 0 or new_pos[1] >= len(self.grid[0])):
            is_valid = False

        return new_pos, is_valid

    def _calculate_obs(self):
        obs = [0, 0, 0, 0]

        for i in range(1, 5):
            new_pos, ok = self._new_pos(self.agent_pos, i)
            if ok:
                obs[i - 1] = self.grid[tuple(new_pos)]
            else:
                obs[i - 1] = 1

        value = obs[3] + 4 * obs[2] + 16 * obs[1] + 64 * obs[0]
        #result = np.zeros(256, int)
        #result[value] = 1
        return value

    def __str__(self):
        s = ''

        tmp_grid = self.grid.copy()
        tmp_grid[tuple(self.agent_pos)] = 4

        def get_grid_char(num):
            if num == 0:
                return '-'
            elif num == 1:
                return ' '
            elif num == 2:
                return '+'
            elif num == 3:
                return '*'
            elif num == 4:
                return 'a'

        for r in tmp_grid:
            for c in r:
                s += get_grid_char(c)
            s += '\n'

        return s
