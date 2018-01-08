from gym.spaces import Discrete

class Owndiscrete(Discrete):
    @property
    def shape(self):
        return (self.n,)
