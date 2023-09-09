class HalfCheetahConfig:
    def __init__(self):
        self.v_lr = 5e-3
        self.pi_lr = 5e-3

        self.gamma = 0.9
        self.lam = 0.97

        self.batch_size = 10000
        self.epoch = 3000
        self.max_ep_len = 1000
        self.clip=0.2

        self.update_freq = 15

class PendulumConfig:
    def __init__(self):
        self.v_lr = 5e-3
        self.pi_lr = 5e-3

        self.gamma = 0.9
        self.lam = 0.97

        self.batch_size = 1000
        self.epoch = 1000
        self.max_ep_len = 1000
        self.clip=0.2


        self.update_freq = 5


class CartPoleConfig:
    def __init__(self):
        self.v_lr = 5e-3
        self.pi_lr = 5e-3

        self.gamma = 0.9
        self.lam = 0.97

        self.batch_size = 1000
        self.epoch = 1000
        self.max_ep_len = 200
        self.clip=0.2


        self.update_freq = 5