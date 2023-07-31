class Config:
    def __init__(self):
        self.buffer_size = None
        self.batch_size = 256
        self.explore_noise = 0.1
        self.tau = 0.005
        self.v_lr = 3e-4
        self.pi_lr = 3e-4
        self.gamma = 0.99
        self.eval_epochs = 10

        self.max_timestamp = 1000000
        self.eval_freq = 5000

class InvertedPendulumConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.a_low = -3.0
        self.a_high = 3.0
        self.start_steps = 1000
        self.env = "InvertedPendulum-v4"
        self.seed = seed
        self.env_name = "InvertedPendulum"

class HalfCheetahConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "HalfCheetah-v4"
        self.seed = seed
        self.env_name = "HalfCheetah"

        self.start_steps = 25000

class AntConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "Ant-v4"
        self.seed = seed
        self.env_name = "Ant"
        self.start_steps = 25000