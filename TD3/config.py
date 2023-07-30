class Config:
    def __init__(self):
        self.buffer_size = 40000
        self.batch_size = 200
        self.explore_noise = 0.1
        self.target_noise = 0.2
        self.noise_clip = 0.5
        self.tau = 0.005
        self.v_lr = 0.001
        self.pi_lr = 0.001
        self.gamma = 0.99
        self.policy_delay = 2
        self.update_freq = 20
        self.epoch = 1000
        self.steps_per_epoch = 3000
        self.start_steps = 10000
        self.eval_epochs = 100

class InvertedPendulumConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.a_low = -3.0
        self.a_high = 3.0
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

        self.buffer_size = 100000
        self.epoch = 3000
        self.update_freq = 20
        self.steps_per_epoch = 3000
