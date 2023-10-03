class Config:
    def __init__(self):
        self.layer_size = 256
        self.buffer_size = 100000
        self.batch_size = 256
        self.num_iter = 500
        self.tau = 0.005
        self.pi_lr = 3e-4
        self.q_lr = 1e-3
        self.alpha_lr = 1e-3
        self.gamma = 0.99
        self.eval_epochs = 10
        self.policy_freq = 2
        self.eval_freq = 5000
        self.start_steps = 5000
        self.max_timesteps = 1000000

        self.alpha_tune=False
        self.alpha = 0.2

class InvertedPendulumConfig(Config):
    def __init__(self, seed):
        super().__init__()


        self.num_iter = 1000
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

        self.start_steps = 5000
        self.update_gradient_freq = 20

class AntConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "Ant-v4"
        self.seed = seed
        self.env_name = "Ant"
        self.gamma = 0.99
        self.update_gradient_freq = 10
