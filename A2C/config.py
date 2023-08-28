class Config:
    def __init__(self):
        self.buffer_size = None
        self.batch_size = 2000
        self.v_lr = 5e-3
        self.pi_lr = 5e-3
        self.gamma = 0.90
        self.eval_epochs = 10
        self.lam = 0.90
        self.max_timestamp = 1000000
        self.eval_freq = 5000
        self.max_ep_len = 200
        self.num_epoch = 500
        self.layer_size = 64

class InvertedPendulumConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.a_low = -3.0
        self.a_high = 3.0
        self.env = "InvertedPendulum-v4"
        self.seed = seed
        self.env_name = "InvertedPendulum"
        self.max_ep_len = 1000
        self.gamma = 0.95
        self.v_lr = 0.01
        self.pi_lr = 0.01
        self.batch_size = 5000
        self.layer_size = 64
        self.num_epoch = 100

class HalfCheetahConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "HalfCheetah-v4"
        self.seed = seed
        self.env_name = "HalfCheetah"

        self.max_ep_len = 150
        self.num_epoch = 150
        self.batch_size = 30000
        self.layer_size = 32
        self.v_lr = 0.02
        self.pi_lr = 0.02

class AntConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "Ant-v4"
        self.seed = seed
        self.env_name = "Ant"
        self.start_steps = 25000
        self.steps_per_epoch = 3000

class HopperConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "Hopper-v4"
        self.seed = seed
        self.env_name = "Hopper"
        self.gamma = 0.99
        self.start_steps = 25000
        self.steps_per_epoch = 3000

class HumanoidConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -0.4
        self.a_high = 0.4
        self.env = "Humanoid-v4"
        self.seed = seed
        self.env_name = "Humanoid"
        self.gamma = 0.99
        self.start_steps = 25000
        self.steps_per_epoch = 3000

class HumanoidStandupConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -0.4
        self.a_high = 0.4
        self.env = "HumanoidStandup-v4"
        self.seed = seed
        self.env_name = "HumanoidStandup"
        self.gamma = 0.99


class InvertedDoublePendulumConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "InvertedDoublePendulum-v4"
        self.seed = seed
        self.env_name = "InvertedDoublePendulum"
        self.gamma = 0.99


class PusherConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -2.0
        self.a_high = 2.0
        self.env = "Pusher-v4"
        self.seed = seed
        self.env_name = "Pusher"
        self.gamma = 0.99


class ReacherConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "Reacher-v4"
        self.seed = seed
        self.env_name = "Reacher"
        self.gamma = 0.99


class SwimmerConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "Swimmer-v4"
        self.seed = seed
        self.env_name = "Swimmer"
        self.gamma = 0.99


class Walker2DConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.a_low = -1.0
        self.a_high = 1.0
        self.env = "Walker2d-v4"
        self.seed = seed
        self.env_name = "Walker2d"
        self.gamma = 0.99
