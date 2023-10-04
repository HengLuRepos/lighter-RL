class Config:
    def __init__(self):
        self.layer_size = 256
        self.buffer_size = 1000000
        self.batch_size = 256
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


        self.env = "InvertedPendulum-v4"
        self.seed = seed
        self.env_name = "InvertedPendulum"

class HalfCheetahConfig(Config):
    def __init__(self, seed):
        super().__init__()


        self.env = "HalfCheetah-v4"
        self.seed = seed
        self.env_name = "HalfCheetah"


class AntConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "Ant-v4"
        self.seed = seed
        self.env_name = "Ant"

class HopperConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.env = "Hopper-v4"
        self.seed = seed
        self.env_name = "Hopper"


class HumanoidConfig(Config):
    def __init__(self, seed):
        super().__init__()
        self.env = "Humanoid-v4"
        self.seed = seed
        self.env_name = "Humanoid"


class HumanoidStandupConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "HumanoidStandup-v4"
        self.seed = seed
        self.env_name = "HumanoidStandup"


class InvertedDoublePendulumConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "InvertedDoublePendulum-v4"
        self.seed = seed
        self.env_name = "InvertedDoublePendulum"


class PusherConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "Pusher-v4"
        self.seed = seed
        self.env_name = "Pusher"

class ReacherConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "Reacher-v4"
        self.seed = seed
        self.env_name = "Reacher"


class SwimmerConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "Swimmer-v4"
        self.seed = seed
        self.env_name = "Swimmer"


class Walker2DConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "Walker2d-v4"
        self.seed = seed
        self.env_name = "Walker2d"
