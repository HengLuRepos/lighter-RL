class Config:
    def __init__(self):
        self.v_lr = 1e-4
        self.pi_lr = 3e-4

        self.gamma = 0.9
        self.lam = 0.97
        self.batch_size = 2048
        self.epoch = 1000
        self.max_ep_len = 200
        self.clip=0.2
        self.layer_size = 256

        self.update_freq = 10
        self.total_timesteps = 1000000
        
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
        self.batch_size = 20000
        self.epoch = 1000
        self.max_ep_len = 150
        self.update_freq = 15
        self.layer_size = 64

class AntConfig(Config):
    def __init__(self, seed):
        super().__init__()

        self.env = "Ant-v4"
        self.seed = seed
        self.env_name = "Ant"

        self.batch_size = 2000
        self.epoch = 500
        self.max_ep_len = 500
        self.update_freq = 15
        self.layer_size = 64


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
