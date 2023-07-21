class Config:
    def __init__(self):
        self.layer_size=256
        self.n_layers=3

        self.pi_lr=3e-4
        self.q_lr=3e-4
        self.v_lr=3e-4
        self.tau=0.005

        self.buffer_size=100000
        self.batch_size=256

        self.gamma=0.99
        self.num_iter = 20000
        self.explore_step = 1000
        self.update_gradient_freq = 1

