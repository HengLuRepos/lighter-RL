import torch
import torch.nn as nn
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def mlp(input_size, out_size, n_layers, layer_size):
    return nn.Sequential(
        nn.Linear(input_size, layer_size),
        nn.ReLU(),
        *([nn.Linear(layer_size, layer_size), nn.ReLU()] * (layer_size - 1)),
        nn.Linear(layer_size, out_size)
    )

def np2torch(np_arr):
    return torch.from_numpy(np_arr).to(device).float() if isinstance(np_arr, np.ndarray) else np_arr.to(device).float()