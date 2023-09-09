import scipy
import numpy as np
import torch
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
def discount_cumsum2(x: torch.Tensor, discount):
    sum = 0.0
    result = torch.zeros_like(x)
    for i in range(len(x) - 1, -1, -1):
        sum = discount * sum + x[i]
        result[i] = sum
    return result
x = np.ones(10)
print(discount_cumsum(x, 0.9))
x = torch.ones(10)
print(discount_cumsum2(x, 0.9))