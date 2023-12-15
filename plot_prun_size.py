import numpy as np
import matplotlib.pyplot as plt
labels = ("HalfCheetah", "HumanoidStandup", "Ant")
x = np.arange(len(labels))
width = 0.3
base = (49,231,54)
prune = (72,346,80)
plt.bar(x, base, width, label="baseline")
plt.bar(x+width, prune, width, label="pruning")
plt.ylabel('model size')
plt.title('ppo pruning size')
plt.xticks(x + width / 2, labels)
plt.legend(loc='best')
plt.savefig(f"ppo-pruning-model-size.png")