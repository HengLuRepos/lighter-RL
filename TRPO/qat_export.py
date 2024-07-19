import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  convert_pt2e,
)
from torch.ao.quantization.quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
from trpo import TRPO
import gymnasium as gym
import argparse
import psutil
import numpy as np
env_map = {
    "HalfCheetah-v4": HalfCheetahConfig,
    "Humanoid-v4": HumanoidConfig,
    "HumanoidStandup-v4": HumanoidStandupConfig,
    "Ant-v4": AntConfig,
    "Hopper-v4": HopperConfig,

}
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    args = parser.parse_args()
    
    return args
args = parse_args()
cfg = env_map[args.env_id]
seed = [2,3,4,5,6,7,8,9,10,11]
#fp32_time = []
int8_time = []
#fp32_step = []
int8_step = []
#fp32_return = []
int8_return = []
fp32_ram = []

config = cfg(seed[0])
env = gym.make(config.env)
agent = TRPO(env, config).to('cpu')
agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")

example_inputs = (torch.from_numpy(env.observation_space.sample()),)

agent_prepared = capture_pre_autograd_graph(agent, *example_inputs)
# we get a model with aten ops

# Step 2. quantization
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they
# want the model to be quantized
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
# or prepare_qat_pt2e for Quantization Aware Training
agent_prepared = prepare_qat_pt2e(agent_prepared, quantizer)
num_updates = config.max_timestamp // config.batch_size
for ep in range(num_updates):
    paths, episodic_rewards = agent_prepared.sample_batch()
    states = np.concatenate([path["states"] for path in paths])
    actions = np.concatenate([path["actions"] for path in paths])
    rewards = np.concatenate([path["rewards"] for path in paths])
    next_states = np.concatenate([path["next_states"] for path in paths])
    done = np.concatenate([path["done"] for path in paths])
    old_logp = np.concatenate([path["log_prob"] for path in paths])
    returns = agent_prepared.get_returns(paths)

    agent_prepared.line_search(states, actions, next_states, rewards, done, old_logp)
    agent_prepared.update_critic(returns, states)
    avg_reward = np.mean(episodic_rewards)
    print(f"Iter {ep} Avg reward {avg_reward:.3f}")

agent_int8 = convert_pt2e(agent_prepared)

torch.ao.quantization.move_exported_model_to_eval(agent_int8)