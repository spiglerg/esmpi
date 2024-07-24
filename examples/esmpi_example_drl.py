"""
Simple example to run OpenAI-ES for DRL.

Run as
    $ CUDA_VISIBLE_DEVICES="" mpirun -n 16 python esmpi_example_drl.py
"""
import time
import numpy as np

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from esmpi import ES_MPI, util_torch_set_parameters, util_torch_get_parameters



def make_env(env_id, rank, seed = 0):
    def _init():
        e = gym.make(env_id, render_mode=None)
        e.reset(seed=seed + rank)
        return e
    set_random_seed(seed)
    return _init
env = DummyVecEnv([make_env("CartPole-v1", i) for i in range(1)])

# Hacky: use sb3 to initialize an actor-critic policy; we will not use it for training, but only
# to store the policy's parameters.
model = A2C("MlpPolicy", env, verbose=1)
initial_parameters = util_torch_get_parameters(model.policy)

optimizer = ES_MPI(n_params=len(initial_parameters),
                   population_size=192,
                   initial_parameters=initial_parameters,
                   learning_rate=0.003,
                   sigma=0.02)



def eval_fn(params):
    util_torch_set_parameters(params, model.policy)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return -mean_reward # negative, since we wish to maximize the reward

for i in range(30):
    st = time.time()
    fit = optimizer.step(eval_fn)
    en = time.time()

    if optimizer.is_master:
        print(f"{i}: mean reward {-np.mean(fit)}, iteration time {en-st}\n")

if optimizer.is_master:
    util_torch_set_parameters(optimizer.get_parameters(), model.policy)
    model.save("cartpolev1")
    