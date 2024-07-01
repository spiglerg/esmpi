"""
Run as
    $ mpirun -n 16 python esmpi_example.py
"""
import numpy as np

from esmpi import ES_MPI

def eval_fn(x):
    return (x[0] - 3) ** 2 + (10 * (x[1] + 2)) ** 2 + np.sum(x[2:]**2)

optimizer = ES_MPI(n_params=10, population_size=16, learning_rate=0.1, sigma=0.02)

for i in range(50):
    fit = optimizer.step(eval_fn)

    if optimizer.is_master == 0:
        print(f"{i}: fitness {np.mean(fit)}, current best params {optimizer.get_parameters()}\n")
