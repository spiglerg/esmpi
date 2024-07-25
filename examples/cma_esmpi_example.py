"""
Run as
    $ mpirun -n 16 python cma_esmpi_example.py
"""

import numpy as np
from mpi4py import MPI

from esmpi import CMA_ESMPI

def eval_fn(x):
    return (x[0] - 3) ** 2 + (10 * (x[1] + 2)) ** 2 + np.sum(x[2:]**2)


if __name__ == "__main__":
    optimizer = CMA_ESMPI(n_params=10, population_size=16)

    for i in range(100):
        fit = optimizer.step(eval_fn)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"{i}: fitness {np.mean(fit)}, current best params {optimizer.get_parameters()}\n")

        if optimizer.should_stop():
            break
