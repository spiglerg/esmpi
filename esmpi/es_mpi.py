"""
Main ES-MPI class.
"""

import copy
import numpy as np

import torch
import torch.nn as nn

from mpi4py import MPI

from .optimizers import Adam


## Adapted from OpenAI - Evolution Strategies
def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def compute_identity_ranks(x):
    return x


class SharedNoiseTable(object):
    """
    A class that creates a shared noise table for parallel computing using MPI.

    This class creates a large table of random numbers that can be shared across multiple processes.
    This is useful for parallel computing tasks where each process needs access to a common set of random numbers.

    From OpenAI-ES.
    """
    def __init__(self):
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        comm = MPI.COMM_WORLD

        if comm.Get_rank() == 0:
            print(f"Sampling {count//1e6}M random numbers with seed {seed}")

        shared_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

        float_size = MPI.FLOAT.Get_size()
        if shared_comm.Get_rank() == 0:
            nbytes = count * float_size
        else:
            nbytes = 0
        self._shared_mem = MPI.Win.Allocate_shared(nbytes, float_size, comm=shared_comm)

        self.buf, itemsize = self._shared_mem.Shared_query(0)
        assert itemsize == MPI.FLOAT.Get_size()
        self.noise = np.ndarray(buffer=self.buf, dtype=np.float32, shape=(count,))
        assert self.noise.dtype == np.float32

        if shared_comm.Get_rank() == 0:
            self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        comm.Barrier()

        if comm.Get_rank() == 0:
            print(f"Sampled {(self.noise.size*4)//1e6}M bytes")

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)
## END OpenAI - Evolution Strategies



class ES_MPI():
    """
    The ESMPI class implements Evolution Strategies using MPI (Message Passing Interface) for parallel computing.
    """
    def __init__(self,
                 n_params,
                 population_size=160,
                 initial_parameters = None,
                 learning_rate=0.003,
                 sigma=0.02,
                 use_antithetic_sampling=True,
                 fitness_transform_fn=compute_centered_ranks):
        """
        Initialize the ESMPI class.

        Args:
            n_params: Number of parameters to optimize.
            population_size: The size of the population in each generation (default is 160). This should be a multiple
                            of the number of parallel MPI processes used.'
            initial_parameters (np.ndarray, optional): The initial mean of the distribution. Default is None,
                                                 corresponding to an initialization to 0. When optimizing neural
                                                 networks, it is recommended to use the initial random weights of the
                                                 network.
            learning_rate: The learning rate for the optimizer (default is 0.003).
            sigma: The standard deviation of the noise added to the parameters during the exploration phase
                    (default is 0.02).
            use_antithetic_sampling: Whether to use antithetic sampling, which can improve the efficiency of the
                                    algorithm (default is True).
            fitness_transform_fn: The function to transform the fitness scores (default is compute_centered_ranks).
        """

        # TODO: add support for different optimizers, i.e., SGD and Adam instead of just Adam.

        self.n_params = n_params
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population_size = population_size
        self.use_antithetic_sampling = use_antithetic_sampling

        self.fitness_transform_fn = fitness_transform_fn

        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self.num_workers = self._comm.Get_size()
        self.is_master = self._rank==0

        self.noise = SharedNoiseTable()
        self.rs = np.random.RandomState()

        self._optimizer = Adam(self.n_params, lr=self.learning_rate)
        #self._optimizer = SGD(self.n_params, lr=self.learning_rate)

        self.population_per_worker = self.population_size // self.num_workers
        if self.population_size % self.num_workers != 0:
            print(f"\033[91m ERROR: population size {self.population_size} \
                  is not a multiple of the number of workers {self.num_workers}\033[0m")
        if self.is_master:
            print(f"\033[93m Running {self.population_per_worker} perturbations on each worker, \
                  over {self.num_workers} workers. \033[0m")

        # Initialize weights and sync them across workers
        if initial_parameters is None:
            self.current_parameters = np.zeros(self.n_params, dtype=np.float32)
        else:
            self.current_parameters = copy.copy(initial_parameters)
        self.current_parameters = self._comm.bcast(self.current_parameters, root=0)

    def step(self, eval_fn, **eval_fn_kwargs):
        """
            eval_fn: function that assesses the performance of a parameter vector passed as argument.
            The function is independent, and can rely on global variables
            (e.g., for datasets or RL environments)

            If master, returns all fitness values from all workers; else, an empty list is returned.
        """

        # Each worker samples its 'self.population_per_worker' random perturbations + evaluate their fitness
        worker_results = []
        for i in range(self.population_per_worker):
            #perturbation_i = np.random.randn(*self.current_parameters.shape).astype(np.float32)
            noise_idx = self.noise.sample_index(self.rs, len(self.current_parameters))
            perturbation_i = self.noise.get(noise_idx, len(self.current_parameters))

            fitness_values = []
            for j in range(2 if self.use_antithetic_sampling else 1):
                sign = 1 if j == 0 else -1
                perturbed_parameters = self.current_parameters + sign * self.sigma * perturbation_i
                fitness_values.append( eval_fn(perturbed_parameters, **eval_fn_kwargs) )

            worker_results.append([noise_idx, fitness_values])

        # Workers send the perturbed weights (indices in the shared noise table) + the corresponding fitnesses
        # to the master
        worker_results = self._comm.gather(worker_results, root=0)

        # The master computes the new weights and sends them to all workers -> or no master, and each worker
        # send their data to all other workers
        if self.is_master:
            # Master
            worker_results = sum(worker_results, [])

            noise_ids, fitness_values = zip(*worker_results)

            # Fitness shaping
            all_fitnesses = np.asarray(fitness_values) # incl. antithetic pairs
            all_fitnesses_flattened = np.reshape(all_fitnesses, -1)
            fitnesses_ = compute_centered_ranks(np.asarray(all_fitnesses_flattened))
            fitnesses_ = np.reshape(fitnesses_, all_fitnesses.shape)

            fitnesses = []
            for i in range(len(fitnesses_)):
            # (if antithetic, only save a single (perturbation, (perf(+epsilon)-perf(-epsilon))/2 ) )
                if self.use_antithetic_sampling:
                    fitnesses.append( fitnesses_[i,0] - fitnesses_[i,1] )
                else:
                    fitnesses.append( fitnesses_[i,0] )

            gradient = np.zeros(self.current_parameters.shape, dtype=np.float32)
            for j in range(len(noise_ids)):
                perturbation_j = self.noise.get(noise_ids[j], len(self.current_parameters))
                gradient += perturbation_j * fitnesses[j]

            gradient = gradient / len(noise_ids) / self.sigma
            if self.use_antithetic_sampling:
                # In antithetic sampling, the gradient is 1/(2*sigma) instead of 1/sigma.
                gradient /= 2.0

            self.current_parameters = self._optimizer.update(self.current_parameters, gradient)

            #if np.isnan(self.current_parameters).any() or np.isinf(self.current_parameters).any():
            #    print('ERROR: nans or infs detected after ES update.')

        self.current_parameters = self._comm.bcast(self.current_parameters, root=0)

        if not self.is_master:
            all_fitnesses_flattened = []
        return all_fitnesses_flattened

    def get_parameters(self):
        """
        Get current mean parameters of the optimizer.
        """
        return self.current_parameters



def util_torch_set_parameters(parameters, network, device="cpu"):
    """
    Set the parameters of the optimizer to the network.
    """
    nn.utils.vector_to_parameters(torch.Tensor(parameters).to(device), network.parameters())
    return network

def util_torch_get_parameters(network):
    """
    Get the parameters of the network.
    """
    return nn.utils.parameters_to_vector(network.parameters()).detach().cpu().numpy()
