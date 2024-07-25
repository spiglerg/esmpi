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

    def sample_index(self, stream, dim, size=None):
        return stream.randint(0, len(self.noise) - dim + 1, size=size)
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

        If antithetic sampling is used, the population of workers is split equally in half, with one half evaluating
        the positive perturbations and the other half evaluating the negative perturbations.
        Perturbations in each pair are evaluated in one of two ways, depending on whether the population size is equal
        to the number of available workers, or if the population size is larger than the number of available workers.

        population_size == num_workers: If the population size is equal to the number of available workers, then
                                        even workers compute the positive perturbation, and odd workers compute the
                                        negative perturbation. Perturbations are pre-computed by the master and sent
                                        to the workers.
                                        *Note that in this case, the number of actual perturbations computed is
                                        population_size/2*.
        population_size > num_workers: If the population size is larger than the number of available workers, then
                                        antithetic pairs are evaluated by the same worker.

        Args:
            n_params: Number of parameters to optimize.
            population_size: The size of the population in each generation (default is 160). This should be a multiple
                            of the number of parallel MPI processes used.'
                            population_size = -1 automatically sets the population size to the number of available
                            workers.
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

        if self.population_size == -1:
            self.population_size = self.num_workers

        self.noise = SharedNoiseTable()
        self.rs = np.random.RandomState()

        self._optimizer = Adam(self.n_params, lr=self.learning_rate)
        #self._optimizer = SGD(self.n_params, lr=self.learning_rate)

        self.population_per_worker = self.population_size // self.num_workers
        if self.population_size % self.num_workers != 0:
            print(f"\033[91m ERROR: population size {self.population_size} \
                  is not a multiple of the number of workers {self.num_workers}\033[0m")
            exit(1)
        if self.is_master:
            print(f"\033[93m Running {self.population_per_worker} perturbations on each worker, \
                  over {self.num_workers} workers. \033[0m")

        if self.population_size == self.num_workers and self.use_antithetic_sampling:
            self.antithetic_on_different_nodes = True
        else:
            self.antithetic_on_different_nodes = False

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

        # If population size is equal to the number of workers, then the master pre-computes the perturbations,
        # since each worker will only evaluate a single perturbation, but we need to guarantee the generation
        # of antithetic pairs.
        master_indices = None
        indices_to_send = None
        if self.antithetic_on_different_nodes:
            if self.is_master:
                master_indices = self.noise.sample_index(self.rs,
                                                         len(self.current_parameters),
                                                         size=self.population_size//2)
                indices_to_send = np.repeat(master_indices, 2)
        else:
            # Each worker receives population_per_worker//2 perturbations is antithetic is used, or the whole if not.
            if self.is_master:
                master_indices = self.noise.sample_index(self.rs,
                                                         len(self.current_parameters),
                                                         size=self.population_size)
                indices_to_send = np.split(master_indices, self.num_workers)
        noise_indices = self._comm.scatter(indices_to_send, root=0)

        # Each worker samples its 'self.population_per_worker' random perturbations + evaluate their fitness
        worker_results = []
        for i in range(self.population_per_worker):
            #perturbation_i = np.random.randn(*self.current_parameters.shape).astype(np.float32)
            if isinstance(noise_indices, np.int64):
                noise_idx = noise_indices
            else:
                noise_idx = noise_indices[i]
            perturbation_i = self.noise.get(noise_idx, len(self.current_parameters))

            fitness_values = []
            if not self.antithetic_on_different_nodes:
                # that is, either each worker evaluates both antithetic samples, or no antithetic sampling is used
                for j in range(2 if self.use_antithetic_sampling else 1):
                    sign = 1 if j == 0 else -1
                    perturbed_parameters = self.current_parameters + sign * self.sigma * perturbation_i
                    fitness_values.append( eval_fn(perturbed_parameters, **eval_fn_kwargs) )

            else:
                if self._rank % 2 == 1:
                    perturbation_i = -perturbation_i
                perturbed_parameters = self.current_parameters + self.sigma * perturbation_i
                fitness_values.append( eval_fn(perturbed_parameters, **eval_fn_kwargs) )

            worker_results.append(fitness_values)

        # Workers send the perturbed weights (indices in the shared noise table) + the corresponding fitnesses
        # to the master
        worker_results = self._comm.gather(worker_results, root=0)

        # The master computes the new weights and sends them to all workers -> or no master, and each worker
        # send their data to all other workers
        if self.is_master:
            # Master
            fitness_values = sum(worker_results, [])

            # Fitness shaping
            ret_all_fitnesses = np.asarray(fitness_values)
            all_fitnesses = np.asarray(fitness_values) # incl. antithetic pairs

            #"""
            # DESIGN CHOICE: fitness shaping is applied to the antithetic pairs, not to the individual perturbations
            if self.antithetic_on_different_nodes:
                # master_indices contains the non-duplicated indices of the perturbations, so it has half the size
                # of all_fitnesses, which contains the fitnesses of the antithetic pairs
                all_fitnesses = np.squeeze(all_fitnesses)
                all_fitnesses = all_fitnesses[::2] - all_fitnesses[1::2]

            elif self.use_antithetic_sampling:
                all_fitnesses = all_fitnesses[:,0] - all_fitnesses[:, 1]
            fitnesses = compute_centered_ranks(all_fitnesses)
            #"""

            """
            # DESIGN CHOICE: fitness shaping is applied to the original scores (separately for each antithetic pair)
            # and only after their difference is computed
            all_fitnesses_flattened = all_fitnesses.flatten()
            fitnesses = compute_centered_ranks(all_fitnesses)
            all_fitnesses = fitnesses.reshape(all_fitnesses.shape)

            if self.antithetic_on_different_nodes:
                # master_indices contains the non-duplicated indices of the perturbations, so it has half the size
                # of all_fitnesses, which contains the fitnesses of the antithetic pairs
                all_fitnesses = np.squeeze(all_fitnesses)
                fitnesses = all_fitnesses[::2] - all_fitnesses[1::2]

            elif self.use_antithetic_sampling:
                fitnesses = all_fitnesses[:,0] - all_fitnesses[:, 1]
            #"""

            gradient = np.zeros(self.current_parameters.shape, dtype=np.float32)
            for j in range(len(master_indices)):
                perturbation_j = self.noise.get(master_indices[j], len(self.current_parameters))
                gradient += perturbation_j * fitnesses[j]

            gradient = gradient / len(master_indices) / self.sigma
            if self.use_antithetic_sampling:
                # In antithetic sampling, the gradient is 1/(2*sigma) instead of 1/sigma.
                gradient /= 2.0

            self.current_parameters = self._optimizer.update(self.current_parameters, gradient)

            #if np.isnan(self.current_parameters).any() or np.isinf(self.current_parameters).any():
            #    print('ERROR: nans or infs detected after ES update.')

        self.current_parameters = self._comm.bcast(self.current_parameters, root=0)

        if not self.is_master:
            ret_all_fitnesses = []
        return ret_all_fitnesses

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
