"""
Simple extension of the CMA-ES optimizer from the `cmaes' to enable parallelization via MPI.
"""

import numpy as np
from mpi4py import MPI
from cmaes import CMA



class CMAES_MPI():
    """
    The CMA_MPI class implements Evolution Strategies using MPI (Message Passing Interface) for parallel computing.

    In particular, the ask-and-tell interface is replaced with a step function that takes an evaluation function as
    input, to be run on the various workers. The master node then collects the results and updates the model parameters
    accordingly.

    Note that due to the overheads in process-synchronization and communication, the parallel version may not always be
    faster. This is particularly useful if the eval_fn() is computationally expensive. If eval_fn() is fast, it may be
    better to run the optimizer without mpi (i.e., on a single core).

    TODO: it is possible to override the ask() method of the cmaes optimizer to use a shared noise table as with the
    ES_MPI class.
    """
    def __init__(self,
                 n_params,
                 population_size = None,
                 initial_parameters = None,
                 sigma = 0.1,
                 bounds = None, # np.ndarray
                 n_max_resampling = 100,
                 seed = None,
                 cov = None,
                 lr_adapt = False,
    ):
        """
        Initializes the CMA_MPI object with the given parameters.

        Most of the arguments are derived from the CMA-ES optimizer from the `cmaes' package:
        https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py

        Args:
            n_params (int): The number of parameters in the model.
            population_size (int, optional): The size of the population. Default is None (= automatic calculation).
                                            population_size = -1 automatically sets the population size to the number
                                            of available workers.
            initial_parameters (np.ndarray, optional): The initial mean of the distribution. Default is None,
                                                 corresponding to an initialization in the middle of the bounds,
                                                 if given, or else 0.
            sigma (float, optional): Initial standard deviation of covariance matrix.
            bounds (np.ndarray, optional): Lower and upper domain boundaries for each parameter (optional),
                                        with shape [n_params, 2].
            n_max_resampling (int, optional): A maximum number of resampling parameters (default: 100).
                                              If all sampled parameters are infeasible, the last sampled one
                                              will be clipped with lower and upper bounds.
            seed (int, optional): The seed for the random number generator. Default is None.
            cov (np.ndarray, optional): The initial covariance matrix. Default is None.
            lr_adapt (bool, optional): Whether to adapt the learning rate. Default is False.
        """

        self.n_params = n_params
        self.population_size = population_size

        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self.num_workers = self._comm.Get_size()
        self.is_master = self._rank==0

        if self.population_size == -1:
            self.population_size = self.num_workers

        if initial_parameters is None:
            if bounds is not None:
                self.initial_mean = np.mean(bounds, axis=1)
            else:
                self.initial_mean = np.zeros(n_params)
        else:
            self.initial_mean = initial_parameters

        self.optimizer = CMA(mean=self.initial_mean,
                             sigma=sigma,
                             bounds=bounds,
                             n_max_resampling=n_max_resampling,
                             seed=seed,
                             population_size=population_size,
                             cov=cov,
                             lr_adapt=lr_adapt)

        self.population_per_worker = self.population_size // self.num_workers

        if self.is_master:
            if self.population_size % self.num_workers != 0:
                print(f"\033[91m ERROR: population size {self.population_size} \
                      is not a multiple of the number of workers {self.num_workers}\033[0m")
            print(f"\033[93m Running {self.population_per_worker} perturbations on each worker, \
                  over {self.num_workers} workers. \033[0m")


    def step(self, eval_fn, **eval_fn_kwargs):
        """
            eval_fn: function that assesses the performance of a model passed as argument.
            The function is independent, and can rely on global variables
            (e.g., for datasets or RL environments)

            If master, returns all fitness values from all workers; else, an empty list is returned.
        """

        if self.is_master:
            sampled_parameters = [[] for _ in range(self.num_workers)]
            for i in range(self.population_size):
                sampled_parameters[i % self.num_workers].append( self.optimizer.ask() )
        else:
            sampled_parameters = []

        worker_sampled_params_list = self._comm.scatter(sampled_parameters, root=0)

        fitness_values = []
        for sample_param in worker_sampled_params_list:
            fitness_values.append( eval_fn(sample_param, **eval_fn_kwargs) )
        worker_results = self._comm.gather(fitness_values, root=0)

        all_fitnesses = []
        if self.is_master:
            all_params = [x for xs in sampled_parameters for x in xs]
            all_fitnesses = [x for xs in worker_results for x in xs]
            results = list(zip(all_params, all_fitnesses))
            self.optimizer.tell(results)

        return all_fitnesses

    def get_parameters(self):
        """
        Get current mean parameters of the optimizer.
        """
        return self.optimizer.mean

    def should_stop(self):
        """
        Returns whether the optimizer should stop, i.e., when a terminating condition is met.
        """
        return self.optimizer.should_stop()
