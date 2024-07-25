"""
ES-MPI Python Library by Giacomo Spigler, 2024.
http://www.spigler.net/giacomo

Part of the code (SGD/Adam optimizers and SharedNoiseTable) are adapted from OpenAI's ES code:
  https://github.com/openai/evolution-strategies-starter
"""

from .es_mpi import ES_MPI, util_torch_set_parameters, util_torch_get_parameters, \
                   compute_centered_ranks, compute_identity_ranks

try:
    from .mpi_vecnormalize import MPIVecNormalize
    mpivecnormalize_imported = True
except ImportError:
    mpivecnormalize_imported = False
    print('\033[91mProblem importing stable-baselines3. MPIVecNormalize will not be available.\033[0m')

try:
    from .cmaes_mpi import CMAES_MPI
    cmaes_imported = True
except ImportError:
    cmaes_imported = False
    print('\033[91mProblem importing cmaes. CMA_MPI will not be available.\033[0m')

__version__ = "1.0"

__all__ = ['ES_MPI',
           'util_torch_set_parameters',
           'util_torch_get_parameters',
           'compute_centered_ranks',
           'compute_identity_ranks',
           '__version__']
if mpivecnormalize_imported:
    __all__ += ['MPIVecNormalize']
if cmaes_imported:
    __all__ += ['CMAES_MPI']
