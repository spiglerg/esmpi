
"""
Extension of stable-baselines3 VecNormalize to support MPI synchronization of running statistics.
"""

import numpy as np

from mpi4py import MPI

try:
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.running_mean_std import RunningMeanStd


    class RunningMeanStdNoOverflow(RunningMeanStd):
        """
        Patch-wrapper for RunningMeanStd to prevent overflow in the update_from_moments method.
        """
        def __init__(self, rms):
            self.mean = np.copy(rms.mean)
            self.var = np.copy(rms.var)
            self.count = np.copy(rms.count)

        def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            with np.errstate(over='raise'):
                try:
                    new_mean = self.mean + delta * batch_count / tot_count
                    m_a = self.var * self.count
                    m_b = batch_var * batch_count

                    # We calculate the products/divisions in an order that reduces the chance of overflow
                    tmp_mult = self.count / (self.count + batch_count)
                    tmp_mult2 = batch_count / (self.count + batch_count)
                    var_delta = np.square(delta) * tmp_mult * tmp_mult2

                    new_var = (m_a + m_b) / (self.count + batch_count) + \
                                var_delta

                    new_count = batch_count + self.count

                    self.mean = new_mean
                    self.var = new_var
                    self.count = new_count
                except FloatingPointError:
                    # This happens because self.count has gotten too large and the multiplication is overflowing
                    # We need to scale down the batch statistics
                    self.count /= 2
                    batch_count /= 2
                    self.update_from_moments(batch_mean, batch_var, batch_count)

    class MPIVecNormalize(VecNormalize):
        """
        Extension of VecNormalize to support MPI.

        A moving average, normalizing wrapper for vectorized environment.
        has support for saving/loading moving average,

        :param venv: the vectorized environment to wrap
        :param training: Whether to update or not the moving average
        :param norm_obs: Whether to normalize observation or not (default: True)
        :param norm_reward: Whether to normalize rewards or not (default: True)
        :param clip_obs: Max absolute value for observation
        :param clip_reward: Max value absolute for discounted reward
        :param gamma: discount factor
        :param epsilon: To avoid division by zero
        :param norm_obs_keys: Which keys from observation dict to normalize.
            If not specified, all keys will be normalized.
        """
        def __init__(
            self,
            venv,
            training = True,
            norm_obs = True,
            norm_reward = True,
            clip_obs = 10.0,
            clip_reward = 10.0,
            gamma = 0.99,
            epsilon = 1e-8,
            norm_obs_keys = None,
        ):
            VecNormalize.__init__(self,
                                  venv,
                                  training,
                                  norm_obs,
                                  norm_reward,
                                  clip_obs,
                                  clip_reward,
                                  gamma, epsilon,
                                  norm_obs_keys)
            if self.norm_reward:
                self.ret_rms = RunningMeanStdNoOverflow(self.ret_rms)
            if self.norm_obs:
                if isinstance(self.obs_rms, dict):
                    for k in self.obs_rms.keys():
                        self.obs_rms[k] = RunningMeanStdNoOverflow(self.obs_rms[k])
                else:
                    self.obs_rms = RunningMeanStdNoOverflow(self.obs_rms)

        def sync_statistics(self):
            """
            Broadcast the statistics from the master's env statistics to the workers.
            The master is the process with MPI rank 0.
            """

            # collect stats from all workers first
            env_stats = (self.obs_rms, self.ret_rms)
            env_stats = MPI.COMM_WORLD.gather(env_stats, root=0)
            if MPI.COMM_WORLD.Get_rank() == 0:
                for i in range(1, len(env_stats)):
                    if self.norm_obs:
                        if isinstance(self.obs_rms, dict):
                            for k in self.obs_rms.keys():
                                self.obs_rms[k].update_from_moments(env_stats[i][0][k].mean,
                                                                    env_stats[i][0][k].var,
                                                                    env_stats[i][0][k].count)
                        else:
                            self.obs_rms.update_from_moments(env_stats[i][0].mean,
                                                             env_stats[i][0].var,
                                                             env_stats[i][0].count)
                    if self.norm_reward:
                        self.ret_rms.update_from_moments(env_stats[i][1].mean,
                                                         env_stats[i][1].var,
                                                         env_stats[i][1].count)

            env_stats = (self.obs_rms, self.ret_rms)
            env_stats = MPI.COMM_WORLD.bcast(env_stats, root=0)
            self.obs_rms, self.ret_rms = env_stats

except ImportError:
    print('MPIVecNormalize not imported because stable-baselines3 is not installed.')
