
## About

Compact implementation of Evolution Strategies using MPI, together with an MPI-based wrapper for the `cmaes' Python library (only parallelizing the function evaluations). This repository uses part of the code from OpenAI's implementation (https://github.com/openai/evolution-strategies-starter).

Example usage:
```python
from esmpi import ES_MPI

def eval_fn(x):
    time.sleep(0.01)
    return (x[0] - 3) ** 2 + (10 * (x[1] + 2)) ** 2

optimizer = ES_MPI(n_params=2, population_size=16, learning_rate=0.1, sigma=0.02)

for i in range(50):
    fit = optimizer.step(eval_fn)

    if optimizer.is_master == 0:
        print(f"{i}: fitness {np.mean(fit)}, current best params {optimizer.get_parameters()}\n")

```
