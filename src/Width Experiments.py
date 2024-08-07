# %% [markdown]
# ## Width experiments
# Investigate dynamic exponent of algorithms by measuring correlation time as a function of lattice width.
# 
# Change the `algorithm = {'metropolis', 'wolff', 'sw'}` variable to test the different algorithms.
# 
# Experimental parameters:
# * `algorithm = {'metropolis', 'wolff', 'sw'}`: change this to change the algorithm
# * `Ls: np.array`: range of lattice widths to be investigated
# * `runs_array: np.array`: number of runs for each $L$ value. Same length as `Ls`
# * `t_maxs: np.array`: total simulation time for each $L$ value. Same length as `Ls`
# * `eq_fraction: float`: the fraction of total simulation time used for equilibration, value between 0 and 1
# * `n_batches: int`: number of batches used for batching approach

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from importlib import reload

sys.path.append('../bin')
import mcmc
import helpers
import plotting

reload(mcmc)
reload(helpers)
reload(plotting);

# %% [markdown]
# ### Sampling using runs

# %%
algorithm = 'wolff'

Ls = np.array([5])
runs_array = np.ones(len(Ls)) * 50
t_maxs = Ls**2
eq_fraction = 0.6

# %%
data_points_runs = mcmc.width_experiment_runs(algorithm, Ls, t_maxs, eq_fraction, runs_array, savedata = True)

# %%
data_points_runs

# %% [markdown]
# ### Sampling using batching (not used in final results)

# %%
algorithm = 'wolff'
Ls = np.linspace(4, 150, 20, dtype=int)
eq_fraction = 0.6
t_maxs = np.ones(len(Ls)) * 10000
n_batches = 100

# %%
data_points_batched = mcmc.width_experiment_batch(algorithm, Ls, t_maxs, eq_fraction, n_batches, savedata = True)

# %%
data_points_batched


