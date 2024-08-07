# %% [markdown]
# ## Temperature experiments
# Investigate critical slowing down by measuring correlation time and average cluster size as a function of temperature
# 
# Experimental parameters:
# * `algorithm = {'metropolis', 'wolff', 'sw'}`: change this to change the algorithm
# * `L: int`: lattice width
# * `betaJs: np.array`: range of temperatures to be investigated
# * `eq_fraction: float`: the fraction of total simulation time used for equilibration, value between 0 and 1
# * `t_maxs: np.array`: total simulation time for each $\beta J$ value. Same length as `betaJs`
# * `runs_array: np.array`: number of runs for each $\beta J$ value. Same length as `betaJs`
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

L = 25
betaJs = np.linspace(0.2,0.6,20)
eq_fraction = 0.8
t_maxs = np.ones(len(betaJs)) * L**2
runs_array = 20 * np.ones(len(betaJs))

# %%
data_points_runs = mcmc.temperature_experiment_runs(algorithm, L, betaJs, t_maxs, eq_fraction, runs_array, savedata=True)

# %%
data_points_runs

# %% [markdown]
# ### Separate code for computing average Wolff cluster size $\langle n\rangle$ and scaled Wolff correlation time

# %%
L = 25
betaJs = np.linspace(0.2,0.6,20)
eq_fraction = 0.8
t_maxs = np.ones(len(betaJs)) * L**2
runs_array = 20 * np.ones(len(betaJs))

# %%
data_points_wolff = mcmc.temperature_experiment_runs_wolff(L, betaJs, runs_array, savedata = True)

# %%
data_points_wolff

# %% [markdown]
# ### Sampling using batching (not used in final results)

# %%
algorithm = 'sw'
L = 25
betaJs = np.linspace(0.2,0.6,20)
eq_fraction = 0.8
t_maxs = np.ones(len(betaJs)) * 10000
n_batches = 100

# %%
data_points_batched = mcmc.temperature_experiment_batch(algorithm, L, betaJs, t_maxs, eq_fraction, n_batches, savedata=True)

# %%
data_points_batched


