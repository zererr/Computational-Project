# %% [markdown]
# ## Time Experiments
# Investigate the time evolution of the 2D Ising Model. 
# 
# Experimental parameters:
# * `algorithm` $\in$ `{'metropolis', 'wolff', 'sw'}`: change this to change the algorithm
# * `L: int`: lattice width
# * `N: int`: number of lattice sites $=L^2$
# * `betaJs: np.array`: range of temperatures to be investigated
# * `t_maxs: np.array`: total simulation time for each $\beta J$ value. Same length as `betaJs`

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
# ### Time evolution of Ising Model at different temperatures

# %%
algorithm = 'metropolis'

L = 4
N = L ** 2
betaJs = np.array([0.01, 0.440, 10])
t_maxs = np.ones(len(betaJs)) * N

assert(len(betaJs) > 1 and len(t_maxs) > 1) # Plot functions below require more than one temperature

data_points = mcmc.time_experiment(algorithm, L, betaJs, t_maxs, savedata = False)

# %%
# Plot lattice evolution
plot_times = np.concatenate((np.array([0]), np.logspace(0, np.log10(N-1), 5, dtype=int)))

plotting.plot_lattice_evolution(algorithm, L, betaJs, plot_times, data_points, savefig = False)

# %%
# Plot magnetisation traces
plotting.plot_trace(algorithm, L, 'magnetisation', 'm', data_points, savefig = False)

# %%
# Plot absolute magnetisation traces
plotting.plot_trace(algorithm, L, 'absolute magnetisation', '|m|', data_points, savefig = False)

# %%
# Plot energy traces
plotting.plot_trace(algorithm, L, 'energy', '$\\beta E$', data_points, savefig = False)


