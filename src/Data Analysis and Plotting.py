# %% [markdown]
# ## Data Analysis and Plotting
# 
# This notebook is used for analysing the data collected from the experimental notebooks Temperature Experiments.ipynb and Width Experiments.ipynb. For Time Experiments.ipynb, the data collection and plotting is self-contained.

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
# ### Temperature plotting

# %%
# Metropolis
algorithm = 'metropolis'
L = 25

data_points = plotting.load_data(algorithm, L, 'temperature')

plotting.plot_critical_slowing_down(data_points, algorithm, L, savefig=False)

# Additional plots for average Wolff cluster size and scaled Wolff plots
if algorithm == 'wolff':
    plotting.plot_wolff_cluster_size(L, savefig=False)
    plotting.plot_scaled_wolff_sweeps(L, savefig=False)

# %% [markdown]
# ### Width plotting

# %%
algorithm = 'wolff'
data_points = plotting.load_data(algorithm, -1, 'width') #-1 because L is the variable here

# %%
m, c, m_err, c_err = plotting.estimate_dynamic_exponent(data_points, algorithm)
plotting.print_dynamic_exponent(m, m_err, algorithm)

# %%
plotting.plot_dynamic_exponent(data_points, algorithm, m, m_err, c, text_x = 2.5, text_y = 0, savefig=False)


