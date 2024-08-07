import numpy as np
import helpers
import pandas as pd

from tqdm import tqdm
from collections import deque

rng = np.random.default_rng()

def metropolis(lattice, exponentials, L):
    '''
    Implements the Metropolis single flip algorithm for one MC step

    1. Select a random spin
    2. Compute energy difference between flipped and unflipped states
    3. Flip according to acceptance probability

    Uses memoization (pre-computing exponentials and storing in a dictionary) for Boltzmann factors
    This saves time if metropolis is called multiples times in one simulation since 
    the Boltzmann factors can only take five possible values (for coordination number z = 4)

    Mutates lattice
    '''
    
    site = tuple(rng.integers(0, L, 2))
    neighbours = helpers.neighbouring_sites(site, L)
    spin_sum = lattice[site] * sum([lattice[neighbour] for neighbour in neighbours])
    if spin_sum < 0 or rng.uniform() < exponentials[spin_sum]:
        lattice[site] *= -1

def wolff(lattice, p_add, L):
    '''
    Implements the Wolff cluster algorithm for one MC step

    1. Select an initial site at random (a.k.a the seed) and flip it
    2. Add parallel neighbours to the stack with probability p_add
    3. Now pull a site (in any order) from the stack and repeat step 2

    Mutates lattice and returns the size of the flipped cluster

    Code sampled from 2023 Budd
    '''

    seed = tuple(rng.integers(0, L, 2)) # Generate a random coordinate for the seed
    spin = lattice[seed] # Store seed spin to compare with neighbours
    lattice[seed] = -spin # Flip seed
    cluster_size = 1

    stack = deque([seed])
    while stack:
        site = stack.pop()
        neighbours = helpers.neighbouring_sites(site, L)
        for neighbour in neighbours:
            if lattice[neighbour] == spin and rng.uniform() < p_add:
                stack.appendleft(neighbour)
                lattice[neighbour] = -spin
                cluster_size += 1
    return cluster_size

def swendsen_wang(lattice, p_add, L):
    '''
    Implements the Swendsen Wang cluster algorithm for one MC step
    1. Cluster decomposition: partitioning the lattice into clusters
    2. Flip each cluster with probability 1/2

    Mutates lattice
    '''

    clusters = {}
    labels = np.zeros((L,L), dtype=int) # Initialize 

    # Step 1: Cluster Decomposition
    label = 1
    for i in range(L):
        for j in range(L):
            # If spin is not yet part of a cluster
            if labels[i,j] == 0:
                labels[i,j] = label # Make a new cluster
                spin = lattice[i,j] # Ask for the spin value for comparison with neighbours
                cluster = [(i,j)] # A list of tuples containing the coordinates of every spin in the cluster
                stack = deque([(i,j)]) # Initialize a frontier
                while stack:
                    site = stack.pop() # Explore
                    neighbours = helpers.neighbouring_sites(site, L)
                    for neighbour in neighbours:
                        if labels[neighbour] == 0 and lattice[neighbour] == spin and rng.uniform() < p_add:
                            stack.appendleft(neighbour) # Expand the frontier
                            cluster.append(neighbour) # Add the neighbour to the cluster
                            labels[neighbour] = label # Labelling the neighbour so future iterations do not consider it
                # Frontier of (i,j) has been explored, so store into clusters dictionary
                clusters[label] = cluster
                label += 1 # Increment for a new cluster to be stored
        
    # Step 2: Flip each cluster with probability 1/2
    for label, cluster in clusters.items():
        if rng.uniform() < 1/2:
            # Fancy indexing of lattice to flip spins
            row_indices, col_indices = zip(*cluster)
            lattice[row_indices, col_indices] = -lattice[row_indices, col_indices]

def time_experiment(algorithm, L, betaJs, t_maxs, savedata = False):
    '''
    Simulates the 2D Ising model for a given algorithm, lattice size and range of temperatures

    Returns magnetisation, energy and lattice time series (measured over the whole simulation) collated in a temperature indexed DataFrame

    Although this DataFrame can be saved to a csv file, the traces are unfortunately not stored due to a size limit of Series objects.
    Hence, the user will need to run the simulation each time they wish to plot the traces.
    '''
    switch = {'metropolis': metropolis, 'wolff': wolff, 'sw': swendsen_wang}
    mc_step = switch[algorithm]

    # 1. Initialize DataFrame    
    magnetisation_trace_array = [pd.Series(index = np.arange(t_max), dtype = float) for t_max in t_maxs]
    energy_trace_array = [pd.Series(index = np.arange(t_max), dtype = float) for t_max in t_maxs]
    lattice_array = [pd.Series(index = np.arange(t_max), dtype = object) for t_max in t_maxs]

    data_points = pd.DataFrame(index = pd.Index(betaJs, name = 'betaJ'), 
                                data = {'simulation time': t_maxs, 'magnetisation trace': magnetisation_trace_array, 
                                        'energy trace': energy_trace_array, 'lattice trace': lattice_array})    
    
    # 2. Simulation
    for betaJ, params in data_points.iterrows():  
        magnetisation_trace = params['magnetisation trace']
        energy_trace = params['energy trace']
        lattice_trace = params['lattice trace']
        t_max = params['simulation time']

        if algorithm == 'metropolis':
            mc_arg = {i:np.exp(-2 * betaJ * i) for i in range(-4, 5, 2)}
        else:
            mc_arg = 1 - np.exp(-2*betaJ)

        # Initialize a new random lattice for a new temperature
        lattice = np.random.choice([-1, 1], size=(L, L))
        for t in tqdm(np.arange(t_max, dtype=int)):
            # Collect data
            magnetisation_trace[t] = helpers.lattice_magnetisation(lattice)
            energy_trace[t] = helpers.lattice_energy(lattice, betaJ)
            lattice_trace[t] = lattice.copy() # Store a copy because lattice is mutable
            
            if algorithm == 'metropolis':
                # Perform one MC sweep
                for _ in np.arange(L**2):
                    mc_step(lattice, mc_arg, L)
            else:
                # Perform one MC step (lattice flip)
                mc_step(lattice, mc_arg, L)
            
    # 3. Post-processing    
    data_points['absolute magnetisation trace'] = np.abs(data_points['magnetisation trace'])

    # 4. Save data points
    if savedata:
        timestamp = helpers.generate_timestamp()
        file_name = '../data/' + algorithm + '/' + f'{timestamp} - ' + algorithm + f' time data - random lattice, L={L}.csv'
        data_points.to_csv(file_name)
    
    return data_points

def temperature_experiment_runs(algorithm, L, betaJs, t_maxs, eq_fraction, runs_array, savedata = False):
    '''
    Simulates the 2D Ising model for a given algorithm, lattice size and range of temperatures

    Samples the correlation time for multiples runs for each temperature betaJ, then computes mean and standard deviation

    The intent here is to sample small widths (max(Ls)<= 25) for a large number of runs (min(runs_array) > 10), balancing out computation time.

    Example usage:
        temperature_experiment_runs('metropolis', 25, [0.1, 0.441, 10], np.ones(3) * 1000, 0.8, [10, 50, 10], savedata = True)
        temperature_experiment_runs('wolff', 25, [0.1, 0.441, 10], np.ones(3) * 1000, 0.6, [10, 50, 10], savedata = True)

    if savedata = True, the user will need to remove the timestamp manually for the plotting functions to work.
    '''
    assert algorithm in ['metropolis', 'sw', 'wolff']

    # 1. Initialization of simulation periods and temperature indexed DataFrame (for data collection)
    equilibration_periods = t_maxs * eq_fraction
    sampling_periods = t_maxs - equilibration_periods
    

    corr_times_array = [pd.Series(index = np.arange(runs), dtype = float) for runs, _ in zip(runs_array, betaJs)]    
    corr_time_column = {'metropolis': 't_|m|^(M)', 'sw': 't_|m|^(SW)', 'wolff': 't_|m|^(W)'}[algorithm]
    data_points = pd.DataFrame(index = pd.Index(betaJs, name = 'betaJ'),
                            data = {'t_max': t_maxs, 't_eq': equilibration_periods, 't_sample': sampling_periods,
                                    'runs': runs_array, 'corr_times': corr_times_array, corr_time_column: None})
    # 2. Simulation
    lattice = np.random.choice([-1, 1], size=(L, L))
    for betaJ, params in tqdm(data_points.iterrows()):
        equilibration_period = params['t_eq']
        sampling_period = params['t_sample']  
        runs = params['runs']
        corr_times = params['corr_times']      

        sample_corr_time(algorithm, L, lattice, betaJ, runs, equilibration_period, sampling_period, corr_times = corr_times)
                                
    # 3. Post-processing
    std_column = corr_time_column + ' sample std'
    data_points[corr_time_column] = data_points['corr_times'].apply(np.mean)
    data_points[std_column] = data_points['corr_times'].apply(lambda x: np.std(x, ddof=1))
    
    # 4. Save data points
    if savedata:
        timestamp = helpers.generate_timestamp()        
        file_name = '../data/' + algorithm + '/' + f'{timestamp} - ' + algorithm + f' temperature data - random lattice, runs, L={L}.csv'
        data_points.to_csv(file_name)

    return data_points 

def temperature_experiment_batch(algorithm, L, betaJs, t_maxs, eq_fraction, n_batches, savedata = False):
    '''
    Simulates the 2D Ising model for a given algorithm, lattice size and range of temperatures

    Samples the correlation time for one run for each temperature betaJ, then computes mean and standard deviation via batching

    The intent here is to sample large widths (L > 25) for one long run (such that equilibrium is reached) then simulating multiple pseudo-runs by batching

    Example usage:
        temperature_experiment_batch('metropolis', 25, np.array([0.01, 0.441, 10]), np.ones(3) * 1000, 0.8, 100, savedata = False)
        temperature_experiment_batch('wolff', 25, np.array([0.01, 0.441, 10]), np.ones(3) * 1000, 0.8, 100, savedata = False)        
    '''
    assert algorithm in ['metropolis', 'sw', 'wolff']

    # 1. Initialization of simulation periods and width indexed DataFrame (for data collection)
    equilibration_periods = t_maxs * eq_fraction
    sampling_periods = t_maxs - equilibration_periods
    assert np.all(sampling_periods % n_batches == 0) # Ensure that sampling periods are divisible by number of batches

    corr_time_column = {'metropolis': 't_|m|^(M)', 'sw': 't_|m|^(SW)', 'wolff': 't_|m|^(W)'}[algorithm]
    std_column = corr_time_column + ' sample std'
    data_points = pd.DataFrame(index = pd.Index(betaJs, name = 'betaJ'),
                            data = {'t_max': t_maxs, 't_eq': equilibration_periods, 't_sample': sampling_periods, corr_time_column: None})

    # 2. Simulation    
    lattice = np.random.choice([-1, 1], size=(L, L))
    for betaJ, params in tqdm(data_points.iterrows()):        
        equilibration_period = params['t_eq']
        sampling_period = params['t_sample']

        # Sample one long run
        abs_mags = sample_corr_time(algorithm, L, lattice, betaJ, 1, equilibration_period, sampling_period, corr_times = None)

        mean, std = helpers.batch_estimate(abs_mags.values, helpers.estimated_fft_corr_time, n_batches)
        data_points.loc[betaJ, corr_time_column] = mean
        data_points.loc[betaJ, std_column] = std

    # 3. Save data points
    if savedata:
        timestamp = helpers.generate_timestamp()         
        file_name = '../data/' + algorithm + '/' + f'{timestamp} - ' + algorithm + f' temperature data - random lattice, batched, L={L}.csv'
        data_points.to_csv(file_name)

    return data_points

def width_experiment_runs(algorithm, Ls, t_maxs, eq_fraction, runs_array, savedata = False):
    '''
    Simulates the 2D Ising model for a given algorithm and range of widths, above the critical temperature

    Samples the correlation time for multiples runs for each lattice width L, then computes mean and standard deviation

    The intent here is to sample small widths (max(Ls)<= 25) for a large number of runs (min(runs_array) > 10), balancing out computation time.

    Example usage:
        width_experiment_runs('metropolis', np.array([2,4,8,16]), np.array([2000, 4000, 8000, 16000]), 0.6, [100,75,50,25], savedata = True)
        width_experiment_runs('wolff', np.array([5,10,15]), np.array([10000, 10000, 10000, 10000]), 0.8, [50,50,50], savedata = False)
    
        if savedata = True, the user will need to remove the timestamp manually for the plotting functions to work.
    '''
    assert algorithm in ['metropolis', 'sw', 'wolff']

    # 1. Initialization of simulation periods and width indexed DataFrame (for data collection)
    equilibration_periods = t_maxs * eq_fraction
    sampling_periods = t_maxs - equilibration_periods

    betaJ = np.log(1+np.sqrt(2))/2 - 0.01 # Work above T_c to use z = z_steps + gamma / nu - d
    
    corr_times_array = [pd.Series(index = np.arange(runs), dtype = float) for runs in runs_array]    
    corr_time_column = {'metropolis': 't_|m|^(M)', 'sw': 't_|m|^(SW)', 'wolff': 't_|m|^(W)'}[algorithm]
    data_points = pd.DataFrame(index = pd.Index(Ls, name = 'L'),
                            data = {'t_max': t_maxs, 't_eq': equilibration_periods, 't_sample': sampling_periods,
                                    'runs': runs_array, 'corr_times': corr_times_array, corr_time_column: None})

    # 2. Simulation    
    for L, params in tqdm(data_points.iterrows()):
        lattice = np.random.choice([-1, 1], size=(L, L)) # Use the same lattice across runs

        equilibration_period = params['t_eq']
        sampling_period = params['t_sample'] 
        runs = params['runs']
        corr_times = params['corr_times']

        sample_corr_time(algorithm, L, lattice, betaJ, runs, equilibration_period, sampling_period, corr_times)
                                
    # 3. Post-processing
    std_column = corr_time_column + ' sample std'
    data_points[corr_time_column] = data_points['corr_times'].apply(np.mean)
    data_points[std_column] = data_points['corr_times'].apply(lambda x: np.std(x, ddof=1))
    
    # 4. Save data points
    if savedata:
        timestamp = helpers.generate_timestamp()        
        file_name = '../data/' + algorithm + '/' + f'{timestamp} - ' + algorithm + f' width data - random lattice, runs.csv'
        data_points.to_csv(file_name)

    return data_points

def width_experiment_batch(algorithm, Ls, t_maxs, eq_fraction, n_batches, savedata = False):
    '''
    Simulates the 2D Ising model for a given algorithm and range of widths, above the critical temperature

    Samples the correlation time for one run for each lattice width L, then computes mean and standard deviation via batching

    The intent here is to sample large widths (L > 25) for one long run (such that equilibrium is reached) then simulating multiple pseudo-runs by batching

    Example usage:
        width_experiment_batch('metropolis', np.array([24,48,72,96]), np.array([5000,5000,5000,5000]), 0.8, 100, savedata = False)
        width_experiment_batch('wolff', np.array([24,48,72,96]), np.array([2000,3000,4000,5000]), 0.8, 100, savedata = False)
        
    '''
    assert algorithm in ['metropolis', 'sw', 'wolff']

    # 1. Initialization of simulation periods and width indexed DataFrame (for data collection)
    equilibration_periods = t_maxs * eq_fraction
    sampling_periods = t_maxs - equilibration_periods
    assert np.all(sampling_periods % n_batches == 0) # Ensure that sampling periods are divisible by number of batches

    betaJ = np.log(1+np.sqrt(2))/2 - 0.01 # Work above T_c to use z = z_steps + gamma / nu - d
    corr_time_column = {'metropolis': 't_|m|^(M)', 'sw': 't_|m|^(SW)', 'wolff': 't_|m|^(W)'}[algorithm]
    std_column = corr_time_column + ' sample std'
    data_points = pd.DataFrame(index = pd.Index(Ls, name = 'L'),
                            data = {'t_max': t_maxs, 't_eq': equilibration_periods, 't_sample': sampling_periods, corr_time_column: None})

    # 2. Simulation    
    for L, params in tqdm(data_points.iterrows()):
        lattice = np.random.choice([-1, 1], size=(L, L))
        equilibration_period = params['t_eq']
        sampling_period = params['t_sample']

        # Sample one long run
        abs_mags = sample_corr_time(algorithm, L, lattice, betaJ, 1, equilibration_period, sampling_period, corr_times = None)

        mean, std = helpers.batch_estimate(abs_mags.values, helpers.estimated_fft_corr_time, n_batches)
        data_points.loc[L, corr_time_column] = mean 
        data_points.loc[L, std_column] = std

    # 3. Save data points
    if savedata:
        timestamp = helpers.generate_timestamp()         
        file_name = '../data/' + algorithm + '/' + f'{timestamp} - ' + algorithm + f' width data - random lattice, batched.csv'
        data_points.to_csv(file_name)

    return data_points 

def sample_corr_time(algorithm, L, lattice, betaJ, runs, equilibration_period, sampling_period, corr_times = None):
    '''
    Populates (mutates) correlation time array by sampling from a equilibrated lattice

    Mutates the lattice as algorithm is applied

    Conditional return value:
        If corr_times is provided, assumes the user wishes to average the correlation time over runs. Return value is None
        Otherwise, assumes the user wishes to average the correlation time over batches from one run. Return value is abs_mags
    '''
    mc_step = {'metropolis': metropolis, 'wolff': wolff, 'sw': swendsen_wang}[algorithm]

    if algorithm == 'metropolis':
        mc_arg = {spin_sum:np.exp(-2 * betaJ * spin_sum) for spin_sum in range(-4, 5, 2)}
    else:
        mc_arg = 1 - np.exp(-2*betaJ)
    
    for i in tqdm(np.arange(runs, dtype=int)):
        abs_mags = pd.Series(index = pd.Index(np.arange(sampling_period)), dtype = float)
        multiplier = L**2 if algorithm == 'metropolis' else 1

        for _ in np.arange(equilibration_period * multiplier, dtype = int):
            mc_step(lattice, mc_arg, L)
        
        for t in np.arange(sampling_period, dtype = int):
            abs_mags.loc[t] = helpers.lattice_magnetisation(lattice)
            for _ in np.arange(multiplier, dtype = int):
                mc_step(lattice, mc_arg, L)
        
        # Convert the magnetisations to absolute values with vectorization
        abs_mags = np.abs(abs_mags)        

        # For run averaging experiments
        if corr_times is not None:
            corr_times[i] = abs_mags.index[helpers.estimated_fft_corr_time(abs_mags)]
            
        # For batching experiments
        else:
            return abs_mags

def temperature_experiment_runs_wolff(L, betaJs, runs_array, savedata = False):
    '''
    Deprecated as I prioritised code cleanliness over redundancy. Ideally, I would integrate these into the code base above, but have not done so due to lack of time.

    In addition to the unscaled correlation time, this function computes average cluster sizes and scaled correlation time of the Wolff algorithm
    '''
    # Initialization
    N = L**2
    t_max = N
    equilibration_period = N * 0.8
    sampling_period = t_max - equilibration_period

    corr_times_array = [pd.Series(index = np.arange(runs), dtype = float) for runs, _ in zip(runs_array, betaJs)]
    average_cluster_sizes_array = [pd.Series(index = np.arange(runs), dtype = float) for runs, _ in zip(runs_array, betaJs)]
    data_points = pd.DataFrame(index = pd.Index(betaJs, name = 'betaJ'), 
                               data = {'runs': runs_array, 'corr_times': corr_times_array, 'average_cluster_sizes': average_cluster_sizes_array,'t_|m|^(W)': None})    
    
    # Simulation
    lattice = np.random.choice([-1, 1], size=(L, L))
    for betaJ, params in tqdm(data_points.iterrows()):
        ## Unpack parameters for each betaJ
        corr_times = params['corr_times']
        runs = params['runs']
        p_add = 1 - np.exp(-2*betaJ)
        average_cluster_sizes = params['average_cluster_sizes'] 

        for i in np.arange(runs, dtype=int):            
            abs_mags = pd.Series(index = pd.Index(np.arange(sampling_period)), dtype = float)        
            total_flips = 0
            
            # Equilibrate
            for _ in np.arange(equilibration_period, dtype=int):                
                wolff(lattice, p_add, L)

            # Sampling
            for t in np.arange(sampling_period, dtype=int):
                abs_mags.loc[t] = np.abs(helpers.lattice_magnetisation(lattice))                
                total_flips += wolff(lattice, p_add, L)
                                    
            # Compute correlation time and <n> for one run
            abs_mag_autocorr = helpers.FFT_autocorrelation(abs_mags).real
            corr_times[i] = helpers.estimate_correlation_time(abs_mag_autocorr)
            average_cluster_sizes[i] = total_flips / sampling_period            
            
        # Average correlation time and mean cluster size across runs       
        data_points.loc[betaJ, 't_|m|^(W)'] = corr_times.mean()
        data_points.loc[betaJ, '<n>'] = average_cluster_sizes.mean()        

    # Post-processing
    data_points['<n>/N'] = data_points['<n>'] / N
    data_points['t_|m|^(M)'] = data_points['t_|m|^(W)'] * data_points['<n>/N']
    data_points['t_|m|^(W) sample std'] = data_points['corr_times'].apply(lambda x: np.std(x, ddof=1))
    data_points['<n> sample std'] = data_points['average_cluster_sizes'].apply(lambda x: np.std(x, ddof=1))

    tW_frac_error = data_points['t_|m|^(W) sample std'] / data_points['t_|m|^(W)'] 
    n_frac_error = data_points['<n> sample std'] / data_points['<n>'] 
    data_points['t_|m|^(M) error'] = (tW_frac_error**2 + n_frac_error**2)**0.5 * data_points['t_|m|^(M)']
    
    # Save data points
    if savedata:
        timestamp = helpers.generate_timestamp()        
        file_name = '../data/wolff/' + f'{timestamp} -' + f' wolff temperature data - random lattice, L={L}.csv'
        data_points.to_csv(file_name)

    return data_points