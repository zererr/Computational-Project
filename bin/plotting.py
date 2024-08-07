import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def plot_lattice(lattice, ax, title):
    '''Plots the 2D lattice. Code sampled from 2023 Budd'''
    ax.matshow(lattice, vmin=-1, vmax=1, cmap=plt.cm.binary)
    ax.title.set_text(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

def plot_lattice_evolution(algorithm, L, betaJs, plot_times, data, savefig = False):
    '''
    Given a temperature indexed DataFrame containing time data, plots the lattice evolution for each temperature

    Possible values for algorithm: 'metropolis', 'wolff', 'sw'

    Notes:
        No regular expressions matching.
    '''
    assert algorithm in ['metropolis', 'wolff', 'sw']

    fig_title = algorithm + f' lattice evolution, L={L}'
    fig_title = fig_title[0].upper() + fig_title[1:] # Capitalize the first character    
    fig, axs = plt.subplots(len(betaJs), len(plot_times), figsize = (12,6))
    fig.suptitle(fig_title, fontsize = 16)

    for axrow, (betaJ, params) in zip(axs, data.iterrows()):
        lattice_trace = params['lattice trace']
        
        axrow[0].set_ylabel(f'$\\beta J$ = {betaJ}', rotation = 0, labelpad = 30)

        for ax, plot_time in zip(axrow, plot_times):
            plot_lattice(lattice_trace.loc[plot_time], ax, f't={plot_time}')

    fig.tight_layout()

    if savefig:
        file_name = '../results/' + algorithm + '/' + algorithm + f' lattice evolution - L={L}.pdf'
        plt.savefig(file_name)

def plot_trace(algorithm, L, quantity, ylabel, data, savefig = False):
    '''
    Given a temperature indexed DataFrame containing time data, plot the trace for a given quantity for each temperature
    Possible values for algorithm: 'metropolis', 'wolff', 'sw'
    Possible values for quantity: 'magnetisation', 'energy', 'absolute magnetisation'

    Notes:
        No regular expressions matching.
    '''
    assert algorithm in ['metropolis', 'wolff', 'sw']
    assert quantity in ['magnetisation', 'energy', 'absolute magnetisation']
    
    fig, axes = plt.subplots(len(data), 1, figsize = (10,6))
    
    string = quantity + ' trace'
    fig_title = algorithm + ' ' + quantity + f' traces at different temperatures, L={L}'
    fig_title = fig_title[0].upper() + fig_title[1:] # Capitalize the first character

    for ax, (betaJ, params) in zip(axes,data.iterrows()):
        quantity_trace = params[string].copy() # Make a copy so we do not mutate data
        quantity_trace.plot(ax=ax, xlabel='t / sweeps', title=f'$\\beta J$ = {betaJ}')
        ax.set_ylabel(ylabel, rotation = 0, labelpad = 15)

    fig.suptitle(fig_title)
    fig.tight_layout()    

    if savefig:
        file_name = '../results/' + algorithm + '/' + algorithm + ' ' + quantity + f' traces - L={L}.pdf'
        plt.savefig(file_name)

def load_data(algorithm, L, variable):
    '''
    Loads temperature or width datasets collected from experiments

    Notes:
        Metropolis time datasets are too big, so for uniformity across algorithms I have decided not to include functionality for time datasets.
        This only works for datasets without timestamps (a mistake I realised only after implementing it)    
        No regular expressions matching.
    '''

    assert algorithm in ['metropolis', 'wolff', 'sw']
    assert variable in ['temperature', 'width']

    if variable == 'temperature':
        file_name = '../data/' + algorithm + '/' + algorithm + ' ' + variable +  f' data - random lattice, runs, L={L}.csv'
        data = pd.read_csv(file_name, index_col='betaJ')
    elif variable == 'width':
        file_name = '../data/' + algorithm + '/' + algorithm + ' ' + variable +  f' data - random lattice, runs.csv'
        data = pd.read_csv(file_name, index_col='L')

        # Additional processing
        y, yerr = {'metropolis': ('t_|m|^(M)', 't_|m|^(M) sample std'), 
                   'wolff': ('t_|m|^(W)', 't_|m|^(W) sample std'), 
                   'sw': ('t_|m|^(SW)', 't_|m|^(SW) sample std')}[algorithm]
        ln_y = 'ln ' + y
        ln_y_err = ln_y + ' error'

        data['ln L'] = np.log(data.index)
        data[ln_y] = np.log(data[y])
        data[ln_y_err] = data[yerr] / data[y]
                
    return data

def plot_critical_slowing_down(data, algorithm, L, savefig = False):
    '''
    Given a temperature indexed DataFrame containing correlation times, plots correlation time against temperature. For the Wolff algorithm, this is the unscaled correlation time.
    '''
    assert algorithm in ['metropolis', 'wolff', 'sw']
    fig_title = 'Absolute magnetisation correlation time (' + algorithm[0].upper() + algorithm[1:] + f') against $\\beta J$, {L=}'
    
    y, ylabel, yerr = {'metropolis': ('t_|m|^(M)','$\\tau_{|m|}^{(M)}$', 't_|m|^(M) sample std'), 
                 'wolff': ('t_|m|^(W)', '$\\tau_{|m|}^{(W)}$', 't_|m|^(W) sample std'), 
                 'sw': ('t_|m|^(SW)', '$\\tau_{|m|}^{(SW)}$', 't_|m|^(SW) sample std')}[algorithm]

    data.plot(y=y, style='x', legend = False, figsize = (10,6), title = fig_title)
    plt.errorbar(x=data.index, y=data[y], yerr=data[yerr], fmt='none', color='k', alpha=0.8, capsize=5)
    plt.axvline(0.441, color='r', linestyle='--')
    plt.ylabel(ylabel, rotation = 0, labelpad=20, fontsize=14)
    plt.xlabel('$\\beta J$', fontsize = 14)    
    
    if savefig:
        file_name = '../results/' + algorithm + '/' + algorithm + f' critical slowing down - random lattice, L={L}.pdf'
        plt.savefig(file_name)    

def plot_wolff_cluster_size(L, savefig=False):
    '''Plots Wolff cluster size as a function of temperature'''

    data = load_data('wolff', L, 'temperature')
    fig_title = f'Mean cluster size against $\\beta J$, L={L}'

    data.plot(y='<n>/N', figsize = (10,6), legend = False, title = fig_title)
    plt.axvline(0.441, color='r', linestyle='--', label='Critical temperature')
    plt.ylabel('$\\frac{\\langle n \\rangle}{N}$', rotation = 0, labelpad=15, fontsize = 14)
    plt.xlabel('$\\beta J$', fontsize = 14)
    plt.yticks(np.arange(0,1.2,0.2))

    if savefig:
        file_name = f'../results/wolff/wolff cluster size - random lattice, L={L}.pdf'
        plt.savefig(file_name) 

def plot_scaled_wolff_sweeps(L, savefig=False):
    '''Plots scaled Wolff correlation time as a function of temperature'''

    data = load_data('wolff', L, 'temperature')
    fig_title = f'Absolute magnetisation correlation time (scaled Wolff) against $\\beta J$, {L=}'

    data.plot(y='t_|m|^(M)', style='x', legend = False, figsize = (10,6), title = fig_title)
    plt.errorbar(x=data.index, y=data['t_|m|^(M)'], yerr=data['t_|m|^(M) error'], fmt='none', color='k', alpha=0.8, capsize=5)
    plt.axvline(0.441, color='r', linestyle='--')
    plt.ylabel('Scaled \n $\\tau_{|m|}^{(W)}$', rotation = 0, labelpad=20, fontsize = 12)
    plt.xlabel('$\\beta J$', fontsize = 14)

    if savefig:
        file_name = f'../results/wolff/scaled wolff critical slowing down - random lattice, L={L}.pdf'
        plt.savefig(file_name) 

def plot_dynamic_exponent_scatter(data, algorithm):
    '''Plot scatter plot of ln t against ln L'''
    assert algorithm in ['metropolis', 'wolff', 'sw']
    fig_title = 'Absolute magnetisation correlation time (' + algorithm[0].upper() + algorithm[1:] + f') against lattice width'
    
    y, ylabel, yerr = {'metropolis': ('ln t_|m|^(M)','ln $\\tau_{|m|}^{(M)}$', 'ln t_|m|^(M) error'), 
                       'wolff': ('ln t_|m|^(W)', 'ln $\\tau_{|m|}^{(W)}$', 'ln t_|m|^(W) error'), 
                       'sw': ('ln t_|m|^(SW)', 'ln $\\tau_{|m|}^{(SW)}$', 'ln t_|m|^(SW) error')}[algorithm]

    fig, ax = plt.subplots(figsize = (10,6))
    plt.scatter('ln L', y, data=data, marker='x')
    plt.errorbar(x=data['ln L'], y=data[y], yerr=data[yerr], fmt='none', color='k', alpha=0.8, capsize=5)    
    plt.ylabel(ylabel, rotation = 0, labelpad=25, fontsize=14)
    plt.xlabel('ln L', fontsize = 14)
    plt.title(fig_title)

def estimate_dynamic_exponent(data, algorithm):
    '''Performs linear least squares fit to width data to extract dynamic exponent and uncertainty'''

    assert algorithm in ['metropolis', 'wolff', 'sw']
    y, yerr = {'metropolis': ('ln t_|m|^(M)', 'ln t_|m|^(M) error'), 
                    'wolff': ('ln t_|m|^(W)', 'ln t_|m|^(W) error'), 
                    'sw': ('ln t_|m|^(SW)', 'ln t_|m|^(SW) error')}[algorithm]           

    popt, pcov = curve_fit(lambda x,m,c: m*x + c , data['ln L'], data[y], sigma=data[yerr])
    m, m_err = popt[0], np.sqrt(np.diag(pcov))[0]
    c, c_err = popt[1], np.sqrt(np.diag(pcov))[1]

    return m, c, m_err, c_err

def print_dynamic_exponent(m, m_err, algorithm):
    '''Given gradient of regression line and its error, prints out the dynamic exponent for the algorithm'''
    assert algorithm in ['metropolis', 'wolff', 'sw']

    if algorithm == 'wolff':
        # Return actual dynamic exponent
        gamma, nu, d = 7/4, 1, 2
        m = m + gamma/nu - d

    print('Dynamic exponent of ' + algorithm, f'{m:.2f} +- {m_err:.2f}')
    
def plot_dynamic_exponent_regression_line(data, m, m_err, c, text_x, text_y):
    '''Plot regression line of ln t against ln L'''
    xs = np.linspace(data['ln L'].iloc[0], data['ln L'].iloc[-1], 25)
    ys = m * xs + c
    plt.plot(xs, ys, 'r-')
    plt.text(text_x, text_y, f'Gradient = {m:.2f} +- {m_err:.2f}', fontsize = 14)

def plot_dynamic_exponent(data, algorithm, m, m_err, c, text_x, text_y, savefig=False):
    '''Plots scatter plot and regression line of ln t against L together'''
    plot_dynamic_exponent_scatter(data, algorithm)
    plot_dynamic_exponent_regression_line(data, m, m_err, c, text_x, text_y)

    if savefig:
        file_name = '../results/' + algorithm + '/' + algorithm + f' dynamic exponent - random lattice.pdf'
        plt.savefig(file_name) 