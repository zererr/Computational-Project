import numpy as np
from datetime import datetime

def neighbouring_sites(s,L):
    '''
    Return the coordinates of the 4 sites adjacent to s on an L*L lattice with periodic boundary conditions.
    
    Code sampled from 2023 Budd
    '''
    return [((s[0]+1)%L, s[1]),
                ((s[0]-1)%L, s[1]),
                    (s[0], (s[1]+1)%L),
                        (s[0], (s[1]-1)%L)]

def lattice_magnetisation(lattice):
    ''' Returns magnetisation per lattice site'''
    return np.average(lattice)

def lattice_energy(lattice, betaJ):
    ''' Returns lattice energy multiplied by beta'''
    nearest_neighbours = np.roll(lattice, 1, axis=0) + np.roll(lattice, -1, axis=0) + \
                         np.roll(lattice, 1, axis=1) + np.roll(lattice, -1, axis=1)
    betaE = - betaJ * np.sum(lattice * nearest_neighbours)
    return betaE  

def FFT_autocorrelation(x):
    '''
    Computes autocorrelation trace of a time series x using FFT

    The autocorrelation has twice the length of x (c.f. footnote 7 in Chapter 3 of 1999 Newman)
    '''
    x_shifted = x - np.mean(x)
    x_shifted_padded = np.pad(x_shifted, (0, len(x_shifted)), 'constant')

    x_omega = np.fft.fft(x_shifted_padded)
    autocorr_omega = np.abs(x_omega)**2
    return np.fft.ifftn(autocorr_omega)

def estimate_correlation_time(autocorr):
    '''
    Estimates exponential correlation time (time index) of an autocorrelation trace
    
    Code sampled from 2023 Budd
    '''
    smaller = np.where(autocorr < np.exp(-1)*autocorr[0])[0]
    return smaller[0] if len(smaller) > 0 else len(autocorr)-1

def estimated_fft_corr_time(x):
    '''
    Composition of FFT_autocorrelation and estimate_correlation_time

    Returns the correlation time (time index) of a trace x
    '''
    auto_corr = FFT_autocorrelation(x).real
    return estimate_correlation_time(auto_corr[:len(auto_corr)//2])

def batch_estimate(data,observable,k):
    '''Divide data into k batches and apply the function observable to each. Returns the mean and standard error.
    
    Code sampled from 2023 Budd
    '''
    batches = np.reshape(data,(k,-1))
    values = np.apply_along_axis(observable, 1, batches)
    return np.mean(values), np.std(values)/np.sqrt(k-1)

def generate_timestamp():
    '''
    Returns a timestamp (Year / Month / Day / Hour / Minute / Second) used for labelling datasets
    This is useful for running simulations without worrying about overwriting previously saved data.
    However, the user will need to remove the timestamp for the plotting functions in plotting.py to work
    '''
    return datetime.now().strftime('%Y%m%d%H%M%S')