import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm



def history_score_probability(energies, weights):
    #
    # 10-bins per decade
    #
    bins = np.logspace(-20, 10, 10*(30)+1)
    history_score, bins = np.histogram(weights, bins)
    x = (bins[1:] + bins[:-1]) /2
    y = history_score/np.diff(bins)
    plt.step(x, y)
    plt.xscale('log')
    plt.yscale('log')    
    
    return x, y 

# Define the function to fit
def func_1_over_sqrtN(x, A):
        return A/np.sqrt(x)

def plot_statistics_variation(events, weights, sample_sizes, random_sampling =False):
    means = []
    variances = []
    var_variances = []
    std    = []
    #
    # "True" Observed Mean and variance
    #
    true_mean = np.average(events, weights=weights)
    
    #  Sarndal et al. (1992) (also presented in Cochran 1977),
    #
    true_variance = np.average((events*weights - true_mean)**2)
    #  Sarndal et al. (1992) (also presented in Cochran 1977),
    #
    true_variance_of_variance = np.average( ((events*weights - true_mean) - true_variance)**2) 
    
    
    true_std = np.sqrt(true_variance)
    
    
    for sample_size in tqdm(sample_sizes):
        if random_sampling:
            subsample_indices = np.random.choice(np.arange(len(events)), sample_size, replace=True)
        else:
            # Ensure sample_size does not exceed the number of events
            if sample_size > len(events):
                sample_size = len(events) 
                print("sample_size exceeds the number of events")
            # Start from 1, adjust to Python's 0-based indexing by subtracting 1
            subsample_indices = np.arange(0, sample_size)
        subsample = events[subsample_indices]
        subsample_weights = weights[subsample_indices]
        
        #
        # Mean of Subsample
        #
        mean = np.average(subsample, weights=subsample_weights)
        variance = np.average((subsample*subsample_weights - mean)**2) 
        var_variance = np.average( ((subsample*subsample_weights - mean) - variance)**2)
        
        means.append(mean)
        variances.append(variance)
        var_variances.append(var_variance)
        std.append(np.sqrt(variance))
        

    
    relative_error = np.abs(std-true_std)/true_std
    relative_mean = np.abs(means-true_mean)/true_mean
    relative_var_variance = np.abs(var_variances - true_variance_of_variance)/true_variance_of_variance
    fig, axs = plt.subplots(1, 3, figsize=(14, 6), sharex = True)
    axs[0].scatter(sample_sizes, relative_mean, label='Relative Mean')
    axs[1].scatter(sample_sizes, relative_error, label='Relative error')
    axs[2].scatter(sample_sizes, relative_var_variance, label='Relative Variance of Variance')

    params, cov = curve_fit(func_1_over_sqrtN, sample_sizes, relative_error)
    axs[1].plot(sample_sizes, params[0]/np.sqrt(sample_sizes), '--', label='{:.2f}/sqrt(N)'.format(params[0]))    
    
    
    for ax in axs:
        ax.set_xlabel('Sample Size')
        ax.legend()
        ax.set_xscale('linear')
    axs[0].set_ylabel('Value')
    return axs
    
    
def expo(t, A, off=0): return off + A[0]  * \
        np.exp(A[1] - A[2] * t) + A[3] + A[4] * np.exp(A[5] - A[6] * t)

def weighted_hist_stderr(E, bins, wgt=None):
    """
    This function creates a weighted histogram for the input data E and weights w, and returns the bin centers and weighted standard error of the histogram values
    Parameters:
        E (numpy array): input data
        bins (int): number of bins for the histogram
        wgt (numpy array): corresponding weights
    Returns:
        bin_centers (numpy array): bin centers
        hist (numpy array): histogram values
        wt_stderr (numpy array): weighted standard error of histogram values
    """
    if wgt is None:
        wgt = np.ones(E.shape)
        
    # Create a histogram with the specified number of bins, weighted by the 'wgt' array
    hist, bin_edges = np.histogram(E, bins=bins, weights=wgt)

    # Estimate the weighted standard error for each bin
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    wt_stderr = []
    for i in range(len(hist)):
        freq_bin = hist[i]
        if freq_bin != 0:
            wgt_sum_i = np.sum( wgt[(E >= bin_edges[i]) & (E < bin_edges[i+1])] **2)
            
            wt_stderr.append(np.sqrt(wgt_sum_i))
        else:
            wt_stderr.append(0)
    return bin_centers, hist, np.array(wt_stderr)
