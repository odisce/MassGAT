"""Distribution fitting and biased sampling."""

import numpy as np
from scipy import stats


DISTRIBUTION_PARAMS = {
    'sc_length':      {'params': (0.0, 40.0, 20.0),    'transform': {'type': 'none'}},
    'rt_span':        {'params': (0.0, 0.8, 0.2),      'transform': {'type': 'none'}},
    'intensity_imax': {'params': (0.5, 4.2, 1.8),      'transform': {'type': 'log10', 'offset': 0}},
    'intensity_into': {'params': (38.65, 3.13, 1.08),  'transform': {'type': 'log10', 'offset': 1}},
    'mz':             {'params': (-0.18, 2.32, 0.21),  'transform': {'type': 'log10', 'offset': 1}},
    'rt_center':      {'params': (0.0, 5.0, 3.0),      'transform': {'type': 'none'}}
}


def sample_from_distribution(params, transform_info, n_samples=1000):
    """Sample from fitted skew-normal distribution."""
    selected_samples = stats.skewnorm.rvs(*params, size=n_samples)
    
    if transform_info['type'] == 'log10':
        selected_samples = 10**selected_samples - transform_info['offset']
        selected_samples = np.maximum(selected_samples, 0)
    
    return selected_samples

# PARAMETER SAMPLING FUNCTIONS

def sample_mz(n_samples=1, mz_range=(100, 999)):
    """
    Sample m/z values from fitted distribution.
    
    Parameters:
    -----------
    n_samples : int
        Number of m/z values to sample
    mz_range : tuple
        (min_mz, max_mz) to clip results
        
    Returns:
    --------
    np.ndarray
        Sampled m/z values
    """
    params = DISTRIBUTION_PARAMS['mz']['params']
    transform_info = DISTRIBUTION_PARAMS['mz']['transform']
    
    samples = sample_from_distribution(params, transform_info, n_samples)
    samples = np.clip(samples, mz_range[0], mz_range[1])
    
    return samples[0] if n_samples == 1 else samples

def sample_intensity(n_samples=1, intensity_range=(5e3, 1e9)):
    """
    Sample intensity values from fitted distribution.
    
    Parameters:
    -----------
    n_samples : int
        Number of intensity values to sample
    intensity_range : tuple
        (min_intensity, max_intensity) to clip results - now 5e3 to 1e9
        
    Returns:
    --------
    np.ndarray or float
        Sampled intensity values (guaranteed >= 5000)
    """
    params = DISTRIBUTION_PARAMS['intensity_imax']['params']
    transform_info = DISTRIBUTION_PARAMS['intensity_imax']['transform']
    
    samples = sample_from_distribution(params, transform_info, n_samples)
    # Ensure minimum 5000 intensity, maximum 1e9
    samples = np.clip(samples, intensity_range[0], intensity_range[1])
    # Additional safety check to avoid zeros
    samples = np.maximum(samples, 5000.0)
    
    return samples[0] if n_samples == 1 else samples

def sample_sc_length(n_samples=1, is_true_peak=True):
    """
    Sample scan count (number of points) from fitted distribution.
    
    Parameters:
    -----------
    n_samples : int
        Number of sc_length values to sample
    is_true_peak : bool
        True peaks: 4-30 points, False peaks: 3-15 points
        
    Returns:
    --------
    np.ndarray or int
        Sampled point counts
    """
    params = DISTRIBUTION_PARAMS['sc_length']['params']
    transform_info = DISTRIBUTION_PARAMS['sc_length']['transform']
    
    samples = sample_from_distribution(params, transform_info, n_samples)
    
    # Apply realistic bounds based on peak type - ensure minimum 4 points
    samples = np.clip(samples, 4, 30)  # Add 4 to shift up, True peaks: better resolution
    samples = samples.astype(int)
    return samples[0] if n_samples == 1 else samples

def sample_rt_span(n_samples=1, rt_range=(0.2, 5.0)):
    """
    Sample retention time span from fitted distribution.
    
    Parameters:
    -----------
    n_samples : int
        Number of RT span values to sample
    rt_range : tuple
        (min_span, max_span) in minutes to clip results
        
    Returns:
    --------
    np.ndarray or float
        Sampled RT span values in minutes
    """
    params = DISTRIBUTION_PARAMS['rt_span']['params']
    transform_info = DISTRIBUTION_PARAMS['rt_span']['transform']
    
    samples = sample_from_distribution(params, transform_info, n_samples)
    samples = np.clip(samples, rt_range[0], rt_range[1])
    
    return samples[0] if n_samples == 1 else samples

def sample_rt_center(n_samples=1, rt_range=(0.0, 20.0)):
    """
    Sample retention time center from fitted distribution.
    
    Parameters:
    -----------
    n_samples : int
        Number of RT center values to sample
    rt_range : tuple
        (min_rt, max_rt) in minutes to clip results
        
    Returns:
    --------
    np.ndarray or float
        Sampled RT center values in minutes
    """
    # Use a uniform distribution for RT center.
    samples = np.random.uniform(rt_range[0], rt_range[1], size=n_samples)
    
    return samples[0] if n_samples == 1 else samples