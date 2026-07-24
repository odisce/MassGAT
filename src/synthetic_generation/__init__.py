"""
Synthetic Chromatography Generator

Easy-to-use functions for generating and visualizing synthetic MS chromatographic data.
"""

__version__ = "0.1.0"

from .core.peak_models import emg_function, generate_emg_peak
from .core.distributions import (sample_from_distribution,
                                DISTRIBUTION_PARAMS, 
                                sample_mz, sample_intensity, sample_sc_length, 
                                sample_rt_span, sample_rt_center)
from .generators.peak_families import generate_true_peak, generate_false_peak, generate_true_peak_family

# Easy-access functions with optional auto-sampling
def make_true_peak(mz=None, intensity=None, rt=None, **kwargs):
    """
    Generate a realistic true peak with optional auto-sampling.
    """
    if mz is None:
        mz = sample_mz()
    if intensity is None:
        intensity = sample_intensity()
    if rt is None:
        rt = sample_rt_center()
    if 'sc_length' not in kwargs:
        kwargs['sc_length'] = sample_sc_length(is_true_peak=True)
    if 'rt_span' not in kwargs:
        kwargs['rt_span'] = sample_rt_span()
    return generate_true_peak(mz, intensity, rt, **kwargs)

def make_false_peak(mz=None, intensity=None, rt=None, **kwargs):
    if mz is None:
        mz = sample_mz()
    if intensity is None:
        intensity = sample_intensity()
    if rt is None:
        rt = sample_rt_center()
    if 'sc_length' not in kwargs:
        kwargs['sc_length'] = sample_sc_length(is_true_peak=False)
    if 'rt_span' not in kwargs:
        kwargs['rt_span'] = sample_rt_span()
    kwargs.pop('noise_level', None)
    return generate_false_peak(mz, intensity, rt, **kwargs)

def make_peak_family(main_mz=300, main_intensity=8000, main_rt=18.0, **kwargs):
    return generate_true_peak_family(main_mz, main_intensity, main_rt, **kwargs)

def make_sampled_true_peak(**overrides):
    sampled_params = {
        'mz': sample_mz(),
        'intensity': sample_intensity(),
        'rt': sample_rt_center(),
        'sc_length': sample_sc_length(is_true_peak=True),
        'rt_span': sample_rt_span()
    }
    sampled_params.update(overrides)
    return make_true_peak(**sampled_params)

def make_sampled_false_peak(**overrides):
    sampled_params = {
        'mz': sample_mz(),
        'intensity': sample_intensity(),
        'rt': sample_rt_center(),
        'sc_length': sample_sc_length(is_true_peak=False),
        'rt_span': sample_rt_span()
    }
    sampled_params.update(overrides)
    return make_false_peak(**sampled_params)

def make_sampled_peak_family(family_size='auto', **overrides):
    main_params = {
        'main_mz': sample_mz(),
        'main_intensity': sample_intensity(),
        'main_rt': sample_rt_center()
    }
    main_params.update(overrides)
    family_params = {'family_size': family_size} if family_size != 'auto' else {}
    return make_peak_family(**main_params, **family_params)

__all__ = [
    'make_true_peak', 'make_false_peak', 'make_peak_family',
    'make_sampled_true_peak', 'make_sampled_false_peak', 'make_sampled_peak_family',
    'sample_mz', 'sample_intensity', 'sample_sc_length', 'sample_rt_span', 'sample_rt_center',
    'emg_function', 'generate_emg_peak', 'sample_from_distribution',
    'DISTRIBUTION_PARAMS'
]
