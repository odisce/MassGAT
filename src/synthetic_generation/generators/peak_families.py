import numpy as np
import os
import copy
import time
from ..core.peak_models import generate_emg_peak
from ..core.distributions import sample_from_distribution, DISTRIBUTION_PARAMS

# Import config - need the parent config module for configure_for_mode()
try:
    if __package__:
        from ... import config
    else:
        import src.config as config
except ImportError:
    import sys
    import importlib
    config = importlib.import_module('src.config')

mode_env = os.environ.get("IONIZATION_MODE", "positive")
config.configure_for_mode(mode_env)

# Note: Import config module directly instead of individual variables
# This ensures we get updates when configure_for_mode() modifies globals
print(f"Configured for {mode_env} mode: {config.ADDUCT_DIFFS.keys()}")


SYNTHETIC_CONFIG = {
    # 1. Instrument Tolerances
    "instrument": {
        "mz_sigma_frac": 1e-6,   # Standard deviation of m/z noise (fraction of m/z)
        "mz_clip_da": 7e-4,      # Maximum m/z error allowed (Da)
        "rt_sigma_min": 0.01,    # Standard deviation of RT variation (minutes)
    },
    
    # 2. Graph thresholds
    "similarity_thresholds": {
        "isotope": 0.5,         # Min cosine similarity to link isotopes
        "chemical": 0.8,        # Min cosine similarity to link adducts/dimers
    },

    # 3. Peak Properties
    "true_peaks": {
        "noise_std_frac": 0.03,  # Base electronic noise added to true EMG shapes
    },
    "false_peaks": {
        "sc_length_range": (4, 50),
        "rt_span_range": (0.4, 0.8),
        "baseline_frac_range": (0.15, 0.22),
        "min_intensity_frac": 0.05,
        "base_noise_frac": 0.35,
        "patterns": ['noise', 'drift_short', 'drift_long', 'bump', 'oscillation', 'irregular'],
        "pattern_probs": [0.18, 0.22, 0.18, 0.22, 0.10, 0.10]
    },

    # 4. Complex "True-with-Noise" Families
    "noise_injection": {
        "simple_main_only_prob": 0.2,
        "extra_noise_range": (1, 3),
        "flatten_strength": (0.05, 0.3),
        "len_ratio_short": (0.45, 0.8),
        "len_ratio_long": (1.2, 1.9),
        "parent_ratio_scale": (0.45, 0.8),
    }
}


DEFAULT_SCAN_INTERVAL_SECONDS = 1.0
DEFAULT_SCAN_INTERVAL_MINUTES = DEFAULT_SCAN_INTERVAL_SECONDS / 60.0
DEFAULT_SCAN_TIMES = np.arange(0.5, 20.0 + 1e-9, DEFAULT_SCAN_INTERVAL_MINUTES)

P13, P15, P18, P34, P37 = 0.011, 0.0037, 0.00205, 0.0425, 0.242 
ADDUCT_H = 1.00782

# Chain Probabilities
MAX_ADDUCT_CHAIN_DEPTH = 4       
MAX_ISOTOPE_CHAIN = 2             
ISOTOPE_CHAIN_PROB = 0.08         
ADDUCT_ISOTOPE_P_MEAN = 0.85      
ADDUCT_ISOTOPE_P_VAR = 0.05       
BASE_CHAIN_PROB = 0.1

def mz_exists(mz, peaks, margin=5e-4):
    """
    Check if a given m/z is already present in a list of peaks (dicts), within a margin.
    """
    for peak in peaks:
        if abs(peak['mz'] - mz) < margin:
            return True
    return False
"""
# === PARAMETER CONSTANTS (From Initial) ===
SYNTHETIC_CONFIG["true_peaks"]["noise_std_frac"] = 0.03
FALSE_PEAK_BASELINE_MIN_FRAC = 0.15
FALSE_PEAK_BASELINE_MAX_FRAC = 0.22
FALSE_PEAK_SC_LENGTH_MIN = 4
FALSE_PEAK_SC_LENGTH_MAX = 50
FALSE_PEAK_NOISE_FRAC = 0.35
FALSE_PEAK_PATTERN_PROBS = [0.18, 0.22, 0.18, 0.22, 0.1, 0.1]
SYNTHETIC_CONFIG["similarity_thresholds"]["isotope"] = 0.6
SYNTHETIC_CONFIG["similarity_thresholds"]["chemical"] = 0.85

# Modelization parameters
SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] = 1e-6  
SYNTHETIC_CONFIG["instrument"]["rt_sigma_min"] = 0.01  
SYNTHETIC_CONFIG["instrument"]["mz_clip_da"] = 7e-4  

# False peak pattern parameters
FALSE_PEAK_DRIFT_SHORT_NOISE_FRAC = 0.18
FALSE_PEAK_DRIFT_LONG_NOISE_FRAC = 0.12
FALSE_PEAK_DRIFT_LONG_MEDIUM_FRAC = 0.18
FALSE_PEAK_BUMP_WIDTH_MIN_FRAC = 0.6
FALSE_PEAK_BUMP_WIDTH_MAX_FRAC = 0.9
FALSE_PEAK_BUMP_HEIGHT_FRAC = 0.35
FALSE_PEAK_BUMP_NOISE_FRAC = 0.12
FALSE_PEAK_OSC_NCYCLES_MIN = 2
FALSE_PEAK_OSC_NCYCLES_MAX = 4
FALSE_PEAK_OSC_AMP_MIN_FRAC = 0.3
FALSE_PEAK_OSC_AMP_MAX_FRAC = 0.5
FALSE_PEAK_OSC_NOISE_FRAC = 0.15
FALSE_PEAK_IRREG_RWALK_FRAC = 0.06
FALSE_PEAK_IRREG_SPIKE_PROB = 0.12
FALSE_PEAK_IRREG_SPIKE_FRAC = 0.22
FALSE_PEAK_MIN_INTENSITY_FRAC = 0.05
FALSE_PEAK_RTSPAN_MIN = 0.4
FALSE_PEAK_RTSPAN_MAX = 0.8

# false-noise shaping in true_with_noise families (keeps cosine thresholds unchanged)
SYNTHETIC_CONFIG["noise_injection"]["len_ratio_short"] = (0.45, 0.8)
TRUE_WITH_NOISE_LEN_RATIO_LONG = (1.2, 1.9)
TRUE_WITH_NOISE_FLATTEN_STRENGTH = (0.05, 0.3)
TRUE_WITH_NOISE_PARENT_RATIO_SCALE = (0.45, 0.8)
SYNTHETIC_CONFIG["noise_injection"]["simple_main_only_prob"] = 0.2
SYNTHETIC_CONFIG["noise_injection"]["extra_noise_range"] = (1, 3)

# Default instrument scan configuration: 1.0 seconds per scan -> minutes units
DEFAULT_SCAN_INTERVAL_SECONDS = 1.0
DEFAULT_SCAN_INTERVAL_MINUTES = DEFAULT_SCAN_INTERVAL_SECONDS / 60.0
DEFAULT_SCAN_TIMES = np.arange(0.5, 20.0 + 1e-9, DEFAULT_SCAN_INTERVAL_MINUTES)

P13 = 0.011       # 13C
P15 = 0.0037      # 15N
P18 = 0.00205     # 18O 
P34 = 0.0425      # 34S
P37 = 0.242       # 37Cl 

ADDUCT_H = 1.00782


MAX_ADDUCT_CHAIN_DEPTH = 4        # max chain length including primary adduct
MAX_ISOTOPE_CHAIN = 2             # max number of isotopes in a chain
ISOTOPE_CHAIN_PROB = 0.08         # probability of adding an isotope in chain
ADDUCT_ISOTOPE_P_MEAN = 0.85      # probability an adduct has a 13C isotope
ADDUCT_ISOTOPE_P_VAR = 0.05       # jitter in that probability
BASE_CHAIN_PROB = 0.1             # base probability for chain extension

# --- HELPERS ---
"""


def sample_family_size(min_size=2, max_size=30):
    weights = [1/i for i in range(min_size, max_size + 1)]
    weights = np.array(weights) / sum(weights)
    return np.random.choice(range(min_size, max_size + 1), p=weights)

def sample_isotope_count(family_size_N):
    max_isotopes = min(family_size_N - 1, 4)
    if max_isotopes == 1:
        return 1
    weights = np.array([1/i**2 for i in range(1, max_isotopes + 1)])
    weights /= weights.sum()
    n_isotopes = np.random.choice(range(1, max_isotopes + 1), p=weights)
    return max(1, n_isotopes)  

def select_isotopes_by_frequency(n_isotopes):
    isotopes, freqs = zip(*config.ISOTOPE_FREQUENCIES.items())
    freqs = np.array(freqs) / np.sum(freqs)
    return np.random.choice(isotopes, size=min(n_isotopes, len(isotopes)), replace=False, p=freqs).tolist()

def select_adducts_and_dimers(n_remaining):
    adducts, a_probs = zip(*config.ADDUCT_FREQUENCIES.items())
    dimers, d_probs = zip(*config.DIMER_FREQUENCIES.items())
    a_probs, d_probs = np.array(a_probs)/sum(a_probs), np.array(d_probs)/sum(d_probs)
    n_adducts = int(n_remaining * 0.9)
    n_dimers = n_remaining - n_adducts
    selected_adducts = np.random.choice(adducts, size=min(n_adducts, len(adducts)), replace=False, p=a_probs)
    selected_dimers = np.random.choice(dimers, size=min(n_dimers, len(dimers)), replace=False, p=d_probs)
    return selected_adducts.tolist(), selected_dimers.tolist()

def weighted_choice_from_compatibility(parent_key):
    compat = config.ADDUCT_COMPATIBILITY.get(parent_key, {})
    if not compat:
        return None
    children, probs = zip(*compat.items())
    global_prior = np.array([config.ADDUCT_FREQUENCIES.get(c, 0.01) for c in children])
    probs = np.array(probs) * global_prior
    probs = probs / probs.sum()
    return np.random.choice(children, p=probs)

def _get_scan_grid_for_peak(t_cont, scan_interval=None, scan_times=None, sc_length=None):
    tmin = float(np.min(t_cont))
    tmax = float(np.max(t_cont))

    if scan_times is not None:
        scans = scan_times[(scan_times >= tmin) & (scan_times <= tmax)]
        if len(scans) == 0:
            idx = np.searchsorted(scan_times, tmin)
            start = max(0, idx - 1)
            end = min(len(scan_times), idx + 2)
            scans = scan_times[start:end]
        return np.array(scans, dtype=float)

    if scan_interval is not None:
        start = np.floor(tmin / scan_interval) * scan_interval
        end = np.ceil(tmax / scan_interval) * scan_interval
        scans = np.arange(start, end + 1e-9, scan_interval)
        scans = scans[(scans >= tmin) & (scans <= tmax)]
        return scans.astype(float)

    return np.array(t_cont, dtype=float)

def _sample_profile_on_scans(t_cont, y_cont, scan_interval=None, scan_times=None):
    scans = _get_scan_grid_for_peak(t_cont, scan_interval=scan_interval, scan_times=scan_times)
    if scans is None:
        return t_cont, y_cont
    scans = scans[(scans >= np.min(t_cont)) & (scans <= np.max(t_cont))]
    if len(scans) == 0:
        scans = np.array([np.min(t_cont)])
    y_scan = np.interp(scans, t_cont, y_cont)
    y_scan = np.maximum(y_scan, 0.0)
    return scans, y_scan


def _peak_to_mean_ratio(intensity_profile):
    x = np.asarray(intensity_profile, dtype=float)
    if len(x) == 0:
        return 0.0
    m = float(np.mean(x))
    if m <= 0:
        return float('inf')
    return float(np.max(x) / m)

# --- GENERATION FUNCTIONS ---

def generate_true_peak(mz, imax, rt, sc_length=None, rt_span=None, noise_level=0.1,
                       scan_interval=None, scan_times=DEFAULT_SCAN_TIMES, sampler=None, mode="positive"):
    if sc_length is None:
        sc_length_sample = sample_from_distribution(DISTRIBUTION_PARAMS['sc_length']['params'],
                                                   DISTRIBUTION_PARAMS['sc_length']['transform'],
                                                   1)[0]
        sc_length = int(np.clip(sc_length_sample + 3, 6, 100))

        if scan_times is not None:
            scan_diffs = np.diff(scan_times)
            if (np.abs(scan_diffs - scan_diffs[0]) < 1e-4).all():
                scan_frequency = scan_diffs.mean()
                rt_span = sc_length*scan_frequency

    if rt_span is None:
        rt_span_sample = sample_from_distribution(DISTRIBUTION_PARAMS['rt_span']['params'],
                                                 DISTRIBUTION_PARAMS['rt_span']['transform'],
                                                 1)[0]
        rt_span = max(0.05, rt_span_sample)

    sigma_factor = np.random.uniform(0.05, 0.3)
    sigma = rt_span * sigma_factor

    skew_type = np.random.random()
    if skew_type < 0.25:
        lam = np.random.uniform(0.1, 0.8) / rt_span     # quite flat
        mu = np.random.uniform(0.1, 0.3) * rt_span
    elif skew_type < 0.5:
        lam = np.random.uniform(0.8, 3.0) / rt_span     # relatively flat
        mu = np.random.uniform(0.3, 0.6) * rt_span
    else:
        lam = np.random.uniform(3.0, 6.0) / rt_span     # sharp/symmetric
        mu = np.random.uniform(0.4, 0.7) * rt_span

    tail_extension = min(3.0 / max(abs(lam), 1e-6), rt_span * 0.75)
    effective_span = rt_span + tail_extension

    t_cont, intensities_cont = generate_emg_peak(sc_length, mu, sigma, lam, imax, effective_span)
    t_cont = t_cont + rt

    if noise_level > 0:
        base_noise = np.random.normal(0, imax * noise_level * 0.3, len(intensities_cont))
        displacement_mask = np.random.random(len(intensities_cont)) < 0.4
        displacement_noise = np.zeros_like(intensities_cont)
        displacement_noise[displacement_mask] = np.random.normal(0, imax * noise_level * 0.8, np.sum(displacement_mask))
        spike_mask = np.random.random(len(intensities_cont)) < 0.1
        spike_noise = np.zeros_like(intensities_cont)
        spike_noise[spike_mask] = np.random.normal(0, imax * noise_level * 2.0, np.sum(spike_mask))
        intensities_cont += base_noise + displacement_noise + spike_noise
        intensities_cont = np.maximum(intensities_cont, imax * 0.01)

    if np.random.random() < 0.05:  
        apex_idx = int(np.argmax(intensities_cont))
        trunc_side = np.random.choice(['left', 'right'])
        
        if trunc_side == 'left' and apex_idx > 2:
            # we cut between 20% and 80% on the left of the apex
            cut = int(np.random.uniform(0.2, 0.8) * apex_idx)
            t_cont, intensities_cont = t_cont[cut:], intensities_cont[cut:]
            
        elif trunc_side == 'right' and apex_idx < len(t_cont) - 3:
            # we cut between 20% and 80% on the right of the apex
            tail_len = len(t_cont) - apex_idx
            keep_len = apex_idx + int(np.random.uniform(0.2, 0.8) * tail_len)
            t_cont, intensities_cont = t_cont[:keep_len], intensities_cont[:keep_len]

    rt_profile, intensity_profile = _sample_profile_on_scans(
        t_cont, intensities_cont,
        scan_interval=scan_interval, scan_times=scan_times
    )

    if len(rt_profile) > 1:
        if scan_times is not None:
            intensity_integral = np.trapz(intensity_profile, rt_profile)
        elif scan_interval is not None:
            dt = rt_profile[1] - rt_profile[0]
            intensity_integral = np.sum(intensity_profile) * dt
        else:
            intensity_integral = np.trapz(intensity_profile, rt_profile)
    else:
        intensity_integral = float(intensity_profile[0])

    return {
        'type': 'true_peak',
        'mz': mz,
        'rt_center': float(rt_profile[np.argmax(intensity_profile)]),
        'intensity': intensity_integral,
        'max_intensity': np.max(intensity_profile),
        'sc_length': len(rt_profile),
        'rt_span': float(rt_profile[-1] - rt_profile[0]),
        'rt_profile': np.array(rt_profile),
        'intensity_profile': np.array(intensity_profile),
    }

def generate_false_peak(mz, intensity, rt, scan_times=DEFAULT_SCAN_TIMES, sc_length=None, rt_span=None, baseline_level=None, max_into=None, sampler=None, mode="positive"):
    cfg = SYNTHETIC_CONFIG["false_peaks"]

    if sc_length is None:
        sc_length_sample = sample_from_distribution(DISTRIBUTION_PARAMS['sc_length']['params'],
                                                   DISTRIBUTION_PARAMS['sc_length']['transform'], 1)[0]
        sc_length = int(np.clip(sc_length_sample, cfg["sc_length_range"][0], cfg["sc_length_range"][1]))

    if baseline_level is None:
        baseline_level = intensity * np.random.uniform(*cfg["baseline_frac_range"])

    if scan_times is None:
        if rt_span is None:
            rt_span = np.random.uniform(*cfg["rt_span_range"])
        t = np.linspace(0, rt_span, sc_length) + rt
        rt_span = t[-1] - t[0]
    else:
        center_idx = np.argmin(np.abs(scan_times - rt))
        half_len = sc_length // 2
        start = max(0, center_idx - half_len)
        end = min(len(scan_times), start + sc_length)
        t = scan_times[start:end]
        rt_span = t[-1] - t[0]

    pattern_type = np.random.choice(cfg["patterns"], p=cfg["pattern_probs"])

    # Local mechanical parameters (hidden from global config to reduce clutter)
    if pattern_type == 'noise':
        noise = np.random.normal(0, baseline_level * cfg["base_noise_frac"], len(t))
        intensities = baseline_level + noise
        
    elif pattern_type == 'drift_short':
        slope = (np.random.random() - 0.5) * baseline_level * 0.18 / rt_span
        trend = slope * (t - t[0]) + baseline_level
        noise = np.random.normal(0, baseline_level * 0.18, len(t))
        intensities = trend + noise
        
    elif pattern_type == 'drift_long':
        slope = (np.random.random() - 0.5) * baseline_level * 0.12 / rt_span
        trend = slope * (t - t[0]) + baseline_level
        medium = baseline_level * 0.18 * np.sin(2 * np.pi * (t - t[0]) / (rt_span * np.random.uniform(2, 4)))
        noise = np.random.normal(0, baseline_level * 0.12, len(t))
        intensities = trend + medium + noise
        
    elif pattern_type == 'bump':
        center = t[np.random.randint(len(t))]
        width = rt_span * np.random.uniform(0.6, 0.9)
        intensities = baseline_level + baseline_level * 0.35 * np.exp(-((t - center) ** 2) / (2 * width ** 2))
        noise = np.random.normal(0, baseline_level * 0.12, len(t))
        intensities += noise
        
    elif pattern_type == 'oscillation':
        n_cycles = np.random.uniform(2, 4)
        freq = 2 * np.pi * n_cycles / rt_span
        phase = np.random.uniform(0, 2 * np.pi)
        osc = baseline_level * np.random.uniform(0.3, 0.5) * np.sin(freq * (t - t[0]) + phase)
        noise = np.random.normal(0, baseline_level * 0.15, len(t))
        intensities = baseline_level + osc + noise
        
    else:  # irregular
        random_walk = np.cumsum(np.random.normal(0, baseline_level * 0.06, len(t)))
        random_walk -= np.mean(random_walk)
        spike_mask = np.random.random(len(t)) < 0.12
        spikes = np.zeros(len(t))
        spikes[spike_mask] = np.random.normal(0, baseline_level * 0.22, np.sum(spike_mask))
        intensities = baseline_level + random_walk + spikes

    intensities = np.maximum(intensities, baseline_level * cfg["min_intensity_frac"])
    intensities_norm = intensities / np.max(intensities)
    intensities = intensities_norm * intensity
    integrated_intensity = np.trapz(intensities, t)

    if max_into is not None and integrated_intensity > max_into:
        scale_factor_max  = max_into / integrated_intensity
        scale_factor = np.random.uniform(0.05 * scale_factor_max, scale_factor_max)
        while np.sum(intensities * scale_factor > 128) < 4 and scale_factor > 0.05 * scale_factor_max:
            scale_factor *= 1.1
        intensities = intensities * scale_factor

    return {
        'type': 'false_peak',
        'mz': mz,
        'rt_profile': t,
        'intensity_profile': intensities,
        'max_intensity': intensities.max(),
        'intensity': np.trapz(intensities, t),
        'rt_center': t[np.argmax(intensities)],
        'sc_length': len(t),
        'rt_span': rt_span,
        'baseline_level': baseline_level,
        'pattern_type': pattern_type
    }

def _cosine_similarity(t1, x1, t2, x2, num_points=200, scan_times=DEFAULT_SCAN_TIMES, use_interpolation=False):
    t1 = np.asarray(t1)
    x1 = np.asarray(x1)
    t2 = np.asarray(t2)
    x2 = np.asarray(x2)

    if not use_interpolation:
        if scan_times is not None:
            rt_union = scan_times
        else:
            rt_union = np.sort(np.unique(np.concatenate([t1, t2])))
        i1 = np.interp(rt_union, t1, x1, left=0, right=0)
        i2 = np.interp(rt_union, t2, x2, left=0, right=0)
        n1, n2 = np.linalg.norm(i1), np.linalg.norm(i2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos = np.dot(i1, i2) / (n1 * n2)
        return float(np.clip(cos, 0.0, 1.0))

    start, end = min(np.min(t1), np.min(t2)), max(np.max(t1), np.max(t2))
    if end <= start or (end - start) <= 1e-6:
        return 0.0
    if scan_times is not None:
        mask = (scan_times >= start) & (scan_times <= end)
        grid = scan_times[mask]
        if len(grid) < 3:
            grid = np.linspace(start, end, num_points)
    else:
        grid = np.linspace(start, end, num_points)
    y1 = np.interp(grid, t1, x1, left=0.0, right=0.0)
    y2 = np.interp(grid, t2, x2, left=0.0, right=0.0)
    n1, n2 = np.linalg.norm(y1), np.linalg.norm(y2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = np.dot(y1, y2) / (n1 * n2)
    return float(np.clip(cos, 0.0, 1.0))

def scale_profile_by_intensity(main_rt_profile, main_intensity_profile, intensity_ratio,
                               noise_scale=0.2, scan_interval=None, scan_times=None, scan_grid=None, sampler=None):
    main_max = np.max(main_intensity_profile)
    if main_max <= 0:
        return None
    target_imax = main_max * intensity_ratio
    scaled_intensity = main_intensity_profile * intensity_ratio
    target_integrated_intensity = np.trapz(scaled_intensity, main_rt_profile)

    if target_imax < 1e2 or target_integrated_intensity < 1e3:
        return None
    
    if noise_scale > 0:
        noise_std = target_imax * 0.15 * noise_scale
        noise = np.random.normal(0, noise_std, len(scaled_intensity))
        scaled_intensity = np.clip(scaled_intensity + noise, 0, None)

    detection_limit = main_max * np.random.uniform(0.0002, 0.005)
    eps_threshold = max(detection_limit, target_imax * np.random.uniform(0.002, 0.01))

    if scan_grid is not None:
        sg = np.array(scan_grid, dtype=float)
        sg = sg[(sg >= np.min(main_rt_profile)) & (sg <= np.max(main_rt_profile))]
        if len(sg) == 0:
            return None
        sampled = np.interp(sg, main_rt_profile, scaled_intensity)
    elif scan_interval is not None or scan_times is not None:
        scans, sampled = _sample_profile_on_scans(main_rt_profile, scaled_intensity,
                                                  scan_interval=scan_interval, scan_times=scan_times)
        sg = scans
    else:
        sg = np.array(main_rt_profile)
        sampled = np.array(scaled_intensity)

    valid_mask = sampled >= eps_threshold
    dropout_prob = np.clip(1.0 - intensity_ratio**0.2, 0.0, 0.7)
    random_mask = np.random.rand(len(valid_mask)) > dropout_prob
    final_mask = valid_mask & random_mask

    if np.sum(final_mask) < 4:
        return None

    filtered_rt = sg[final_mask]
    filtered_intensity = np.maximum(sampled[final_mask], 0.0)
    integrated_intensity = np.trapz(filtered_intensity, filtered_rt)
    filtered_intensity = filtered_intensity * (target_integrated_intensity / integrated_intensity)

    return filtered_rt, filtered_intensity

def generate_profile_with_similarity(main_rt_profile, main_intensity_profile, ratio,
                                     noise_scale, kind, cosine_fn, max_retries_base=10, sampler=None, **kwargs):
    threshold = SYNTHETIC_CONFIG["similarity_thresholds"]["isotope"] if kind == 'isotope' else SYNTHETIC_CONFIG["similarity_thresholds"]["chemical"]
    if ratio < 0.1:
        max_retries = max_retries_base + 2
    elif ratio < 0.05:
        max_retries = max_retries_base + 3
    else:
        max_retries = max_retries_base

    for _ in range(max_retries):
        result = scale_profile_by_intensity(main_rt_profile, main_intensity_profile, ratio, noise_scale=noise_scale, sampler=sampler, **kwargs)
        if result is None:
            continue
        rt, intensity = result
        score = cosine_fn(main_rt_profile, main_intensity_profile, rt, intensity)
        if score >= threshold:
            return rt, intensity
    return None

def _estimate_element_counts_from_mass(mono_mass):
    jitter = np.random.uniform(0.7, 1.0)  
    nC = max(1, int(round((mono_mass / 12.0) * jitter)))
    nN = max(0, int(round(nC * np.random.uniform(0.06, 0.18))))   
    nO = max(0, int(round(nC * np.random.uniform(0.08, 0.35))))   
    nS = max(0, int(round(nC * np.random.uniform(0.0, 0.05))))    
    cl_prob = min(0.5, max(0.02, (mono_mass - 120) / 800.0))  
    nCl = 1 if np.random.rand() < cl_prob else 0
    return nC, nN, nO, nS, nCl

def _attempt_chain_from_peak(parent_peak, parent_key, depth, rt_profile, intensity_profile, accum_peaks,
                             max_depth=MAX_ADDUCT_CHAIN_DEPTH, child_limit=1, base_chain_prob=BASE_CHAIN_PROB):
    if depth >= max_depth:
        return
    tries = 0
    while tries < child_limit:
        tries += 1
        if np.random.rand() > base_chain_prob:
            continue
        next_adduct = weighted_choice_from_compatibility(parent_key)
        if next_adduct is None:
            continue
        parent_mz = parent_peak['mz']
        if next_adduct in config.ADDUCT_LOSSES:
            add_mz = parent_mz - config.ADDUCT_DIFFS.get(next_adduct, 0.0)
        else:
            add_mz = parent_mz + config.ADDUCT_DIFFS.get(next_adduct, 0.0)
        if add_mz <= 75. :
            continue
        parent_imax = parent_peak.get('max_intensity', np.max(parent_peak.get('intensity_profile', [1.0])))
        min_ratio, max_ratio = max(128./parent_imax, 1e-4), min(1e10/parent_imax, 1e2)
        if np.random.rand() < 0.05:
            ratio = np.exp(np.random.uniform(np.log(1.0), np.log(max_ratio)))
        else:
            ratio = np.exp(np.random.uniform(np.log(min_ratio), np.log(1.0)))

        child_intensity = parent_imax * ratio
        scaled = scale_profile_by_intensity(rt_profile, intensity_profile, child_intensity, noise_scale=0.2)
        if scaled is None:
            continue
        child_rt, child_int_profile = scaled
        child_peak = {
            'type': 'true_peak',
            'mz': add_mz,
            'rt_profile': child_rt,
            'intensity_profile': child_int_profile,
            'max_intensity': np.max(child_int_profile),
            'rt_center': child_rt[np.argmax(child_int_profile)],
            'sc_length': len(child_int_profile),
            'rt_span': child_rt[-1] - child_rt[0] if len(child_rt) > 1 else 0.0,
            'adduct_type': next_adduct,
            'chained_from': parent_key
        }
        accum_peaks.append(child_peak)
        _attempt_chain_from_peak(child_peak, next_adduct, depth + 1, rt_profile, intensity_profile, accum_peaks,
                                 max_depth=max_depth, child_limit=child_limit, base_chain_prob=base_chain_prob)
        if len(accum_peaks) > 8:
            return

def generate_true_peak_family(main_mz, imax, main_rt, family_size=None, sampler=None, mode="positive", **kwargs):
    if family_size is None:
        N = sample_family_size()

    n_iso = sample_isotope_count(N)
    selected_isotopes = select_isotopes_by_frequency(n_iso)
    n_remaining = max(0, N - 1 - n_iso)
    selected_adducts, selected_dimers = select_adducts_and_dimers(n_remaining)

    main_peak = None
    for _ in range(10):
        cand = generate_true_peak(main_mz, imax, main_rt, sampler=sampler, mode=mode)
        if cand and cand.get('sc_length', 0) >= 4:
            noise_std = np.max(cand['intensity_profile']) * SYNTHETIC_CONFIG["true_peaks"]["noise_std_frac"]
            noise = np.random.normal(0, noise_std, len(cand['intensity_profile']))
            cand['intensity_profile'] = np.maximum(cand['intensity_profile'] + noise, 0)
            cand['max_intensity'] = np.max(cand['intensity_profile'])
            main_peak = cand
            break
    if main_peak is None:
        main_peak = generate_true_peak(main_mz, imax, main_rt, sampler=sampler, mode=mode)
        noise_std = np.max(main_peak['intensity_profile']) * SYNTHETIC_CONFIG["true_peaks"]["noise_std_frac"]
        noise = np.random.normal(0, noise_std, len(main_peak['intensity_profile']))
        main_peak['intensity_profile'] = np.maximum(main_peak['intensity_profile'] + noise, 0)
        main_peak['max_intensity'] = np.max(main_peak['intensity_profile'])

    main_rt_profile = main_peak['rt_profile']
    main_intensity_profile = main_peak['intensity_profile']
    main_rt_center = main_peak.get('rt_center', main_rt)
    main_rt_span = main_peak.get('rt_span', main_rt_profile[-1] - main_rt_profile[0] if len(main_rt_profile) > 1 else 0.0)

    family = {
        'main_peak': main_peak,
        'isotope_peaks': [],
        'adduct_peaks': [],
        'dimer_peaks': [],
        'family_id': f"family_{main_mz:.4f}_{main_rt:.3f}",
        'family_size': None
    }

    nC, nN, nO, nS, nCl = _estimate_element_counts_from_mass(main_mz)

    iso_added = 0
    for isotope_type in selected_isotopes:
        if isotope_type not in config.ISOTOPE_DIFFS:
            continue
        shift = config.ISOTOPE_DIFFS[isotope_type]
        iso_mz = main_mz + shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        if mz_exists(iso_mz, [main_peak] + family['isotope_peaks'] + family['adduct_peaks'] + family['dimer_peaks']):
            continue
        if isotope_type == 'isotope_13C': ratio_exp = nC * P13
        elif isotope_type == 'isotope_15N': ratio_exp = nN * P15
        elif isotope_type == 'isotope_18O': ratio_exp = nO * P18
        elif isotope_type == 'isotope_34S': ratio_exp = nS * P34
        elif isotope_type == 'isotope_37Cl': ratio_exp = nCl * P37
        else: ratio_exp = 0.0005 * (main_mz / 200.0)

        ratio = ratio_exp * np.random.uniform(0.85, 1.15)
        scaled = generate_profile_with_similarity(
            main_rt_profile, main_intensity_profile, ratio=ratio, noise_scale=0.3,
            kind='isotope', cosine_fn=_cosine_similarity, sampler=sampler
        )
        if scaled is None:
            continue
        iso_rt_prof, iso_int_prof = scaled
        iso_peak = {
            'type': 'true_peak', 'mz': iso_mz, 'rt_profile': iso_rt_prof,
            'intensity_profile': iso_int_prof, 'max_intensity': np.max(iso_int_prof),
            'intensity': np.sum(iso_int_prof) * (iso_rt_prof[1] - iso_rt_prof[0] if len(iso_rt_prof) > 1 else iso_int_prof[0]),
            'rt_center': main_rt_center, 'sc_length': len(iso_int_prof), 'rt_span': main_rt_span,
            'isotope_type': isotope_type, 'intensity_ratio': ratio
        }
        family['isotope_peaks'].append(iso_peak)
        iso_added += 1

        if iso_added < MAX_ISOTOPE_CHAIN and np.random.rand() < ISOTOPE_CHAIN_PROB:
            candidates = [it for it in config.ISOTOPE_DIFFS.keys() if (isotope_type == 'isotope_13C' or it != isotope_type)]
            if candidates:
                sec = np.random.choice(candidates)
                sec_shift = config.ISOTOPE_DIFFS[sec]
                combo_mz = iso_mz + sec_shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
                if sec == 'isotope_13C': r2 = nC * P13
                elif sec == 'isotope_15N': r2 = nN * P15
                elif sec == 'isotope_18O': r2 = nO * P18
                elif sec == 'isotope_34S': r2 = nS * P34
                elif sec == 'isotope_37Cl': r2 = nCl * P37
                else: r2 = 0.0002 * (main_mz / 200.0)
                combo_ratio = (ratio_exp * r2) * np.random.uniform(0.8, 1.2)  
                scaled2 = generate_profile_with_similarity(
                    main_rt_profile, main_intensity_profile, ratio=combo_ratio, noise_scale=0.18,
                    kind='isotope', cosine_fn=_cosine_similarity, sampler=sampler
                )
                if scaled2 is not None:
                    combo_rt, combo_int_prof = scaled2
                    combo_peak = {
                        'type': 'true_peak', 'mz': combo_mz, 'rt_profile': combo_rt,
                        'intensity_profile': combo_int_prof, 'max_intensity': np.max(combo_int_prof),
                        'intensity': np.sum(combo_int_prof) * (combo_rt[1] - combo_rt[0]) if len(combo_rt) > 1 else combo_int_prof[0],
                        'rt_center': main_rt_center, 'sc_length': len(combo_int_prof), 'rt_span': main_rt_span,
                        'isotope_type': f"{isotope_type}+{sec}", 'intensity_ratio': combo_ratio
                    }
                    family['isotope_peaks'].append(combo_peak)
                    iso_added += 1

    for adduct_type in selected_adducts:
        if adduct_type not in config.ADDUCT_DIFFS:
            continue
        if adduct_type in config.ADDUCT_LOSSES:
            adduct_mz = main_mz - config.ADDUCT_DIFFS[adduct_type] + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        else:
            adduct_mz = main_mz + config.ADDUCT_DIFFS[adduct_type] + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        if mz_exists(adduct_mz, [main_peak] + family['isotope_peaks'] + family['adduct_peaks'] + family['dimer_peaks']):
            continue
        main_imax = np.max(main_intensity_profile)
        min_ratio, max_ratio = max(128./main_imax, 1e-4), min(1e10/main_imax, 1e2)
        if np.random.rand() < 0.1:
            ratio = np.exp(np.random.uniform(np.log(1.0), np.log(max_ratio)))
        else:
            ratio = np.exp(np.random.uniform(np.log(min_ratio), np.log(1.0)))

        ad_res = generate_profile_with_similarity(main_rt_profile, main_intensity_profile, ratio=ratio, noise_scale=0.3, 
                                                  kind='adduct', cosine_fn=_cosine_similarity, sampler=sampler)
        if ad_res is None:
            continue
        ad_rt_prof, ad_int_prof = ad_res
        ad_peak = {
            'type': 'true_peak', 'mz': adduct_mz, 'rt_profile': ad_rt_prof,
            'intensity_profile': ad_int_prof, 'max_intensity': np.max(ad_int_prof),
            'intensity' : np.trapz(ad_int_prof, ad_rt_prof),
            'rt_center': main_rt_center, 'sc_length': len(ad_int_prof),
            'rt_span': main_rt_span, 'adduct_type': adduct_type
        }
        family['adduct_peaks'].append(ad_peak)

        chained = []
        if adduct_mz > 75. :
            _attempt_chain_from_peak(ad_peak, adduct_type, depth=0, rt_profile=ad_rt_prof,
                                    intensity_profile=ad_int_prof, accum_peaks=chained,
                                    max_depth=MAX_ADDUCT_CHAIN_DEPTH, child_limit=1, base_chain_prob=BASE_CHAIN_PROB)
        chained_unique = [p for p in chained if not mz_exists(p['mz'], [main_peak] + family['isotope_peaks'] + family['adduct_peaks'] + family['dimer_peaks'])]
        if chained_unique:
            family['adduct_peaks'].extend(chained_unique)

        p_ad_iso = np.clip(np.random.normal(ADDUCT_ISOTOPE_P_MEAN, ADDUCT_ISOTOPE_P_VAR), 0.0, 1.0)
        if np.random.rand() < p_ad_iso:
            ad_iso_mz = adduct_mz + config.ISOTOPE_DIFFS['isotope_13C']
            if mz_exists(ad_iso_mz, [main_peak] + family['isotope_peaks'] + family['adduct_peaks'] + family['dimer_peaks']):
                continue
            iso13_ratio = max(1, nC) * P13 * np.random.uniform(0.85, 1.15)
            scaled_ad_iso = generate_profile_with_similarity(ad_rt_prof, ad_int_prof, ratio=iso13_ratio, 
                                                             noise_scale=0.15, kind='isotope', cosine_fn=_cosine_similarity, sampler=sampler)
            if scaled_ad_iso is not None:
                adiso_rt, adiso_int = scaled_ad_iso
                ad_iso_peak = {
                    'type': 'true_peak', 'mz': ad_iso_mz, 'rt_profile': adiso_rt,
                    'intensity_profile': adiso_int, 'max_intensity': np.max(adiso_int),
                    'intensity': np.trapz(adiso_int, adiso_rt),
                    'rt_center': main_rt_center, 'sc_length': len(adiso_int),
                    'rt_span': main_rt_span, 'isotope_type': 'isotope_13C', 'parent_adduct': adduct_type
                }
                family['adduct_peaks'].append(ad_iso_peak)

    for dimer_type in selected_dimers:
        if dimer_type not in config.DIMER_DIFFS:
            continue
        adduct_h = ADDUCT_H
        if mode == 'positive':
            dimer_mz = 2 * (main_mz - adduct_h) + config.DIMER_DIFFS[dimer_type] + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        elif mode == 'negative':
            dimer_mz = 2 * (main_mz + adduct_h) + config.DIMER_DIFFS[dimer_type] + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        main_imax = np.max(main_intensity_profile)
        min_ratio, max_ratio = max(128./main_imax, 1e-4), min(1e10/main_imax, 1e2)
        if np.random.rand() < 0.2:
            ratio = np.random.uniform(1.0, max_ratio)
        else:
            ratio = np.random.uniform(min_ratio, 1.0)

        d_res = generate_profile_with_similarity(main_rt_profile, main_intensity_profile, ratio = ratio, noise_scale=0.2,
                                                  kind='dimer', cosine_fn=_cosine_similarity, sampler=sampler)
        if d_res is None:
            continue
        d_rt_prof, d_int_prof = d_res
        dimer_peak = {
            'type': 'true_peak', 'mz': dimer_mz, 'rt_profile': d_rt_prof,
            'intensity_profile': d_int_prof, 'max_intensity': np.max(d_int_prof),
            'intensity' : np.trapz(d_int_prof, d_rt_prof),
            'rt_center': main_rt_center, 'sc_length': len(d_int_prof),
            'rt_span': main_rt_span, 'dimer_type': dimer_type
        }
        family['dimer_peaks'].append(dimer_peak)

        p_dimer_iso = np.clip(np.random.normal(ADDUCT_ISOTOPE_P_MEAN, ADDUCT_ISOTOPE_P_VAR), 0.0, 1.0)
        if np.random.rand() < p_dimer_iso:
            iso13_ratio = max(1, nC) * P13 * np.random.uniform(0.85, 1.15)
            dimer_iso_mz = dimer_mz + config.ISOTOPE_DIFFS['isotope_13C']
            scaled_dimer_iso = generate_profile_with_similarity(d_rt_prof, d_int_prof, iso13_ratio, 
                                                               noise_scale=0.15, kind='isotope', cosine_fn=_cosine_similarity, sampler=sampler)
            if scaled_dimer_iso is not None:
                dimeriso_rt, dimeriso_int = scaled_dimer_iso
                dimer_iso_peak = {
                    'type': 'true_peak', 'mz': dimer_iso_mz, 'rt_profile': dimeriso_rt,
                    'intensity_profile': dimeriso_int, 'max_intensity': np.max(dimeriso_int),
                    'intensity': np.trapz(dimeriso_int, dimeriso_rt),
                    'rt_center': main_rt_center, 'sc_length': len(dimeriso_int),
                    'rt_span': main_rt_span, 'isotope_type': 'isotope_13C', 'parent_dimer': dimer_type
                }
                family['dimer_peaks'].append(dimer_iso_peak)

    if len(family['isotope_peaks']) == 0:
        iso_type = 'isotope_13C'
        forced_ratio = max(1, nC) * P13 * np.random.uniform(0.8, 1.0)
        result = generate_profile_with_similarity(
            main_peak['rt_profile'], main_peak['intensity_profile'], ratio=forced_ratio, noise_scale=0.,
            kind='isotope', cosine_fn=_cosine_similarity, max_retries_base=100, sampler=sampler
        )
        if result is not None:
            rt, intensity = result
            family['isotope_peaks'].append({
                'type': 'true_peak', 'mz': main_peak['mz'] + config.ISOTOPE_DIFFS[iso_type],
                'rt_profile': rt, 'intensity_profile': intensity,
                'isotope_type': iso_type, 'intensity_ratio': forced_ratio,
                'max_intensity': np.max(intensity), 'sc_length': len(intensity),
                'intensity': np.trapz(intensity, rt),
                'similarity': _cosine_similarity(main_peak['rt_profile'], main_peak['intensity_profile'], rt, intensity)
            })

    family['family_size'] = 1 + len(family['isotope_peaks']) + len(family['adduct_peaks']) + len(family['dimer_peaks'])
    return family

def generate_false_peak_family(main_mz, main_rt, scan_times=DEFAULT_SCAN_TIMES, p_chain=0.2, max_chain_depth=2, max_extra_branches=3, sampler=None, mode="positive"):
    family = {
        'main_peak': None, 'isotope_peaks': [], 'adduct_peaks': [],
        'dimer_peaks': [], 'extra_peaks': [], 'family_id': f"false_family_{main_mz:.1f}_{main_rt:.1f}",
    }

    effective_scan_times = scan_times if scan_times is not None else DEFAULT_SCAN_TIMES

    N = max(2, sample_family_size())
    n_iso = sample_isotope_count(N)
    selected_isotopes = select_isotopes_by_frequency(n_iso)
    n_remaining = N - 1 - n_iso
    selected_adducts, selected_dimers = select_adducts_and_dimers(n_remaining)

    main_int = sample_from_distribution(DISTRIBUTION_PARAMS['intensity_imax']['params'], DISTRIBUTION_PARAMS['intensity_imax']['transform'], 1)[0]

    for _ in range(20):
        main_peak = generate_false_peak(main_mz, main_int, main_rt, scan_times=effective_scan_times, sampler=sampler, mode=mode)
        if len(main_peak['rt_profile']) >= 3:
            break
    family['main_peak'] = main_peak
    all_peaks = [main_peak]

    def _spawn_chained_peak(parent_peak, chain_depth=0):
        if chain_depth >= max_chain_depth or np.random.rand() > p_chain:
            return None
        parent_mz = parent_peak['mz']
        parent_rt = np.mean(parent_peak['rt_profile'])
        parent_int = parent_peak['max_intensity']
        rel_type = np.random.choice(['isotope', 'adduct', 'dimer'])
        frac = np.random.uniform(0.05, 0.6)
        child_int = parent_int * frac
        child_rt = parent_rt + np.random.normal(0, 0.02)

        adduct_h = ADDUCT_H
        if rel_type == 'isotope':
            shift = np.random.choice(list(config.ISOTOPE_DIFFS.values()))
            child_mz = parent_mz + shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
            sim_thresh = 0.6
        elif rel_type == 'adduct':
            shift = np.random.choice(list(config.ADDUCT_DIFFS.values()))
            if (np.random.rand() > 0.9) and (parent_mz - shift > 75):
                shift = -shift
            if mode == 'positive':
                child_mz = parent_mz + shift - adduct_h + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
            else:
                child_mz = parent_mz + shift + adduct_h + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
            sim_thresh = 0.85
        else:
            shift = np.random.choice(list(config.DIMER_DIFFS.values()))
            if mode == 'positive':
                child_mz = 2 * (parent_mz - adduct_h) + shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
            else:
                child_mz = 2 * (parent_mz + adduct_h) + shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
            sim_thresh = 0.85

        child_peak = None
        sim = None
        for _ in range(15):
            if mz_exists(child_mz, all_peaks):
                continue
            candidate_peak = generate_false_peak(child_mz, child_int, child_rt, scan_times=effective_scan_times, sampler=sampler, mode=mode)
            candidate_peak['type'] = 'false_peak'
            candidate_sim = _cosine_similarity(
                parent_peak['rt_profile'], parent_peak['intensity_profile'],
                candidate_peak['rt_profile'], candidate_peak['intensity_profile'])
            if len(candidate_peak['rt_profile']) >= 3 and candidate_sim >= sim_thresh:
                child_peak = candidate_peak
                sim = candidate_sim
                break
        if child_peak is not None:
            child_peak['relation_type'] = rel_type
            child_peak['similarity_to_parent'] = sim
            all_peaks.append(child_peak)
            if np.random.rand() < p_chain:
                _spawn_chained_peak(child_peak, chain_depth + 1)
        return child_peak

    for iso_type in selected_isotopes:
        iso_mz = main_mz + config.ISOTOPE_DIFFS[iso_type] + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        frac = np.random.uniform(0.05, 0.5)
        iso_intensity = family['main_peak']['max_intensity'] * frac
        for _ in range(20):
            if mz_exists(iso_mz, all_peaks):
                continue
            iso_rt = main_rt + np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["rt_sigma_min"])
            iso_peak = generate_false_peak(iso_mz, iso_intensity, iso_rt, scan_times=effective_scan_times, sampler=sampler, mode=mode)
            iso_peak['type'] = 'false_peak'
            sim = _cosine_similarity(family['main_peak']['rt_profile'], family['main_peak']['intensity_profile'],
                                     iso_peak['rt_profile'], iso_peak['intensity_profile'])
            if len(iso_peak['rt_profile']) >= 3 and sim >= 0.6:
                break
        iso_peak['isotope_type'] = iso_type
        iso_peak['intensity_fraction_of_main'] = frac
        iso_peak['similarity_to_main'] = sim
        family['isotope_peaks'].append(iso_peak)
        all_peaks.append(iso_peak)
        _spawn_chained_peak(iso_peak, 1)

    for adduct_type in selected_adducts:
        shift = config.ADDUCT_DIFFS[adduct_type]
        if (np.random.rand() > 0.9) and (main_mz - shift > 75):
                shift = -shift
        adduct_h = ADDUCT_H
        if mode == 'positive':
            adduct_mz = main_mz + shift - adduct_h 
        else:
            adduct_mz = main_mz + shift + adduct_h 

        adduct_rt = main_rt + np.random.normal(0, 0.02)
        adduct_intensity = sample_from_distribution(DISTRIBUTION_PARAMS['intensity_imax']['params'], DISTRIBUTION_PARAMS['intensity_imax']['transform'], 1)[0]
        adduct_peak = None
        sim = None
        for _ in range(20):
            if mz_exists(adduct_mz, all_peaks):
                continue
            candidate_peak = generate_false_peak(adduct_mz, adduct_intensity, adduct_rt, scan_times=effective_scan_times, sampler=sampler, mode=mode)
            candidate_peak['type'] = 'false_peak'
            candidate_sim = _cosine_similarity(family['main_peak']['rt_profile'], family['main_peak']['intensity_profile'],
                                               candidate_peak['rt_profile'], candidate_peak['intensity_profile'])
            if len(candidate_peak['rt_profile']) >= 4 and candidate_sim >= 0.85:
                adduct_peak = candidate_peak
                sim = candidate_sim
                break
        if adduct_peak is not None:
            adduct_peak['adduct_type'] = adduct_type
            adduct_peak['similarity_to_main'] = sim
            family['adduct_peaks'].append(adduct_peak)
            all_peaks.append(adduct_peak)
            _spawn_chained_peak(adduct_peak, 1)

    for dimer_type in selected_dimers:
        adduct_h = ADDUCT_H
        if mode == 'positive':
            dimer_mz = 2 * (main_mz - adduct_h) + config.DIMER_DIFFS[dimer_type] + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        else:
            dimer_mz = 2 * (main_mz + adduct_h) + config.DIMER_DIFFS[dimer_type] + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * main_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
            
        dimer_rt = main_rt + np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["rt_sigma_min"])
        dimer_intensity = sample_from_distribution(DISTRIBUTION_PARAMS['intensity_imax']['params'], DISTRIBUTION_PARAMS['intensity_imax']['transform'], 1)[0]
        dimer_peak = None
        sim = None
        for _ in range(20):
            if mz_exists(dimer_mz, all_peaks):
                continue
            candidate_peak = generate_false_peak(dimer_mz, dimer_intensity, dimer_rt, scan_times=effective_scan_times, sampler=sampler, mode=mode)
            candidate_peak['type'] = 'false_peak'
            candidate_sim = _cosine_similarity(family['main_peak']['rt_profile'], family['main_peak']['intensity_profile'],
                                               candidate_peak['rt_profile'], candidate_peak['intensity_profile'])
            if len(candidate_peak['rt_profile']) >= 4 and candidate_sim >= 0.85:
                dimer_peak = candidate_peak
                sim = candidate_sim
                break
        if dimer_peak is not None:
            dimer_peak['dimer_type'] = dimer_type
            dimer_peak['similarity_to_main'] = sim
            family['dimer_peaks'].append(dimer_peak)
            all_peaks.append(dimer_peak)
            _spawn_chained_peak(dimer_peak, 1)

    for _ in range(max_extra_branches):
        parent = np.random.choice(all_peaks)
        extra_peak = _spawn_chained_peak(parent, 0)
        if extra_peak is not None:
            family['extra_peaks'].append(extra_peak)

    family['family_size'] = len(all_peaks)
    return family



def generate_true_with_noise_family(main_mz, main_intensity, main_rt, scan_times=DEFAULT_SCAN_TIMES, n_noise_links=None, p_link_to_adduct=0.4, similarity_thresholds={"isotope": 0.65, "adduct": 0.85, "dimer": 0.85}, max_noise_attempts=20, sampler=None, mode="positive"):
    effective_scan_times = scan_times if scan_times is not None else DEFAULT_SCAN_TIMES

    true_family = generate_true_peak_family(main_mz, main_intensity, main_rt, scan_times=effective_scan_times, sampler=sampler, mode=mode)
    if isinstance(true_family, list) and len(true_family) > 0:
        true_family = true_family[0]

    for peak_list in ['main_peak', 'isotope_peaks', 'adduct_peaks', 'dimer_peaks']:
        peaks = true_family[peak_list] if peak_list == 'main_peak' else true_family.get(peak_list, [])
        if isinstance(peaks, dict):
            peaks = [peaks]
        for peak in peaks:
            noise_std = np.max(peak['intensity_profile']) * 0.05
            noise = np.random.normal(0, noise_std, len(peak['intensity_profile']))
            peak['intensity_profile'] = np.maximum(peak['intensity_profile'] + noise, 0)
            peak['max_intensity'] = np.max(peak['intensity_profile'])

    family = {
        'main_peak': true_family['main_peak'],
        'isotope_peaks': true_family.get('isotope_peaks', []),
        'adduct_peaks': true_family.get('adduct_peaks', []),
        'dimer_peaks': true_family.get('dimer_peaks', []),
        'noise_peaks': [],
        'family_id': f"true_with_noise_{main_mz:.4f}_{main_rt:.3f}",
    }

    use_simple_main_plus_noise = (np.random.rand() < SYNTHETIC_CONFIG["noise_injection"]["simple_main_only_prob"])
    if use_simple_main_plus_noise:
        family['isotope_peaks'] = []
        family['adduct_peaks'] = []
        family['dimer_peaks'] = []

    all_valid_parents = [family['main_peak']] + family['adduct_peaks']
    if len(all_valid_parents) == 0:
        all_valid_parents = [family['main_peak']]

    if n_noise_links is None:
        if use_simple_main_plus_noise:
            n_noise_links = int(np.random.randint(
                SYNTHETIC_CONFIG["noise_injection"]["extra_noise_range"][0],
                SYNTHETIC_CONFIG["noise_injection"]["extra_noise_range"][1] + 1
            ))
        else:
            n_true_peaks = 1 + len(family['isotope_peaks']) + len(family['adduct_peaks']) + len(family['dimer_peaks'])
            raw = int(np.random.poisson(max(1, n_true_peaks * 0.5)))
            n_noise_links = max(1, min(raw, n_true_peaks))

    attempts = 0
    added = 0
    while added < n_noise_links and attempts < n_noise_links * max_noise_attempts:
        if use_simple_main_plus_noise:
            parent = family['main_peak']
        elif len(family['adduct_peaks']) > 0 and np.random.rand() < p_link_to_adduct:
            parent = np.random.choice(family['adduct_peaks'])
        else:
            parent = family['main_peak']

        parent_mz = parent['mz']
        parent_rt = np.mean(parent['rt_profile'])
        parent_int = parent['max_intensity']

        if parent_mz < 30.:
            attempts += 1
            continue

        if use_simple_main_plus_noise:
            link_type = np.random.choice(['isotope', 'adduct', 'dimer'], p=[0.1, 0.8, 0.1])
        else:
            link_type = np.random.choice(['isotope', 'adduct', 'dimer'], p=[0.15, 0.8, 0.05])

        adduct_h = ADDUCT_H
        if link_type == 'isotope':
            mz_shift = np.random.choice(list(config.ISOTOPE_DIFFS.values()))
            similarity_min = similarity_thresholds['isotope']
            noise_mz = parent_mz + mz_shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        elif link_type == 'adduct':
            mz_shift = np.random.choice(list(config.ADDUCT_DIFFS.values()))
            if (np.random.rand() > 0.9) and (parent_mz - mz_shift > 75):
                mz_shift = -mz_shift
            similarity_min = similarity_thresholds['adduct']
            noise_mz = parent_mz + mz_shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
        else:
            mz_shift = np.random.choice(list(config.DIMER_DIFFS.values()))
            similarity_min = similarity_thresholds['dimer']
            if mode == 'positive':
                noise_mz = 2 * (parent_mz - adduct_h) + mz_shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])
            else:
                noise_mz = 2 * (parent_mz + adduct_h) + mz_shift + np.clip(np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["mz_sigma_frac"] * parent_mz), -SYNTHETIC_CONFIG["instrument"]["mz_clip_da"], SYNTHETIC_CONFIG["instrument"]["mz_clip_da"])


        noise_rt = parent_rt + np.random.normal(0, SYNTHETIC_CONFIG["instrument"]["rt_sigma_min"])
        noise_int = parent_int * np.random.uniform(1e-5, 0.5)

        all_mz_peaks = [family['main_peak']] + family.get('isotope_peaks', []) + family.get('adduct_peaks', []) + family.get('dimer_peaks', []) + family.get('noise_peaks', [])
        if mz_exists(noise_mz, all_mz_peaks):
            attempts += 1
            continue
            
        parent_sc_length = int(parent['sc_length'])
        parent_flatness = _peak_to_mean_ratio(parent['intensity_profile'])

        if np.random.rand() < 0.5:
            len_ratio = np.random.uniform(*SYNTHETIC_CONFIG["noise_injection"]["len_ratio_short"])
        else:
            len_ratio = np.random.uniform(*SYNTHETIC_CONFIG["noise_injection"]["len_ratio_long"])
        noise_sc_length = int(np.clip(round(parent_sc_length * len_ratio), SYNTHETIC_CONFIG["false_peaks"]["sc_length_range"][0], SYNTHETIC_CONFIG["false_peaks"]["sc_length_range"][1]))

        for _ in range(max_noise_attempts):
            noise_peak = generate_false_peak(
                noise_mz,
                noise_int,
                noise_rt,
                scan_times=effective_scan_times,
                sc_length=noise_sc_length,
                sampler=sampler,
                mode=mode,
            )

            flat_rt, flat_int = noise_peak['rt_profile'], noise_peak['intensity_profile']
            noise_peak['rt_profile'] = flat_rt
            noise_peak['intensity_profile'] = flat_int
            noise_peak['max_intensity'] = float(np.max(flat_int)) if len(flat_int) > 0 else 0.0
            noise_peak['intensity'] = float(np.trapz(flat_int, flat_rt)) if len(flat_rt) > 1 else float(noise_peak['max_intensity'])
            if len(flat_rt) > 0:
                noise_peak['rt_center'] = float(flat_rt[np.argmax(flat_int)])
            noise_peak['sc_length'] = int(len(flat_rt))
            noise_peak['rt_span'] = float(flat_rt[-1] - flat_rt[0]) if len(flat_rt) > 1 else 0.0

            len_ratio_observed = float(noise_peak['sc_length']) / max(1, parent_sc_length)
            len_different_enough = (len_ratio_observed <= 0.85) or (len_ratio_observed >= 1.15)

            parent_ratio_target = parent_flatness * np.random.uniform(*SYNTHETIC_CONFIG["noise_injection"]["parent_ratio_scale"])
            candidate_flatness = _peak_to_mean_ratio(noise_peak['intensity_profile'])
            is_flatter_than_parent = candidate_flatness <= parent_ratio_target

            sim = _cosine_similarity(parent['rt_profile'], parent['intensity_profile'], noise_peak['rt_profile'], noise_peak['intensity_profile'])
            if len(noise_peak['rt_profile']) >= 4 and sim >= similarity_min and len_different_enough and is_flatter_than_parent:
                noise_peak.update({
                    'type': 'false_peak', 'linked_to': link_type, 'similarity_to_parent': sim, 'relation_type': 'noise_link'
                })
                family['noise_peaks'].append(noise_peak)
                added += 1
                break
        attempts += 1

    family['family_size'] = 1 + len(family['isotope_peaks']) + len(family['adduct_peaks']) + len(family['dimer_peaks']) + len(family['noise_peaks'])
    return family

