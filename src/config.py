"""
Configuration file for MassGAT project.
Loads hyperparameters from config.yaml and constants for graph construction, model architecture, and training.
"""

import os
import torch
import yaml
from pathlib import Path

# Load configuration from YAML (or from the file given by MASSGAT_CONFIG)
def _load_config_yaml():
    config_path = Path(os.environ.get('MASSGAT_CONFIG') or Path(__file__).parent.parent / 'config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

_CONFIG = _load_config_yaml()

# Graph Construction parameters
RT_WINDOW = _CONFIG['graph']['rt_window']
MZ_TOLERANCE = _CONFIG['graph']['mz_tolerance']
EIC_LEN = _CONFIG['graph']['eic_len']
MIN_SIGNALS = _CONFIG['graph']['min_signals']
MAX_COMPONENT_SIZE = _CONFIG['graph'].get('max_component_size')
COMPONENT_SPLIT = _CONFIG['graph'].get('component_split', 'cut')

# Peak Picking parameters
RTTOL = _CONFIG['peak_picking']['rttol']
PPM = _CONFIG['peak_picking']['ppm']
MZ_RES = _CONFIG['peak_picking']['mz_res']
RT_RES = _CONFIG['peak_picking']['rt_res']
XIC_RTTOL = _CONFIG['peak_picking']['xic_rttol']

# Hierarchical correlation thresholds
ISOTOPE_CORR_THRESHOLD = _CONFIG['thresholds']['isotope_corr']
ADDUCT_CORR_THRESHOLD = _CONFIG['thresholds']['adduct_corr']
DIMER_CORR_THRESHOLD = _CONFIG['thresholds']['dimer_corr']
ISOTOPE_37CL_MIN_RATIO = _CONFIG['thresholds']['isotope_37cl_ratio']
ISOTOPE_LABEL_RULE = _CONFIG['thresholds'].get('isotope_label_rule', 'legacy')

# Model inference parameters
BATCH_SIZE_INFERENCE = _CONFIG['model']['batch_size_inference']

# Random seed for reproducibility
RANDOM_SEED = _CONFIG['system']['random_seed']

# Device configuration
_device_config = _CONFIG['system']['device']
if _device_config == 'auto':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = _device_config


# Feature vector indices (Node feature organization)
# Feature vector layout: [mz, rt, log_into, rtmin, rtmax, log_imax, mz_float, sc_len, ...seqs...]
IDX_LOG_INTO = 2              # Index of log(intensity) feature
IDX_LOG_IMAX = 5              # Index of log(max intensity) feature
IDX_MZ_FLOAT = 6              # Index of m/z float feature
START_INT_SEQ = 40            # Start index of intensity sequence (EIC profile)
END_INT_SEQ = 72              # End index of intensity sequence (EIC profile)
NUM_INT_SEQ = END_INT_SEQ - START_INT_SEQ  # Length of intensity sequence = 32

# Edge feature indices
IDX_EDGE_INTO_RATIO = -3      # Edge intensity ratio (relative indexing)
IDX_EDGE_IMAX_RATIO = -2      # Edge max intensity ratio (relative indexing)

# Data augmentation parameters
INTENSITY_NOISE_SCALE = 0.2   # Scale for intensity jitter augmentation
RT_SHIFT_SCALE = 0.5          # Scale for retention time shift augmentation
MASS_JITTER_SCALE = 5e-6      # Scale for mass jitter augmentation
PROFILE_NOISE_STD = 0.1       # Std for profile Gaussian noise
FEATURE_MASK_PROB = 0.05      # Probability of feature masking (5% of nodes)
SPIKE_PROB = 0.1              # Probability of spike noise


# Common mass differences for isotopes (same for both modes)
from pathlib import Path
from openpyxl import load_workbook

def _empty_cfg():
    return {
        'ISOTOPE_DIFFS': {}, 'ISOTOPE_FREQUENCIES': {},
        'MOD_DIFFS_POSITIVE': {}, 'MOD_DIFFS_NEGATIVE': {},
        'MOD_FREQUENCIES_POSITIVE': {}, 'MOD_FREQUENCIES_NEGATIVE': {},
        'DIMER_DIFFS_POSITIVE': {}, 'DIMER_DIFFS_NEGATIVE': {},
        'DIMER_FREQUENCIES_POSITIVE': {}, 'DIMER_FREQUENCIES_NEGATIVE': {},
        'MOD_COMPATIBILITY_POSITIVE': {}, 'MOD_COMPATIBILITY_NEGATIVE': {},
        'ADDUCT_LOSSES': set(),
        'DISABLED_DIFFS': {'isotopes': set(), 'positive': set(), 'negative': set()},
    }

def _clean(x):
    x = str(x or '')
    return x.replace('\ufeff', '').strip().strip('"\'')

def _sheet_rows(ws):
    """
    Checks whether the mass differences are enabled or disabled in the Excel
    Returns a list of tuples (row, enabled) where 'enabled' is a boolean indicating if the row is enabled.
    """
    rows = list(ws.iter_rows(values_only=True))
    header = [_clean(h).lower() for h in rows[0]]
    idx = header.index('enabled') if 'enabled' in header else None
    out = []
    for row in rows[1:]:
        flag = None if idx is None or idx >= len(row) else row[idx]
        enabled = flag is None or flag == '' or str(flag).strip().strip('"\'').lower() not in ('0','false','no','n','off')
        out.append((row, enabled))
    return out

def load_config(excel_path=None):
    if not excel_path:
        excel_path = Path(__file__).parent / 'mass_differences.xlsx'
    else:
        excel_path = Path(excel_path)
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing config: {excel_path}")
    
    wb = load_workbook(excel_path, data_only=True, read_only=True)
    cfg = _empty_cfg()
    
    # Isotopes
    if 'isotopes' in wb:
        for row, enabled in _sheet_rows(wb['isotopes']):
            name, diff, freq = row[:3]
            if name:
                name = _clean(name)
                if diff is not None:
                    cfg['ISOTOPE_DIFFS'][name] = float(diff)
                if not enabled:
                    cfg['DISABLED_DIFFS']['isotopes'].add(name)
                elif freq is not None:
                    cfg['ISOTOPE_FREQUENCIES'][name] = float(freq)

    
    # Mods, dimers, compatibility for both modes
    for mode in ('positive', 'negative'):
        disabled = cfg['DISABLED_DIFFS'][mode]

        # Mods
        sheet = f'mods_{mode}'
        if sheet in wb:
            for row, enabled in _sheet_rows(wb[sheet]):
                name, diff, freq, is_loss = row[:4]
                if name:
                    name = _clean(name)
                    if diff is not None:
                        cfg[f'MOD_DIFFS_{mode.upper()}'][name] = float(diff)
                    if is_loss and str(is_loss).lower() in ('1', 'true', 'yes', 'vrai', 'x'):
                        cfg['ADDUCT_LOSSES'].add(name)
                    if not enabled:
                        disabled.add(name)
                    elif freq is not None:
                        cfg[f'MOD_FREQUENCIES_{mode.upper()}'][name] = float(freq)
        
        # Dimers
        sheet = f'dimers_{mode}'
        if sheet in wb:
            for row, enabled in _sheet_rows(wb[sheet]):
                name, diff, freq = row[:3]
                if name:
                    name = _clean(name)
                    if diff is not None:
                        cfg[f'DIMER_DIFFS_{mode.upper()}'][name] = float(diff)
                    if not enabled:
                        disabled.add(name)
                    elif freq is not None:
                        cfg[f'DIMER_FREQUENCIES_{mode.upper()}'][name] = float(freq)
        
        # Compatibility
        sheet = f'compat_{mode}'
        if sheet in wb:
            for row, _ in _sheet_rows(wb[sheet]):
                source, target, weight = row[:3]
                if source and target and weight is not None:
                    source, target = _clean(source), _clean(target)
                    if source in disabled or target in disabled:
                        continue
                    target_dict = cfg[f'MOD_COMPATIBILITY_{mode.upper()}']
                    target_dict.setdefault(source, {})[target] = float(weight)

    
    # Validation
    if not cfg['ISOTOPE_DIFFS'] or not cfg['MOD_DIFFS_POSITIVE'] or not cfg['MOD_DIFFS_NEGATIVE']:
        raise ValueError(f"Missing isotope data in {excel_path}")
    
    return cfg

_TABLE_CFG = load_config()
ISOTOPE_DIFFS = _TABLE_CFG['ISOTOPE_DIFFS']
ISOTOPE_FREQUENCIES = _TABLE_CFG['ISOTOPE_FREQUENCIES']
MOD_DIFFS_POSITIVE = _TABLE_CFG['MOD_DIFFS_POSITIVE']
MOD_DIFFS_NEGATIVE = _TABLE_CFG['MOD_DIFFS_NEGATIVE']
MOD_FREQUENCIES_POSITIVE = _TABLE_CFG['MOD_FREQUENCIES_POSITIVE']
MOD_FREQUENCIES_NEGATIVE = _TABLE_CFG['MOD_FREQUENCIES_NEGATIVE']
DIMER_DIFFS_POSITIVE = _TABLE_CFG['DIMER_DIFFS_POSITIVE']
DIMER_DIFFS_NEGATIVE = _TABLE_CFG['DIMER_DIFFS_NEGATIVE']
DIMER_FREQUENCIES_POSITIVE = _TABLE_CFG['DIMER_FREQUENCIES_POSITIVE']
DIMER_FREQUENCIES_NEGATIVE = _TABLE_CFG['DIMER_FREQUENCIES_NEGATIVE']
MOD_COMPATIBILITY_POSITIVE = _TABLE_CFG['MOD_COMPATIBILITY_POSITIVE']
MOD_COMPATIBILITY_NEGATIVE = _TABLE_CFG['MOD_COMPATIBILITY_NEGATIVE']
ADDUCT_LOSSES = _TABLE_CFG['ADDUCT_LOSSES']
DISABLED_DIFFS = _TABLE_CFG['DISABLED_DIFFS']

# Dynamic variables initialized based on mode detection
MASS_DIFFS = ISOTOPE_DIFFS.copy()
ADDUCT_DIFFS = {}  # Will be set by configure_for_mode()
DIMER_DIFFS = {}   # Will be set by configure_for_mode()
ADDUCT_FREQUENCIES = {}  # Will be set by configure_for_mode()
DIMER_FREQUENCIES = {}  # Will be set by configure_for_mode()
ADDUCT_COMPATIBILITY = {}  # Will be set by configure_for_mode()

# Same dicts minus the entries switched off in mass_differences.xlsx. Edge detection
# reads these, everything that encodes or decodes edge type numbers reads the ones
# above, which always hold every entry of the sheet.
ACTIVE_MASS_DIFFS = {}
ACTIVE_ADDUCT_DIFFS = {}
ACTIVE_DIMER_DIFFS = {}


def configure_for_mode(mode):
    """
    Configure les différences de masses selon le mode d'ionisation.
    
    Args:
        mode: 'positive' ou 'negative'
    """
    global MASS_DIFFS, ADDUCT_DIFFS, DIMER_DIFFS, ADDUCT_COMPATIBILITY, ADDUCT_FREQUENCIES, DIMER_FREQUENCIES
    global ACTIVE_MASS_DIFFS, ACTIVE_ADDUCT_DIFFS, ACTIVE_DIMER_DIFFS
    
    # Always include isotopes
    MASS_DIFFS = ISOTOPE_DIFFS.copy()
    
    if mode == 'positive':
        ADDUCT_DIFFS = MOD_DIFFS_POSITIVE.copy()
        DIMER_DIFFS = DIMER_DIFFS_POSITIVE.copy()
        ADDUCT_COMPATIBILITY = MOD_COMPATIBILITY_POSITIVE.copy()
        ADDUCT_FREQUENCIES = MOD_FREQUENCIES_POSITIVE.copy()
        DIMER_FREQUENCIES = DIMER_FREQUENCIES_POSITIVE.copy()
    elif mode == 'negative':
        ADDUCT_DIFFS = MOD_DIFFS_NEGATIVE.copy()
        DIMER_DIFFS = DIMER_DIFFS_NEGATIVE.copy()
        ADDUCT_COMPATIBILITY = MOD_COMPATIBILITY_NEGATIVE.copy()
        ADDUCT_FREQUENCIES = MOD_FREQUENCIES_NEGATIVE.copy()
        DIMER_FREQUENCIES = DIMER_FREQUENCIES_NEGATIVE.copy()
    else:
        raise ValueError(f"Unknown ionization mode: {mode}")

    disabled = DISABLED_DIFFS['isotopes'] | DISABLED_DIFFS[mode]
    ACTIVE_MASS_DIFFS = {k: v for k, v in MASS_DIFFS.items() if k not in disabled}
    ACTIVE_ADDUCT_DIFFS = {k: v for k, v in ADDUCT_DIFFS.items() if k not in disabled}
    ACTIVE_DIMER_DIFFS = {k: v for k, v in DIMER_DIFFS.items() if k not in disabled}
    if disabled:
        print(f"Disabled mass differences ({mode}): {sorted(disabled)}")
    



if __name__ == '__main__':
    configure_for_mode('positive')
    print(f"Configured mass differences: {MASS_DIFFS}")
    print(f"Configured adduct differences: {ADDUCT_DIFFS}")
    print(f"Configured dimer differences: {DIMER_DIFFS}")