"""
Configuration file for MassGAT project.
Loads hyperparameters from config.yaml and constants for graph construction, model architecture, and training.
"""

import torch
import yaml
from pathlib import Path

# Load configuration from YAML
def _load_config_yaml():
    config_path = Path(__file__).parent.parent / 'config.yaml'
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
        'ISOTOPE_DIFFS': {},'MOD_DIFFS_POSITIVE': {}, 'MOD_DIFFS_NEGATIVE': {},
        'DIMER_DIFFS_POSITIVE': {}, 'DIMER_DIFFS_NEGATIVE': {},
        'MOD_COMPATIBILITY_POSITIVE': {}, 'MOD_COMPATIBILITY_NEGATIVE': {},
        'ADDUCT_LOSSES': set(),
    }

def _clean(x):
    x = str(x or '')
    return x.replace('\ufeff', '').strip().strip('"\'')

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
        for row in wb['isotopes'].iter_rows(min_row=2, values_only=True):
            name, diff, freq = row[:3]
            if name:
                if diff is not None:
                    cfg['ISOTOPE_DIFFS'][_clean(name)] = float(diff)

    
    # Mods, dimers, compatibility for both modes
    for mode in ('positive', 'negative'):
        # Mods
        sheet = f'mods_{mode}'
        if sheet in wb:
            for row in wb[sheet].iter_rows(min_row=2, values_only=True):
                name, diff, freq, is_loss = row[:4]
                if name:
                    name = _clean(name)
                    if diff is not None:
                        cfg[f'MOD_DIFFS_{mode.upper()}'][name] = float(diff)
                    if is_loss and str(is_loss).lower() in ('1', 'true', 'yes', 'x'):
                        cfg['ADDUCT_LOSSES'].add(name)
        
        # Dimers
        sheet = f'dimers_{mode}'
        if sheet in wb:
            for row in wb[sheet].iter_rows(min_row=2, values_only=True):
                name, diff, freq = row[:3]
                if name:
                    name = _clean(name)
                    if diff is not None:
                        cfg[f'DIMER_DIFFS_{mode.upper()}'][name] = float(diff)
        
        # Compatibility
        sheet = f'compat_{mode}'
        if sheet in wb:
            for row in wb[sheet].iter_rows(min_row=2, values_only=True):
                source, target, weight = row[:3]
                if source and target and weight is not None:
                    source, target = _clean(source), _clean(target)
                    target_dict = cfg[f'MOD_COMPATIBILITY_{mode.upper()}']
                    target_dict.setdefault(source, {})[target] = float(weight)

    
    # Validation
    if not cfg['ISOTOPE_DIFFS'] or not cfg['MOD_DIFFS_POSITIVE'] or not cfg['MOD_DIFFS_NEGATIVE']:
        raise ValueError(f"Missing isotope data in {excel_path}")
    
    return cfg

_TABLE_CFG = load_config()
ISOTOPE_DIFFS = _TABLE_CFG['ISOTOPE_DIFFS']
MOD_DIFFS_POSITIVE = _TABLE_CFG['MOD_DIFFS_POSITIVE']
MOD_DIFFS_NEGATIVE = _TABLE_CFG['MOD_DIFFS_NEGATIVE']
DIMER_DIFFS_POSITIVE = _TABLE_CFG['DIMER_DIFFS_POSITIVE']
DIMER_DIFFS_NEGATIVE = _TABLE_CFG['DIMER_DIFFS_NEGATIVE']
ADDUCT_LOSSES = _TABLE_CFG['ADDUCT_LOSSES']


def detect_ionization_mode(peaks_df, file_id=None):
    """
    Détecte le mode d'ionisation (positif/négatif) à partir des données.
    
    Args:
        peaks_df: DataFrame des pics
        file_id: ID du fichier pour le logging
        
    Returns:
        str: 'positive' ou 'negative'
    """
    peaks_df = peaks_df[peaks_df["file_id"] == file_id].copy()
    # Méthode 1: Vérifier s'il y a une colonne 'file_polarity'
    if 'file_polarity' in peaks_df.columns:
        polarity = peaks_df['file_polarity'].iloc[0]
        mode = 'positive' if polarity == 1 else 'negative'
        if file_id:
            print(f"Mode détecté via file_polarity pour file_id_{file_id}: {mode}")
        return mode
    else:
        print(f"Found columns {peaks_df.columns}")
    
    if file_id:
        print(f"ATTENTION: Mode non détectable pour file_id_{file_id}, utilise mode positif par défaut")
    return 'positive'  # Défaut si détection impossible

def configure_for_mode(mode):
    """
    Configure les différences de masses selon le mode d'ionisation.
    
    Args:
        mode: 'positive' ou 'negative'
    """
    global MASS_DIFFS, ADDUCT_DIFFS, DIMER_DIFFS, ADDUCT_COMPATIBILITY, ADDUCT_FREQUENCIES, DIMER_FREQUENCIES
    
    # Toujours inclure les isotopes
    MASS_DIFFS = ISOTOPE_DIFFS.copy()
    
    if mode == 'positive':
        ADDUCT_DIFFS = MOD_DIFFS_POSITIVE.copy()
        DIMER_DIFFS = DIMER_DIFFS_POSITIVE.copy()
    elif mode == 'negative':  # negative
        ADDUCT_DIFFS = MOD_DIFFS_NEGATIVE.copy()
        DIMER_DIFFS = DIMER_DIFFS_NEGATIVE.copy()
    else:
        raise ValueError(f"Mode inconnu pour la configuration: {mode}")
    
    print(f"Configuration appliquée pour mode {mode}:")
    print(f"  - Adduits: {len(ADDUCT_DIFFS)} types")
    print(f"  - Dimères: {len(DIMER_DIFFS)} types")



if __name__ == '__main__':
    configure_for_mode('positive')
    print(f"Mass differences configurées: {MASS_DIFFS}")
    print(f"Adduct differences configurées: {ADDUCT_DIFFS}")
    print(f"Dimer differences configurées: {DIMER_DIFFS}")