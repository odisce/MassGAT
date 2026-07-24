"""
Configuration for synthetic data generation.
Imports chemistry constants from the parent config module.
"""

try:
    # Try package-style import first (when installed as package)
    from ..config import (
        # Chemistry constants from mass_differences.xlsx
        ISOTOPE_DIFFS,
        ISOTOPE_FREQUENCIES,
        MOD_DIFFS_POSITIVE,
        MOD_DIFFS_NEGATIVE,
        MOD_FREQUENCIES_POSITIVE,
        MOD_FREQUENCIES_NEGATIVE,
        DIMER_DIFFS_POSITIVE,
        DIMER_DIFFS_NEGATIVE,
        DIMER_FREQUENCIES_POSITIVE,
        DIMER_FREQUENCIES_NEGATIVE,
        MOD_COMPATIBILITY_POSITIVE,
        MOD_COMPATIBILITY_NEGATIVE,
        ADDUCT_LOSSES,
        MASS_DIFFS,
        ADDUCT_DIFFS,
        DIMER_DIFFS,
        ADDUCT_FREQUENCIES,
        DIMER_FREQUENCIES,
        ADDUCT_COMPATIBILITY,
        configure_for_mode,
        # Graph parameters
        RT_WINDOW,
        MZ_TOLERANCE,
        EIC_LEN,
        # Correlation thresholds
        ISOTOPE_CORR_THRESHOLD,
        ADDUCT_CORR_THRESHOLD,
        DIMER_CORR_THRESHOLD,
    )
except ImportError:
    # Fallback: try absolute import (when run as script from src/)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import (
        ISOTOPE_DIFFS,
        ISOTOPE_FREQUENCIES,
        MOD_DIFFS_POSITIVE,
        MOD_DIFFS_NEGATIVE,
        MOD_FREQUENCIES_POSITIVE,
        MOD_FREQUENCIES_NEGATIVE,
        DIMER_DIFFS_POSITIVE,
        DIMER_DIFFS_NEGATIVE,
        DIMER_FREQUENCIES_POSITIVE,
        DIMER_FREQUENCIES_NEGATIVE,
        MOD_COMPATIBILITY_POSITIVE,
        MOD_COMPATIBILITY_NEGATIVE,
        ADDUCT_LOSSES,
        MASS_DIFFS,
        ADDUCT_DIFFS,
        DIMER_DIFFS,
        ADDUCT_FREQUENCIES,
        DIMER_FREQUENCIES,
        ADDUCT_COMPATIBILITY,
        configure_for_mode,
        RT_WINDOW,
        MZ_TOLERANCE,
        EIC_LEN,
        ISOTOPE_CORR_THRESHOLD,
        ADDUCT_CORR_THRESHOLD,
        DIMER_CORR_THRESHOLD,
    )

# Re-export for backward compatibility
__all__ = [
    'ISOTOPE_DIFFS',
    'ISOTOPE_FREQUENCIES',
    'MOD_DIFFS_POSITIVE',
    'MOD_DIFFS_NEGATIVE',
    'MOD_FREQUENCIES_POSITIVE',
    'MOD_FREQUENCIES_NEGATIVE',
    'DIMER_DIFFS_POSITIVE',
    'DIMER_DIFFS_NEGATIVE',
    'DIMER_FREQUENCIES_POSITIVE',
    'DIMER_FREQUENCIES_NEGATIVE',
    'MOD_COMPATIBILITY_POSITIVE',
    'MOD_COMPATIBILITY_NEGATIVE',
    'ADDUCT_LOSSES',
    'MASS_DIFFS',
    'ADDUCT_DIFFS',
    'DIMER_DIFFS',
    'ADDUCT_FREQUENCIES',
    'DIMER_FREQUENCIES',
    'ADDUCT_COMPATIBILITY',
    'configure_for_mode',
    'RT_WINDOW',
    'MZ_TOLERANCE',
    'EIC_LEN',
    'ISOTOPE_CORR_THRESHOLD',
    'ADDUCT_CORR_THRESHOLD',
    'DIMER_CORR_THRESHOLD',
]
