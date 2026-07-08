"""
Model definitions for MassGAT
"""
from .massgat import MassGAT, FourierScalarEncoder, ProfileTransformerEncoder, CNNPeakEncoder
from .baselines import IndependentPeakClassifier

__all__ = ['MassGAT', 
           'IndependentPeakClassifier', 'FourierScalarEncoder',
           'ProfileTransformerEncoder', 'CNNPeakEncoder']