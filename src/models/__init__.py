"""
Model definitions for MassGAT
"""
from .massgat import BaselineHierarchicalGNN, MassGAT, FourierScalarEncoder, ProfileTransformerEncoder, CNNPeakEncoder
from .baselines import IndependentPeakClassifier

__all__ = ['BaselineHierarchicalGNN', 'MassGAT', 
           'IndependentPeakClassifier', 'FourierScalarEncoder',
           'ProfileTransformerEncoder', 'CNNPeakEncoder']