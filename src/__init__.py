"""
MassGAT: Graph Neural Network for LC-MS Data Analysis
"""

__version__ = "1.0.0"
__author__ = "MassGAT Team"

# Import configuration
try:
    from .config import *
except ImportError:
    pass

# Modular imports - only import what exists
try:
    from .data import load_data
    from .data.augmentation import augment_graph
except ImportError:
    pass

try:
    from .training import ChroMSTrainer, contrastive_loss
except ImportError:
    pass

try:
    from .utils import setup_mlflow, log_training_params, log_training_metrics
except ImportError:
    pass

__all__ = [
    # Data handling
    'load_data',
    'augment_graph',
    
    # Models
    'simple_gnn_model',
    'AttentionGNN',
    
    # Training
    'ChroMSTrainer',
    'contrastive_loss',
    
    # Utilities
    'setup_mlflow',
    'log_training_params',
    'log_training_metrics'
]
