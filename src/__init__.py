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

# Modular imports - only import what exists, on first use
def __getattr__(name):
    import importlib
    for module in ('.utils', '.training', '.data', '.data.augmentation'):
        try:
            return getattr(importlib.import_module(module, __name__), name)
        except (ImportError, AttributeError):
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
