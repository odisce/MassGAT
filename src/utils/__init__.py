"""
Utility functions for MassGAT
"""
from .common import set_seed

# mlflow is imported on first use, it is not needed for inference
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import mlflow_utils
    return getattr(mlflow_utils, name)

__all__ = ['setup_mlflow', 'log_training_params', 'log_training_metrics', 
           'save_and_log_model', 'end_mlflow_run', 'set_seed']
