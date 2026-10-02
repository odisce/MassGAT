"""
Common utility functions for MassGAT
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_feather(path):
    """
    Manages the versions of pyarrow and pandas to read feather files.
    """
    import pandas as pd
    import pyarrow as pa
    try:
        with pa.OSFile(str(path), "rb") as f:
            return pa.ipc.open_file(f).read_all().to_pandas()
    except pa.ArrowInvalid:
        return pd.read_feather(path)
