"""
Data loading, graph construction and augmentation
"""
# Imported on first use, so that building graphs does not load torch_geometric
_MODULES = {'augment_graph': '.augmentation', 'build_hierarchical_graph': '.hierarchical_graph_construction'}

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(_MODULES.get(name, '.loader'), __name__), name)

__all__ = [
    'reindex_graph', 
    'augment_graph',
    'load_synthetic_dataset',
    'load_mzml_file',
    'build_hierarchical_graph',
]
