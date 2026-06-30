"""
Graph augmentation and subsampling
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def augment_graph(data, aug_type="mixed", intensity_noise_scale=0.2,
                 rt_shift_scale=0.5, mass_jitter_scale=5e-6, profile_noise_std=0.1,
                 feature_mask_prob=0.05, spike_prob=0.1):
    """
    Graph Augmentation.
    4 types of augmentations:
        1. Intensity Jitter
        2. Feature Masking
        3. Edge Noise
        4. RT Shift
    """
    if aug_type == "none":
        return data

    is_batched = hasattr(data, 'batch') and hasattr(data, 'num_graphs')
    
    # Clone data
    x = data.x.clone()
    edge_attr = data.edge_attr.clone() if data.edge_attr is not None else None
    edge_index = data.edge_index.clone()
    
    # [mz, rt, log_into, rtmin, rtmax, log_imax, mz_float, sc_len, ...seqs...]
    idx_log_into = 2
    idx_log_imax = 5
    idx_mz_float = 6
    start_int_seq, end_int_seq = 40, 72
    
    idx_edge_into_ratio = -3 
    idx_edge_imax_ratio = -2

    # 1. intensity jitter
    if aug_type in ["intensity", "mixed"]:
        scale_factors = 1.0 + (torch.rand(x.size(0), device=x.device) * 2 - 1) * intensity_noise_scale
        # to avoid negative or zero intensity issues
        scale_factors = torch.clamp(scale_factors, 0.1, 10.0)
        
        log_scale = torch.log(scale_factors)
        
        x[:, idx_log_into] += log_scale
        x[:, idx_log_imax] += log_scale
        x[:, start_int_seq:end_int_seq] *= scale_factors.unsqueeze(1)
        
        # Update edge attribute columns (last 3 features added in loader)
        if edge_attr is not None:
            row, col = edge_index
            s_row, s_col = log_scale[row], log_scale[col]
            
            # Use negative indexing to target the features you appended in loader
            edge_attr[:, idx_edge_into_ratio] += (s_row - s_col)
            edge_attr[:, idx_edge_imax_ratio] += (s_row - s_col)


    # 2. masking features : randomly zero out intensity/profile for 5% of nodes
    if aug_type in ["masking", "mixed"] and feature_mask_prob > 0:
        mask_indices = torch.rand(x.size(0), device=x.device) < feature_mask_prob
        # no intensity or shape info anymore
        x[mask_indices, idx_log_into] = 0 
        x[mask_indices, idx_log_imax] = 0


    # 3. edge noise
    if aug_type in ["edge_noise", "mixed"] and edge_attr is not None:
        num_static_feats = edge_attr.size(1) - 3
        if num_static_feats > 0:
            noise = torch.randn(edge_attr.size(0), num_static_feats, device=x.device) * 0.01
            edge_attr[:, :num_static_feats] += noise

    # 4. augmentations in rt, mz, intensity, profile
    if aug_type in ["rt_shift", "mixed"]:
        if is_batched:
            shifts = (torch.rand(data.num_graphs, device=x.device) * 2 - 1) * rt_shift_scale
            x[:, 1] += shifts[data.batch]
        else:
            x[:, 1] += (torch.rand(1, device=x.device) * 2 - 1) * rt_shift_scale

    if aug_type in ["mass_jitter", "mixed"]:
        x[:, idx_mz_float] += torch.randn(x.size(0), device=x.device) * mass_jitter_scale

    if aug_type in ["profile_noise", "mixed"]:
        # A. Gaussian noise
        noise = torch.randn(x.size(0), 32, device=x.device) * profile_noise_std
        mask = (x[:, start_int_seq:end_int_seq] > 0)
        x[:, start_int_seq:end_int_seq] += noise * mask
        
        # B. Spikes
        if spike_prob > 0:
            spike_mask = torch.rand(x.size(0), device=x.device) < spike_prob
            spike_noise = (torch.rand(x.size(0), 32, device=x.device) < 0.05).float() * 0.5
            x[:, start_int_seq:end_int_seq] += spike_noise * spike_mask.unsqueeze(1)
            

    # Reconstruct Data Object
    if is_batched:
        aug_data = data.clone()
        aug_data.x = x
        aug_data.edge_attr = edge_attr
        return aug_data
    else:
        # manual reconstruction for single graph
        aug_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                       y=data.y, positive_mask=data.positive_mask, 
                       roi_values=data.roi_values)
        if hasattr(data, 'graph_label'): aug_data.graph_label = data.graph_label
        return aug_data