"""
Loss functions for MassGAT training
"""
import torch
import torch.nn.functional as F



def binary_focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, pos_weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
    pt = torch.exp(-bce_loss) # This is 'p' when target=1, and '1-p' when target=0
    focal_weight = (1 - pt) ** gamma
    
    # Apply class balancing alpha if needed (simplified focal alpha)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = focal_weight * alpha_t
        
    return (focal_weight * bce_loss).mean()



def full_hierarchical_loss(node_logits, graph_logits, edge_logits, edge_index, labels, positive_mask, batch, node_embeddings=None,
                                alpha=1.0, beta=2.0, gamma=1.0, delta=1.0,
                                use_focal=True, focal_gamma=2.0, focal_alpha=0.75,
                                balance_classes=True, hierarchical_consistency=True,
                                label_smoothing=0.0,
                                pos_factor=1.0, is_singleton=None):
    """
    Loss with explicit label logic:
    -1 : False Peak (0.0)
    >=0 : True Peak (1.0)
    -2 : Unknown node in a component (from partially annotated real data)
    """
    device = node_logits.device
    
    # node logic masks
    valid_node_mask = (labels != -2)
    node_targets = (labels >= 0).float()
    
    if label_smoothing > 0.0:
        node_targets = torch.where(node_targets > 0.5, 1.0 - label_smoothing, 0.0)

    unique_graphs = torch.unique(batch)
    num_graphs = len(unique_graphs)
    
    graph_labels = torch.zeros(num_graphs, device=device)
    for i in unique_graphs:
        mask_i = (batch == i)
        if (labels[mask_i] >= 0).any():
            graph_labels[i] = 1.0
            
    if label_smoothing > 0.0:
        graph_labels = torch.where(graph_labels > 0.5, 1.0 - label_smoothing, 0.0)

    graph_sizes = torch.zeros(num_graphs, dtype=torch.long, device=device)
    graph_sizes.scatter_add_(0, batch, torch.ones(batch.size(0), dtype=torch.long, device=device))
    singleton_graph_mask = (graph_sizes == 1)
    non_singleton_mask   = ~singleton_graph_mask

    # 1. NODE LOSS — separate for connected vs singleton
    loss_node = torch.tensor(0.0, device=device)

    if is_singleton is None:
        if edge_index is not None and edge_index.numel() > 0:
            connected = torch.zeros(node_logits.size(0), dtype=torch.bool, device=device)
            connected[edge_index[0]] = True
            connected[edge_index[1]] = True
            is_singleton = ~connected
        else:
            is_singleton = torch.ones(node_logits.size(0), dtype=torch.bool, device=device)

    connected_valid = valid_node_mask & ~is_singleton
    singleton_valid = valid_node_mask & is_singleton

    # Connected node loss
    if connected_valid.sum() > 0:
        w = (node_targets[connected_valid] < 0.5).sum() / (node_targets[connected_valid] > 0.5).sum().clamp(min=1)
        pos_weight_c = w * pos_factor
        if use_focal:
            loss_node_c = binary_focal_loss_with_logits(
                node_logits[connected_valid], node_targets[connected_valid],
                alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight_c)
        else:
            loss_node_c = F.binary_cross_entropy_with_logits(
                node_logits[connected_valid], node_targets[connected_valid],
                pos_weight=pos_weight_c)
    else:
        loss_node_c = torch.tensor(0.0, device=device)

    # Singleton node loss
    if singleton_valid.sum() > 0:
        w_s = (node_targets[singleton_valid] < 0.5).sum() / (node_targets[singleton_valid] > 0.5).sum().clamp(min=1)
        pos_weight_s = w_s * pos_factor * 0.1
        if use_focal:
            loss_node_s = binary_focal_loss_with_logits(
                node_logits[singleton_valid], node_targets[singleton_valid],
                alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight_s)
        else:
            loss_node_s = F.binary_cross_entropy_with_logits(
                node_logits[singleton_valid], node_targets[singleton_valid],
                pos_weight=pos_weight_s)
    else:
        loss_node_s = torch.tensor(0.0, device=device)

    loss_node = loss_node_c + loss_node_s

    # 2. EDGE LOSS
    loss_edge = torch.tensor(0.0, device=device)
    edge_consistency_loss = torch.tensor(0.0, device=device)
    
    if edge_logits is not None and edge_index is not None:
        row, col = edge_index
        valid_edge_mask = (labels[row] != -2) & (labels[col] != -2)
        edge_probs = torch.sigmoid(edge_logits) if edge_logits is not None else None

        if valid_edge_mask.sum() > 0:
            edge_targets = ((labels[row] >= 0) & (labels[col] >= 0)).float()

            if label_smoothing > 0.0:
                edge_targets = torch.where(edge_targets > 0.5, 1.0 - label_smoothing, 0.0)

            valid_edge_logits = edge_logits[valid_edge_mask]
            valid_edge_targets = edge_targets[valid_edge_mask]

            if use_focal:
                loss_edge = binary_focal_loss_with_logits(
                    valid_edge_logits, valid_edge_targets,
                    alpha=0.5, gamma=focal_gamma, pos_weight=None)
            else:
                loss_edge = F.binary_cross_entropy_with_logits(valid_edge_logits, valid_edge_targets)

            if hierarchical_consistency:
                p_u = torch.sigmoid(node_logits[row])[valid_edge_mask]
                p_v = torch.sigmoid(node_logits[col])[valid_edge_mask]
                min_p_nodes = torch.min(p_u, p_v)
                valid_edge_probs = torch.sigmoid(valid_edge_logits)
                edge_consistency_loss = F.relu(valid_edge_probs - min_p_nodes).mean()

    # 3. GRAPH LOSS
    if non_singleton_mask.sum() > 0:
        ns_logits = graph_logits[non_singleton_mask]
        ns_labels = graph_labels[non_singleton_mask]
        pos_weight_g = None
        if balance_classes:
            wg = (ns_labels < 0.5).sum() / (ns_labels > 0.5).sum().clamp(min=1)
            pos_weight_g = wg * pos_factor
        if use_focal:
            loss_graph = binary_focal_loss_with_logits(
                ns_logits, ns_labels,
                alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight_g)
        else:
            loss_graph = F.binary_cross_entropy_with_logits(
                ns_logits, ns_labels, pos_weight=pos_weight_g)
    else:
        loss_graph = torch.tensor(0.0, device=device)

    # 4. CONSISTENCY LOSS
    loss_hier_consistency = torch.tensor(0.0, device=device)
    if hierarchical_consistency and gamma > 0 and non_singleton_mask.sum() > 0:
        p_graph = torch.sigmoid(graph_logits[non_singleton_mask])
        p_nodes_all = torch.sigmoid(node_logits)

        max_node_probs = torch.zeros(num_graphs, device=device)
        try:
            max_node_probs.scatter_reduce_(0, batch, p_nodes_all, reduce="amax", include_self=False)
        except AttributeError:
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.any():
                    max_node_probs[i] = p_nodes_all[mask].max()

        max_node_probs_ns = max_node_probs[non_singleton_mask]
                     
        margin_consistency = 0.1
        loss_hier_consistency = (
            F.relu(max_node_probs_ns - p_graph - margin_consistency) +
            F.relu(p_graph - max_node_probs_ns - margin_consistency)
        ).mean()
    

    # TOTAL LOSS
    total_loss = (alpha * loss_graph) + \
                 (beta * loss_node) + \
                 (delta * loss_edge) + \
                 (gamma * (loss_hier_consistency + edge_consistency_loss))


    # computing some metrics
    with torch.no_grad():
        num_true_graphs = int((graph_labels > 0.5).sum().item())
        num_false_graphs = int((graph_labels < 0.5).sum().item())
        
        num_true_nodes = int(((labels >= 0) & valid_node_mask).sum().item())
        num_false_nodes = int(((labels == -1) & valid_node_mask).sum().item())

        graph_pred_probs = torch.sigmoid(graph_logits)
        avg_true_graph_prob = graph_pred_probs[graph_labels > 0.5].mean() if num_true_graphs > 0 else 0
        avg_false_graph_prob = graph_pred_probs[graph_labels < 0.5].mean() if num_false_graphs > 0 else 0

    return {
        'total_loss': total_loss,
        'graph_loss': loss_graph,
        'node_loss': loss_node,
        'edge_loss': loss_edge,
        'consistency_loss': loss_hier_consistency + edge_consistency_loss,
        'avg_edge_prob': edge_probs.mean() if edge_probs is not None else 0.0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'delta': delta,
        'num_true_graphs': num_true_graphs,
        'num_false_graphs': num_false_graphs,
        'num_true_nodes': num_true_nodes,
        'num_false_nodes': num_false_nodes,
        'avg_true_graph_prob': avg_true_graph_prob,
        'avg_false_graph_prob': avg_false_graph_prob,
    }