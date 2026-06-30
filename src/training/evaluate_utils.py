from sklearn.decomposition import PCA
import torch
import numpy as np
import pandas as pd
import ast

def predictions_to_dataframe(predictions, graphs):
    rows = []
    for i, data_obj in enumerate(graphs):
        pred = predictions.get(i, {})
        # Get probabilities
        graph_prob = pred['graph_probs'][0] if 'graph_probs' in pred and pred['graph_probs'] is not None and len(pred['graph_probs']) > 0 else None
        node_probs = pred['node_probs'] if 'node_probs' in pred else None
        roi_ids = data_obj.roi_values.tolist()
        component_id = roi_ids[0] if len(roi_ids) > 0 else -1
        num_peaks = data_obj.num_nodes
        # Write node_probs and roi_ids as string representations of lists
        if node_probs is not None:
            try:
                arr = node_probs.tolist() if hasattr(node_probs, 'tolist') else node_probs
                if isinstance(arr, (float, int)):
                    arr = [arr]
                node_probs_str = str(arr)
            except Exception:
                node_probs_str = str(node_probs)
        else:
            node_probs_str = "[]"
        roi_ids_str = str(roi_ids)
        rows.append({
            'component_id': component_id,
            'roi_ids': roi_ids_str,
            'num_peaks': num_peaks,
            'graph_prob': graph_prob,
            'node_probs': node_probs_str
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    return df

def get_matches_and_total(predictions, qehf_gt, qehf_graphs):
    predictions_df = predictions_to_dataframe(predictions, qehf_graphs)
    import ast
    # compute node_probs dictionary: roi_id -> prob
    node_probs = {}
    for index, row in predictions_df.iterrows():
        roi_ids = ast.literal_eval(row["roi_ids"])
        node_probs_list = ast.literal_eval(row["node_probs"])
        for roi_id, prob in zip(roi_ids, node_probs_list):
            node_probs[roi_id] = prob
    # match node probabilities with the ground truth
    qehf_gt['node_prob'] = qehf_gt['roi'].apply(lambda roi: node_probs.get(roi, float('nan')))
    # count number of matches higher than 0.5 and 0.8
    gt_probs = qehf_gt['node_prob'].dropna().values
    num_matches_05 = (gt_probs > 0.5).sum()
    num_matches_08 = (gt_probs > 0.8).sum()
    total_true_predictions_05 = sum(1 for prob in node_probs.values() if prob > 0.5)
    total_true_predictions_08 = sum(1 for prob in node_probs.values() if prob > 0.8)
    return num_matches_05, num_matches_08, total_true_predictions_05, total_true_predictions_08



def compute_qehf_peak_pca(model,qehf_graphs,qehf_gt,device='cpu',n_components=3,save_path=None,include_negatives=False,
    n_negatives=1000,random_state=0):
    """
    Compute PCA of node embeddings for peaks in `qehf_gt`.
    """
    model.eval()
    roi_set = set(qehf_gt['roi'].values)

    # First pass: find graphs that contain any ROI of interest
    graphs_with_targets = []
    graphs_candidate_neg = []
    for gi, graph in enumerate(qehf_graphs):
        roi_vals = None
        try:
            roi_vals = graph.roi_values.cpu().numpy()
        except Exception:
            try:
                roi_vals = np.asarray(graph.roi_values)
            except Exception:
                roi_vals = None

        if roi_vals is None:
            continue

        if np.isin(roi_vals, list(roi_set)).any():
            graphs_with_targets.append(gi)
        else:
            graphs_candidate_neg.append(gi)

    if len(graphs_with_targets) == 0 and not include_negatives:
        return None, None, None

    rng = np.random.RandomState(random_state)

    # Prepare negative sampling
    negative_selection = []
    if include_negatives and n_negatives > 0:
        candidates = []
        for gi in graphs_candidate_neg:
            g = qehf_graphs[gi]
            try:
                roi_vals = g.roi_values.cpu().numpy()
            except Exception:
                try:
                    roi_vals = np.asarray(g.roi_values)
                except Exception:
                    roi_vals = None
            if roi_vals is None:
                continue
            for ni in range(len(roi_vals)):
                candidates.append((gi, ni))

        if len(candidates) < n_negatives:
            for gi in graphs_with_targets:
                g = qehf_graphs[gi]
                try:
                    roi_vals = g.roi_values.cpu().numpy()
                except Exception:
                    try:
                        roi_vals = np.asarray(g.roi_values)
                    except Exception:
                        roi_vals = None
                if roi_vals is None:
                    continue
                for ni, rv in enumerate(roi_vals):
                    if rv not in roi_set:
                        candidates.append((gi, ni))

        if len(candidates) > 0:
            chosen = rng.choice(len(candidates), size=min(len(candidates), n_negatives), replace=False)
            chosen_pairs = [candidates[i] for i in chosen]
            per_graph = {}
            for gi, ni in chosen_pairs:
                per_graph.setdefault(gi, []).append(ni)
            negative_selection = list(per_graph.items())

    all_embeddings = []
    all_roi_ids = []
    all_probs = []
    all_is_gt = []

    with torch.no_grad():
        # Process target graphs
        for gi in graphs_with_targets:
            graph = qehf_graphs[gi].to(device)
            output = model(graph)
            node_emb = output['node_embeddings'].cpu().numpy()

            if 'node_probs' in output and output['node_probs'] is not None:
                probs = np.asarray(output['node_probs'].cpu().numpy())
            elif 'node_logits' in output and output['node_logits'] is not None:
                logits = output['node_logits'].cpu()
                try:
                    probs = torch.sigmoid(logits).cpu().numpy()
                except Exception:
                    probs = logits.cpu().numpy()
            else:
                probs = np.full((node_emb.shape[0],), np.nan)

            try:
                roi_ids = graph.roi_values.cpu().numpy()
            except Exception:
                try:
                    roi_ids = np.asarray(graph.roi_values)
                except Exception:
                    roi_ids = np.array([None] * node_emb.shape[0])

            for emb, rid, prob in zip(node_emb, roi_ids, probs):
                try:
                    rint = int(rid)
                except Exception:
                    rint = rid
                if rint in roi_set:
                    all_embeddings.append(emb)
                    all_roi_ids.append(rint)
                    all_probs.append(float(np.asarray(prob).squeeze()) if prob is not None else float('nan'))
                    all_is_gt.append(True)

        # Process sampled negatives
        for gi, node_indices in negative_selection:
            graph = qehf_graphs[gi].to(device)
            output = model(graph)
            node_emb = output['node_embeddings'].cpu().numpy()

            if 'node_probs' in output and output['node_probs'] is not None:
                probs = np.asarray(output['node_probs'].cpu().numpy())
            elif 'node_logits' in output and output['node_logits'] is not None:
                logits = output['node_logits'].cpu()
                try:
                    probs = torch.sigmoid(logits).cpu().numpy()
                except Exception:
                    probs = logits.cpu().numpy()
            else:
                probs = np.full((node_emb.shape[0],), np.nan)

            try:
                roi_ids = graph.roi_values.cpu().numpy()
            except Exception:
                try:
                    roi_ids = np.asarray(graph.roi_values)
                except Exception:
                    roi_ids = np.array([None] * node_emb.shape[0])

            for ni in node_indices:
                if ni < 0 or ni >= node_emb.shape[0]:
                    continue
                emb = node_emb[ni]
                rid = roi_ids[ni]
                try:
                    rint = int(rid)
                except Exception:
                    rint = rid
                all_embeddings.append(emb)
                all_roi_ids.append(rint)
                all_probs.append(float(np.asarray(probs[ni]).squeeze()) if probs is not None else float('nan'))
                all_is_gt.append(False)

    if len(all_embeddings) == 0:
        return None, None, None

    emb_matrix = np.stack(all_embeddings)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(emb_matrix)

    if save_path is not None:
        df = pd.DataFrame({
            'roi_id': all_roi_ids,
            'pc1': pcs[:, 0],
            'pc2': pcs[:, 1],
            'pc3': pcs[:, 2] if pcs.shape[1] > 2 else np.nan,
            'prob': all_probs,
            'is_gt': all_is_gt,
        })
        df.to_csv(save_path, index=False)

    return pcs, pca, all_roi_ids


def compute_training_pca(model, train_graphs, n_graphs=1000, n_components=3, 
                          save_path=None, device='cpu', random_state=42):
    """
    Compute PCA of node embeddings for a sample of the training set.
    Designed for synthetic data where all nodes have clean ground truth labels:
        label >= 0  -> True peak
        label == -1 -> False peak / noise
        label == -2 -> Unknown
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    model.eval()

    true_graphs = []
    false_graphs = []
    for i, g in enumerate(train_graphs):
        labels = g.y.cpu().numpy() if hasattr(g.y, 'cpu') else np.asarray(g.y)
        if (labels >= 0).any():
            true_graphs.append(i)
        else:
            false_graphs.append(i)

    rng = np.random.default_rng(random_state)

    # we take up to n_graphs/2 from each class
    n_true = min(len(true_graphs), n_graphs // 2)
    n_false = min(len(false_graphs), n_graphs - n_true)

    sampled_true = rng.choice(true_graphs, size=n_true, replace=False).tolist() if n_true > 0 else []
    sampled_false = rng.choice(false_graphs, size=n_false, replace=False).tolist() if n_false > 0 else []
    sampled_indices = sampled_true + sampled_false

    all_embeddings = []
    all_labels = []       
    all_probs = []        
    all_roi_ids = []
    all_graph_ids = []  
    all_graph_sizes = []  
    all_file_ids = []      

    with torch.no_grad():
        for gi in sampled_indices:
            g = train_graphs[gi].to(device)
            output = model(g)

            node_emb = output['node_embeddings'].cpu().numpy()
            node_logits = output['node_logits'].cpu()
            probs = torch.sigmoid(node_logits).numpy()

            labels = g.y.cpu().numpy() if hasattr(g.y, 'cpu') else np.asarray(g.y)

            try:
                roi_ids = g.roi_values.cpu().numpy() if hasattr(g.roi_values, 'cpu') else np.asarray(g.roi_values)
            except AttributeError:
                roi_ids = np.arange(len(labels))  # fallback if no roi_values

            n_nodes = node_emb.shape[0]

            file_id_current = getattr(g, 'file_id', None)
            if file_id_current is None:
                file_id_current = getattr(g, 'file_name', None)
            try:
                if hasattr(file_id_current, 'item'):
                    file_id_current = file_id_current.item()
            except Exception:
                pass


            for ni in range(n_nodes):
                all_embeddings.append(node_emb[ni])
                all_labels.append(int(labels[ni]))
                all_probs.append(float(probs[ni]))
                all_roi_ids.append(roi_ids[ni] if ni < len(roi_ids) else -1)
                all_graph_ids.append(gi)
                all_graph_sizes.append(n_nodes)
                all_file_ids.append(file_id_current)

    if len(all_embeddings) == 0:
        print("[compute_training_pca] No embeddings collected.")
        return None, None, None

    emb_matrix = np.stack(all_embeddings)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    # --- PCA ---
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(emb_matrix)

    explained = pca.explained_variance_ratio_
    print(f"[PCA] Explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}" +
          (f", PC3={explained[2]:.3f}" if n_components > 2 else "") +
          f"  |  Total: {explained.sum():.3f}")


    label_category = []
    for lbl, prob in zip(labels_arr, probs_arr):
        if lbl >= 0:
            label_category.append('true')
        elif lbl == -1:
            label_category.append('false')
        else:
            label_category.append('unknown')  # -2

    prob_bucket = []
    for p in probs_arr:
        if p < 0.1:
            prob_bucket.append('very_low (<0.1)')
        elif p < 0.35:
            prob_bucket.append('low (0.1-0.35)') 
        elif p < 0.65:
            prob_bucket.append('uncertain (0.35-0.65)')
        elif p < 0.9:
            prob_bucket.append('high (0.65-0.9)')
        else:
            prob_bucket.append('very_high (>0.9)')

    # Save
    df = pd.DataFrame({
        'roi_id':         all_roi_ids,
        'graph_id':       all_graph_ids,
        'file_id':        all_file_ids,
        'graph_size':     all_graph_sizes,
        'label_raw':      all_labels,
        'label_category': label_category,
        'prob':           probs_arr,
        'prob_bucket':    prob_bucket,
        'pc1': pcs[:, 0],
        'pc2': pcs[:, 1],
        'pc3': pcs[:, 2] if pcs.shape[1] > 2 else np.nan,
    })

    print("\n[PCA] Label × Probability bucket cross-tab (counts):")
    print(pd.crosstab(df['label_category'], df['prob_bucket']))
    print("\n[PCA] Mean prob per label category:")
    print(df.groupby('label_category')['prob'].describe()[['mean','std','min','max']])

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"[PCA] Saved to {save_path}")

    return pcs, pca, df


def compute_combined_pca(
    model,train_graphs,qehf_graphs,qehf_gt, n_train_graphs=500, n_qehf_random=5000, n_components=3,save_path_nodes=None,
    save_path_graphs=None,device='cpu',random_state=42,
):
    """
    Combined PCA of node embeddings and graph embeddings mixing:
      - QEHF ground truth nodes (832 known true)
      - QEHF random nodes (sampled from non-GT graphs)
      - Training nodes
    Returns two DataFrames with PCA coordinates and metadata for nodes and graphs.
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    model.eval()
    rng = np.random.default_rng(random_state)

    roi_set = set(qehf_gt['roi'].values)


    # Storage for nodes
    node_embeddings_list = []
    node_rows = [] 
    # Storage for graphs
    graph_embeddings_list = []
    graph_rows = []

    #  A. QEHF graphs
    # Separate GT-containing graphs from the rest
    qehf_gt_graph_indices      = []
    qehf_nongт_graph_indices   = []

    for gi, g in enumerate(qehf_graphs):
        try:
            rv = g.roi_values.cpu().numpy()
        except Exception:
            rv = np.asarray(g.roi_values)
        if np.isin(rv, list(roi_set)).any():
            qehf_gt_graph_indices.append(gi)
        else:
            qehf_nongт_graph_indices.append(gi)

    # GT-containing graphs (keep all GT nodes + their companions)
    with torch.no_grad():
        for gi in qehf_gt_graph_indices:
            g = qehf_graphs[gi].to(device)
            out = model(g)

            node_emb   = out['node_embeddings'].cpu().numpy()   # [N, D]
            node_logits = out['node_logits'].cpu()
            probs       = torch.sigmoid(node_logits).numpy()
            g_emb  = np.concatenate([node_emb.mean(0), node_emb.max(0)])
            g_prob = torch.sigmoid(out['graph_logits']).cpu().item() \
                     if out['graph_logits'].numel() == 1 \
                     else torch.sigmoid(out['graph_logits']).cpu().numpy().mean()

            try:
                roi_ids = g.roi_values.cpu().numpy()
            except Exception:
                roi_ids = np.asarray(g.roi_values)

            n_nodes = node_emb.shape[0]

            for ni in range(n_nodes):
                try:
                    rint = int(roi_ids[ni])
                except Exception:
                    rint = None
                is_gt = (rint in roi_set) if rint is not None else False
                source = 'qehf_gt' if is_gt else 'qehf_random'

                node_embeddings_list.append(node_emb[ni])
                node_rows.append(dict(
                    roi_id        = rint,
                    graph_size    = n_nodes,
                    prob          = float(probs[ni]),
                    source        = source,
                    file_id       = None,
                    label_raw     = None,
                    label_category= None,
                ))

            graph_embeddings_list.append(g_emb)
            graph_rows.append(dict(
                graph_id   = f'qehf_{gi}',
                n_nodes    = n_nodes,
                graph_prob = g_prob,
                source     = 'qehf',
                file_id    = None,
                graph_label= None,
            ))

    # Random QEHF nodes from non-GT graphs ---
    candidates = []
    for gi in qehf_nongт_graph_indices:
        g = qehf_graphs[gi]
        try:
            n = g.x.shape[0]
        except Exception:
            continue
        for ni in range(n):
            candidates.append((gi, ni))

    n_sample = min(len(candidates), n_qehf_random)
    chosen_idx = rng.choice(len(candidates), size=n_sample, replace=False)
    chosen_pairs = [candidates[i] for i in chosen_idx]

    # Group by graph to avoid re-running model per node
    per_graph_random = {}
    for gi, ni in chosen_pairs:
        per_graph_random.setdefault(gi, []).append(ni)

    with torch.no_grad():
        for gi, node_indices in per_graph_random.items():
            g = qehf_graphs[gi].to(device)
            out = model(g)

            node_emb    = out['node_embeddings'].cpu().numpy()
            node_logits = out['node_logits'].cpu()
            probs       = torch.sigmoid(node_logits).numpy()

            g_emb  = np.concatenate([node_emb.mean(0), node_emb.max(0)])
            g_prob = torch.sigmoid(out['graph_logits']).cpu().item() \
                     if out['graph_logits'].numel() == 1 \
                     else torch.sigmoid(out['graph_logits']).cpu().numpy().mean()

            try:
                roi_ids = g.roi_values.cpu().numpy()
            except Exception:
                roi_ids = np.asarray(g.roi_values)

            n_nodes = node_emb.shape[0]

            for ni in node_indices:
                if ni >= n_nodes:
                    continue
                try:
                    rint = int(roi_ids[ni])
                except Exception:
                    rint = None

                node_embeddings_list.append(node_emb[ni])
                node_rows.append(dict(
                    roi_id        = rint,
                    graph_size    = n_nodes,
                    prob          = float(probs[ni]),
                    source        = 'qehf_random',
                    file_id       = None,
                    label_raw     = None,
                    label_category= None,
                ))

            # Only add graph once even if we sampled multiple nodes from it
            graph_embeddings_list.append(g_emb)
            graph_rows.append(dict(
                graph_id   = f'qehf_{gi}',
                n_nodes    = n_nodes,
                graph_prob = g_prob,
                source     = 'qehf',
                file_id    = None,
                graph_label= None,
            ))

    # B. Training graphs
    train_true_idx  = []
    train_false_idx = []
    for i, g in enumerate(train_graphs):
        lbl = g.y.cpu().numpy() if hasattr(g.y, 'cpu') else np.asarray(g.y)
        if (lbl >= 0).any():
            train_true_idx.append(i)
        else:
            train_false_idx.append(i)

    n_true  = min(len(train_true_idx),  n_train_graphs // 2)
    n_false = min(len(train_false_idx), n_train_graphs - n_true)

    sampled_train = (
        rng.choice(train_true_idx,  size=n_true,  replace=False).tolist() +
        rng.choice(train_false_idx, size=n_false, replace=False).tolist()
    )

    with torch.no_grad():
        for gi in sampled_train:
            g = train_graphs[gi].to(device)
            out = model(g)

            node_emb    = out['node_embeddings'].cpu().numpy()
            node_logits = out['node_logits'].cpu()
            probs       = torch.sigmoid(node_logits).numpy()

            g_emb  = np.concatenate([node_emb.mean(0), node_emb.max(0)])
            g_prob = torch.sigmoid(out['graph_logits']).cpu().item() \
                     if out['graph_logits'].numel() == 1 \
                     else torch.sigmoid(out['graph_logits']).cpu().numpy().mean()

            labels = g.y.cpu().numpy() if hasattr(g.y, 'cpu') else np.asarray(g.y)

            try:
                roi_ids = g.roi_values.cpu().numpy() \
                          if hasattr(g.roi_values, 'cpu') else np.asarray(g.roi_values)
            except AttributeError:
                roi_ids = np.arange(len(labels))

            file_id = getattr(g, 'file_id', None) or getattr(g, 'file_name', None)
            try:
                if hasattr(file_id, 'item'):
                    file_id = file_id.item()
            except Exception:
                pass

            n_nodes = node_emb.shape[0]

            # Graph-level label: true if any node >= 0
            g_label = int((labels >= 0).any())

            for ni in range(n_nodes):
                lbl_i = int(labels[ni])
                if lbl_i >= 0:
                    cat = 'true'
                elif lbl_i == -1:
                    cat = 'false'
                else:
                    cat = 'unknown'

                try:
                    rint = int(roi_ids[ni])
                except Exception:
                    rint = None

                node_embeddings_list.append(node_emb[ni])
                node_rows.append(dict(
                    roi_id        = rint,
                    graph_size    = n_nodes,
                    prob          = float(probs[ni]),
                    source        = 'train',
                    file_id       = file_id,
                    label_raw     = lbl_i,
                    label_category= cat,
                ))

            graph_embeddings_list.append(g_emb)
            graph_rows.append(dict(
                graph_id   = f'train_{gi}',
                n_nodes    = n_nodes,
                graph_prob = g_prob,
                source     = 'train',
                file_id    = file_id,
                graph_label= g_label,
            ))

    #  PCA — nodes
    if len(node_embeddings_list) == 0:
        print("[combined PCA] No node embeddings collected.")
        return None, None

    node_matrix = np.stack(node_embeddings_list)
    pca_node = PCA(n_components=n_components)
    node_pcs  = pca_node.fit_transform(node_matrix)

    ev = pca_node.explained_variance_ratio_
    print(f"[Node PCA] Explained variance: "
          f"PC1={ev[0]:.3f}, PC2={ev[1]:.3f}"
          + (f", PC3={ev[2]:.3f}" if n_components > 2 else "")
          + f"  |  Total: {ev.sum():.3f}")

    node_df = pd.DataFrame(node_rows)
    node_df['pc1'] = node_pcs[:, 0]
    node_df['pc2'] = node_pcs[:, 1]
    node_df['pc3'] = node_pcs[:, 2] if n_components > 2 else np.nan

    if save_path_nodes is not None:
        node_df.to_csv(save_path_nodes, index=False)
        print(f"[Node PCA] Saved to {save_path_nodes}")

    #  PCA — graphs
    seen_graph_ids = set()
    unique_graph_embs = []
    unique_graph_rows = []
    for emb, row in zip(graph_embeddings_list, graph_rows):
        gid = row['graph_id']
        if gid not in seen_graph_ids:
            seen_graph_ids.add(gid)
            unique_graph_embs.append(emb)
            unique_graph_rows.append(row)

    graph_matrix = np.stack(unique_graph_embs)
    pca_graph = PCA(n_components=n_components)
    graph_pcs  = pca_graph.fit_transform(graph_matrix)

    ev_g = pca_graph.explained_variance_ratio_
    print(f"[Graph PCA] Explained variance: "
          f"PC1={ev_g[0]:.3f}, PC2={ev_g[1]:.3f}"
          + (f", PC3={ev_g[2]:.3f}" if n_components > 2 else "")
          + f"  |  Total: {ev_g.sum():.3f}")

    graph_df = pd.DataFrame(unique_graph_rows)
    graph_df['pc1'] = graph_pcs[:, 0]
    graph_df['pc2'] = graph_pcs[:, 1]
    graph_df['pc3'] = graph_pcs[:, 2] if n_components > 2 else np.nan

    if save_path_graphs is not None:
        graph_df.to_csv(save_path_graphs, index=False)
        print(f"[Graph PCA] Saved to {save_path_graphs}")

    return node_df, graph_df