import os
import time
import torch
import argparse
import glob
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models.MassGAT import BaselineHierarchicalGNN, MassGAT, IterativeCNNIndependentClassifier
from src.models.baselines import IndependentPeakClassifier
from src.data.loader import load_single_synthetic_file, load_mzml_file
from src.data.single_loader import SinglePeakDataset, extract_peaks_from_pt
from src.training.trainer import ChroMSTrainer
from src.utils.annotate_component import annotate_component


def _load_state_dict(path):
    """Support both raw state_dict and checkpoint dictionaries."""
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
    return checkpoint


def _count_csv_data_rows(csv_path):
    """Fast newline-based data row count (excluding header)."""
    line_count = 0
    with open(csv_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            line_count += chunk.count(b'\n')
    if line_count <= 0:
        return 0
    return max(0, line_count - 1)


def main(args):
    folder_path = args.folder
    weights_path = args.weights
    output_csv = args.output
    data_type = args.data_type
    model_type = args.model_type
    encoder_type = args.encoder_type
    hidden_dim = args.hidden_dim
    embedding_dim = args.embedding_dim
    use_attention = args.use_attention
    memory_efficient = args.memory_efficient
    block_type = args.block_type
    baseline_source = args.baseline_source
    baseline_use_singleton_head = args.baseline_use_singleton_head
    output_dir = args.output_dir
    assembly_log_every = args.assembly_log_every
    skip_metadata = args.skip_metadata
    skip_top_annotations = args.skip_top_annotations
    zero_metadata_all_nodes = args.zero_metadata_all_nodes
    
    # timestamp for output file name (microseconds + pid to avoid collisions)
    timestamp = pd.to_datetime('now').strftime('%Y%m%d_%H%M%S_%f')
    run_suffix = f"{timestamp}_{os.getpid()}"
    if output_dir is None:
        output_dir = os.getenv('CHROMS_RESULTS_DIR', 'results')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_csv)
    if str(output_path.parent) == '.':
        output_path = output_dir / output_path.name
    output_csv = str(output_path.with_name(f"{output_path.stem}_{run_suffix}{output_path.suffix or '.csv'}"))

    # check the GPU availability and print its characteristics (for materials and methods)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB memory')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    if model_type == 'baseline':
        print('Running baseline model prediction...')
        if baseline_source == 'independent':
            model = IndependentPeakClassifier(num_features=72, hidden_dim=hidden_dim, encoder_type=encoder_type)
            model.load_state_dict(_load_state_dict(weights_path))
            print(f'Loaded IndependentPeakClassifier weights from {weights_path}')
        elif baseline_source == 'iterative_cnn':
            iterative_model = MassGAT(
                num_features=72,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                use_attention=use_attention,
                memory_efficient=memory_efficient,
                encoder_type=encoder_type,
                block_type=block_type,
                zero_metadata_all_nodes=zero_metadata_all_nodes,
            )
            iterative_model.load_state_dict(_load_state_dict(weights_path))
            model = IterativeCNNIndependentClassifier(
                iterative_model,
                use_singleton_head=baseline_use_singleton_head
            )
            head_name = 'singleton_node_head' if baseline_use_singleton_head else 'node_head'
            print(f'Loaded MassGAT weights from {weights_path} and using independent CNN pathway with {head_name}.')
        else:
            raise ValueError(f'Unknown baseline_source: {baseline_source}')

        model.eval()
        model = model.to(device)
        rows = []
        # If folder_path is a folder of .npy or .pt files, use SinglePeakDataset or extract_peaks_from_pt
        if os.path.isdir(folder_path):
            folder_contents = os.listdir(folder_path)
            npy_files = [f for f in folder_contents if f.endswith('.npy') and os.path.isfile(os.path.join(folder_path, f))]
            pt_files = [f for f in folder_contents if f.endswith('.pt') and os.path.isfile(os.path.join(folder_path, f))]
            # If .npy files exist, use SinglePeakDataset
            if len(npy_files) > 0:
                dataset = SinglePeakDataset(folder_path)
                loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
                with torch.no_grad():
                    for x, y in loader:
                        x = x.to(device)
                        logits = model(x)
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        for i in range(x.size(0)):
                            rows.append({
                                'peak_index': i,
                                'probability': float(probs[i]),
                                'prediction': int(preds[i]),
                                'true_label': int(y[i])
                            })
            # If .pt files exist, use extract_peaks_from_pt for each .pt file
            elif len(pt_files) > 0:
                for pt_file in pt_files:
                    pt_path = os.path.join(folder_path, pt_file)
                    peaks = extract_peaks_from_pt(pt_path)
                    batch_size = 64
                    all_x = [x for x, roi_id in peaks]
                    all_roi = [roi_id for x, roi_id in peaks]
                    num_peaks = len(all_x)
                    with torch.no_grad():
                        for start in range(0, num_peaks, batch_size):
                            end = min(start + batch_size, num_peaks)
                            x_batch = torch.stack(all_x[start:end]).to(device)
                            roi_batch = all_roi[start:end]
                            logits = model(x_batch)
                            probs = torch.sigmoid(logits)
                            preds = (probs > 0.5).float()
                            for i in range(x_batch.size(0)):
                                rows.append({
                                    'file_name': pt_file,
                                    'peak_index': start + i,
                                    'roi_id': roi_batch[i],
                                    'probability': float(probs[i]),
                                    'prediction': int(preds[i])
                                })
            else:
                print('Error: folder must contain .npy or .pt files for baseline prediction.')
                return
        elif folder_path.endswith('.pt') and os.path.isfile(folder_path):
            # If folder_path is a single .pt file
            peaks = extract_peaks_from_pt(folder_path)
            batch_size = 64
            all_x = [x for x, roi_id in peaks]
            all_roi = [roi_id for x, roi_id in peaks]
            num_peaks = len(all_x)
            file_name = os.path.basename(folder_path)
            with torch.no_grad():
                for start in range(0, num_peaks, batch_size):
                    end = min(start + batch_size, num_peaks)
                    x_batch = torch.stack(all_x[start:end]).to(device)
                    roi_batch = all_roi[start:end]
                    logits = model(x_batch)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    for i in range(x_batch.size(0)):
                        rows.append({
                            'file_name': file_name,
                            'peak_index': start + i,
                            'roi_id': roi_batch[i],
                            'probability': float(probs[i]),
                            'prediction': int(preds[i])
                        })
        else:
            print('Error: folder argument must be a folder with .npy/.pt files or a single .pt file for baseline prediction.')
            return
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False, lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
        print(f'Baseline predictions saved to {output_csv}')
        return

    # List all .pt files in the folder
    pt_files = sorted(glob.glob(os.path.join(folder_path, '*.pt')))
    if not pt_files:
        print(f'No .pt files found in {folder_path}')
        return
    print(f'Found {len(pt_files)} .pt files for prediction.')

    start_time = time.time()
    # Load graphs
    graphs = []
    file_names = []
    for pt_file in tqdm(pt_files, desc="Loading .pt files"):
        if data_type == 'real':
            data, _ = load_mzml_file(pt_file)  # Assuming same loader for simplicity
            if data is not None:
                graphs.extend(data)  # load_mzml_file returns a list of graphs
                file_names.extend([os.path.basename(pt_file)] * len(data))
        else:
            data, _ = load_single_synthetic_file(pt_file)
            if data is not None:
                graphs.append(data)
                file_names.append(os.path.basename(pt_file))

    # print type and number of graphs loaded
    print(f'Loaded {len(graphs)} graphs for prediction in {time.time() - start_time:.2f} seconds.')

    # Instantiate model
    model_kwargs = dict(
        num_features=72,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        use_attention=use_attention,
        memory_efficient=memory_efficient,
        encoder_type=encoder_type,
        block_type=block_type,
        zero_metadata_all_nodes=zero_metadata_all_nodes,
    )
    if model_type == 'baseline_gnn':
        model = BaselineHierarchicalGNN(**model_kwargs)
    elif model_type == 'iterative':
        model = MassGAT(**model_kwargs)
    else:
        raise ValueError('Unknown model_type')

    # Load weights
    model.load_state_dict(_load_state_dict(weights_path))
    print(f'Loaded {model_type} model with weights {weights_path}')

    model.eval()

    # Create trainer
    trainer = ChroMSTrainer(model, device=str(device), model_type=model_type)
    print(f'Running inference on {len(graphs)} graphs...')
    inf_start = time.time()
    predictions = trainer.predict(graphs, batch_size=64)
    print(f'Inference completed in {time.time() - inf_start:.2f} seconds.')

    # Stream CSV writes to avoid very large in-memory lists/dataframes.
    assembly_start = time.time()
    metadata_output_csv = output_csv.replace('.csv', '_metadata.csv')
    edges_output_csv = output_csv.replace('.csv', '_edges.csv')

    pred_fields = ['file_name', 'component_id', 'roi_ids', 'num_peaks', 'graph_prob', 'node_probs']
    meta_fields = ['file_name', 'component_id', 'roi_id', 'mz', 'rt', 'intensity', 'rtmin', 'rtmax', 'node_prob'] + [f'top_annotation_{k+1}' for k in range(5)]
    edge_fields = ['file_names', 'edge_index', 'edge_probs']

    n_pred_rows = 0
    n_meta_rows = 0
    n_edge_rows = 0
    t_annotations = 0.0
    t_metadata = 0.0
    t_edges = 0.0

    with open(output_csv, 'w', newline='') as pred_f, open(edges_output_csv, 'w', newline='') as edge_f:
        pred_writer = csv.DictWriter(pred_f, fieldnames=pred_fields, lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
        edge_writer = csv.DictWriter(edge_f, fieldnames=edge_fields, lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
        pred_writer.writeheader()
        edge_writer.writeheader()

        if skip_metadata:
            metadata_cm = None
            meta_writer = None
        else:
            metadata_cm = open(metadata_output_csv, 'w', newline='')
            meta_writer = csv.DictWriter(metadata_cm, fieldnames=meta_fields, lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
            meta_writer.writeheader()

        total_files = len(file_names)
        assembly_log_every = max(1, total_files // 20)  # every 5%
        assembly_start = time.time()


        try:
            for i, fname in enumerate(file_names):
                pred = predictions.get(i, {})
                data_obj = graphs[i]

                graph_prob = None
                node_probs = pred.get('node_probs', None)
                graph_probs = pred.get('graph_probs', None)
                if graph_probs is not None and len(graph_probs) > 0:
                    graph_prob = graph_probs[0]

                roi_ids = data_obj.roi_values.tolist()
                component_id = roi_ids[0] if len(roi_ids) > 0 else -1
                num_peaks = data_obj.num_nodes
                node_features = data_obj.x

                top_annotations = []
                if (not skip_top_annotations) and data_type == 'real' and num_peaks > 1 and node_probs is not None:
                    ann_start = time.perf_counter()
                    try:
                        node_features_real = node_features.clone()
                        node_features_real[:, 0] = node_features[:, 0] * 1000 + node_features[:, 6]
                        node_features_real[:, 1] = node_features[:, 1] * 1800
                        node_features_real[:, 2] = torch.expm1(node_features[:, 2])
                        node_features_real[:, 3] = node_features[:, 3] * 60
                        node_features_real[:, 4] = node_features[:, 4] * 60

                        edge_index_roi = data_obj.roi_values[data_obj.edge_index]
                        node_probs_map = {
                            (fname, int(roi_ids[idx])): float(node_probs[idx])
                            for idx in range(len(roi_ids))
                        }

                        ranked = annotate_component(
                            component_gidx_set=set(range(num_peaks)),
                            node_features=node_features_real,
                            edge_index=edge_index_roi,
                            edge_attr=data_obj.edge_attr,
                            node_probs=node_probs_map,
                            file_name=fname,
                            roi_values=data_obj.roi_values,
                            prob_threshold=0.5,
                            temperature=3.0,
                        )
                        top_annotations = [str(item.get('node_annotations', {})) for item in ranked[:5]]
                    except Exception as e:
                        top_annotations = [f"annotation_error: {e}"]
                    t_annotations += time.perf_counter() - ann_start

                top_ann_cols = {f'top_annotation_{k+1}': (top_annotations[k] if k < len(top_annotations) else '') for k in range(5)}

                if node_probs is not None:
                    if isinstance(node_probs, torch.Tensor):
                        node_probs_arr = node_probs.detach().cpu().reshape(-1).numpy()
                    else:
                        node_probs_arr = np.asarray(node_probs).reshape(-1)

                    if node_probs_arr.size == 1 and len(roi_ids) > 1:
                        node_probs_arr = np.repeat(node_probs_arr, len(roi_ids))

                    if not skip_metadata and meta_writer is not None:
                        meta_start = time.perf_counter()
                        if data_type == 'real':
                            mz_vals = (node_features[:, 0] * 1000 + node_features[:, 6]).detach().cpu().numpy()
                            rt_vals = (node_features[:, 1] * 1800).detach().cpu().numpy()
                            intensity_vals = torch.expm1(node_features[:, 2]).detach().cpu().numpy()
                            rtmin_vals = (node_features[:, 3] * 60).detach().cpu().numpy()
                            rtmax_vals = (node_features[:, 4] * 60).detach().cpu().numpy()
                        else:
                            mz_vals = node_features[:, 0].detach().cpu().numpy()
                            rt_vals = node_features[:, 1].detach().cpu().numpy()
                            intensity_vals = node_features[:, 2].detach().cpu().numpy()
                            rtmin_vals = np.full(len(roi_ids), np.nan)
                            rtmax_vals = np.full(len(roi_ids), np.nan)

                        for roi, mz_v, rt_v, inten_v, rtmin_v, rtmax_v, p_v in zip(
                            roi_ids, mz_vals, rt_vals, intensity_vals, rtmin_vals, rtmax_vals, node_probs_arr
                        ):
                            meta_writer.writerow({
                                'file_name': fname,
                                'component_id': component_id,
                                'roi_id': int(roi),
                                'mz': float(mz_v),
                                'rt': float(rt_v),
                                'intensity': float(inten_v),
                                'rtmin': float(rtmin_v) if not np.isnan(rtmin_v) else np.nan,
                                'rtmax': float(rtmax_v) if not np.isnan(rtmax_v) else np.nan,
                                'node_prob': float(p_v),
                                **top_ann_cols,
                            })
                            n_meta_rows += 1
                        t_metadata += time.perf_counter() - meta_start

                    node_probs_str = str(node_probs_arr.tolist())
                else:
                    node_probs_str = "[]"

                pred_writer.writerow({
                    'file_name': fname,
                    'component_id': component_id,
                    'roi_ids': str(roi_ids),
                    'num_peaks': num_peaks,
                    'graph_prob': graph_prob,
                    'node_probs': node_probs_str
                })
                n_pred_rows += 1

                edge_start = time.perf_counter()
                edge_index = pred.get('edge_index', None)
                edge_probs = pred.get('edge_probs', None)
                if edge_index is not None and edge_probs is not None:
                    try:
                        ei = np.array(edge_index)
                        for k in range(ei.shape[0]):
                            u = int(ei[k, 0])
                            v = int(ei[k, 1])
                            p = float(edge_probs[k])
                            edge_writer.writerow({'file_names': fname, 'edge_index': str([u, v]), 'edge_probs': p})
                            n_edge_rows += 1
                    except Exception:
                        try:
                            for idx, p in zip(edge_index, edge_probs):
                                u = int(idx[0])
                                v = int(idx[1])
                                edge_writer.writerow({'file_names': fname, 'edge_index': str([u, v]), 'edge_probs': float(p)})
                                n_edge_rows += 1
                        except Exception:
                            pass
                t_edges += time.perf_counter() - edge_start

                done = i + 1
                # Print every 5% (or at the final graph)
                if done % assembly_log_every == 0 or done == total_files:
                    elapsed = time.time() - assembly_start
                    rate = done / max(elapsed, 1e-9)
                    remaining = (total_files - done) / max(rate, 1e-9)
                    pct = done / total_files * 100
                    print(f"[assembly] {pct:.0f}% ({done}/{total_files}) | "
                        f"{rate:.0f} rows/s | elapsed {elapsed:.1f}s | eta {remaining:.1f}s")
                        
        finally:
            if metadata_cm is not None:
                metadata_cm.close()

        # Ensure buffers are flushed all the way to disk (best effort).
        pred_f.flush()
        os.fsync(pred_f.fileno())
        edge_f.flush()
        os.fsync(edge_f.fileno())

    written_pred_rows = _count_csv_data_rows(output_csv)
    if written_pred_rows != n_pred_rows:
        print(
            f'WARNING: predicted rows counter ({n_pred_rows}) does not match rows present in file ({written_pred_rows}). '
            'This can happen with file truncation/network FS issues or if another process wrote the same path.'
        )

    total_assembly = time.time() - assembly_start
    if not skip_metadata:
        print(f'Metadata saved to {metadata_output_csv} with {n_meta_rows} rows.')
    else:
        print('Metadata export skipped (--skip_metadata).')
    print(f'Predictions saved to {output_csv} with {n_pred_rows} rows.')
    if n_edge_rows > 0:
        print(f'Edge predictions saved to {edges_output_csv} with {n_edge_rows} rows.')
    else:
        print('No edge predictions available to save.')
    print(f'CSV assembly completed in {total_assembly:.2f} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with trained ChroMS-GNN or Baseline model')
    parser.add_argument('--folder', type=str, help='Folder containing .pt files or .npy files (or single .pt file for baseline)')
    parser.add_argument('--data_type', type=str, default='synthetic', choices=['synthetic', 'real'], help='Type of data (GNN only)')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth or .pt)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory when --output has no directory part (default: CHROMS_RESULTS_DIR or results/)')
    parser.add_argument('--model_type', type=str, default='iterative', choices=['baseline', 'iterative', 'baseline_gnn'], help='Model type')
    parser.add_argument('--baseline_source', type=str, default='independent', choices=['independent', 'iterative_cnn'], help='Baseline backend: classic independent model or iterative weights with independent CNN path')
    parser.add_argument('--baseline_use_singleton_head', action='store_true', default=True, help='When baseline_source=iterative_cnn, use singleton node head instead of regular node head')
    parser.add_argument('--baseline_use_regular_head', dest='baseline_use_singleton_head', action='store_false', help='When baseline_source=iterative_cnn, use regular node head')
    parser.add_argument('--encoder_type', type=str, default='transformer', choices=['transformer', 'cnn'], help='Encoder type')
    parser.add_argument('--hidden_dim', type=int, default=48, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=24, help='Embedding dimension (GNN only)')
    parser.add_argument('--zero_metadata_all_nodes', action='store_true', default=False, help='Zero out first 8 metadata features for all nodes')
    parser.add_argument('--use_attention', action='store_true', default=True, help='Use attention mechanism (GNN only)')
    parser.add_argument('--memory_efficient', action='store_true', default=False, help='Use memory efficient mode (GNN only)')
    parser.add_argument('--block_type', type=str, default='gate', choices=['gate', 'gat', 'gcn'], help='Type of GNN block for ablation (gate, gat, gcn)')
    parser.add_argument('--assembly_log_every', type=int, default=50000, help='How many graphs between assembly timing logs')
    parser.add_argument('--skip_metadata', action='store_true', default=False, help='Skip writing _metadata.csv (faster for very large runs)')
    parser.add_argument('--skip_top_annotations', action='store_true', default=True, help='Skip costly annotate_component calls in real-data mode (default: True)')
    parser.add_argument('--with_top_annotations', dest='skip_top_annotations', action='store_false', help='Enable annotate_component calls in real-data mode (slow)')
    args = parser.parse_args()
    main(args)