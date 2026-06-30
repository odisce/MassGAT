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
from src.models import BaselineHierarchicalGNN, MassGAT, IndependentPeakClassifier
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
    block_type = args.block_type
    baseline_source = args.baseline_source
    baseline_use_singleton_head = args.baseline_use_singleton_head
    output_dir = args.output_dir
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
    target_tsv = output_path.with_name(f"{output_path.stem}_{run_suffix}.tsv")
    output_csv = str(target_tsv)
    
    # check the GPU availability and print its characteristics
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB memory')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    if model_type == 'baseline':
        print('Running baseline model prediction...')
        model = IndependentPeakClassifier(num_features=72, hidden_dim=hidden_dim, encoder_type=encoder_type)
        model.load_state_dict(_load_state_dict(weights_path))
        print(f'Loaded IndependentPeakClassifier weights from {weights_path}')

        model.eval()
        model = model.to(device)
        rows = []
        if os.path.isdir(folder_path):
            folder_contents = os.listdir(folder_path)
            npy_files = [f for f in folder_contents if f.endswith('.npy') and os.path.isfile(os.path.join(folder_path, f))]
            pt_files = [f for f in folder_contents if f.endswith('.pt') and os.path.isfile(os.path.join(folder_path, f))]
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
        df.to_csv(output_csv, index=False, sep='\t', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
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
            data, _ = load_mzml_file(pt_file)
            if data is not None:
                graphs.extend(data)
                file_names.extend([os.path.basename(pt_file)] * len(data))
        else:
            data, _ = load_single_synthetic_file(pt_file)
            if data is not None:
                graphs.append(data)
                file_names.append(os.path.basename(pt_file))

    print(f'Loaded {len(graphs)} graphs for prediction in {time.time() - start_time:.2f} seconds.')

    # Create mapping for XCMS 'sample' ID
    unique_files = list(dict.fromkeys(file_names))
    file_to_sample_idx = {fname: idx + 1 for idx, fname in enumerate(unique_files)}

    # Instantiate model
    model_kwargs = dict(
        num_features=72,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
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

    # Stream CSV writes
    assembly_start = time.time()
    metadata_output_csv = str(target_tsv.with_name(f"{target_tsv.stem}_metadata.tsv"))
    edges_output_csv = str(target_tsv.with_name(f"{target_tsv.stem}_edges.tsv"))

    pred_fields = ['file_name', 'component_id', 'roi_ids', 'num_peaks', 'graph_prob', 'node_probs']
    # XCMS compatible headers
    meta_fields = ['mz', 'mzmin', 'mzmax', 'rt', 'rtmin', 'rtmax', 'into', 'maxo', 'sample', 
                   'file_name', 'component_id', 'roi_id', 'node_prob', 'graph_prob'] +                   [f'top_annotation_{k+1}' for k in range(5)]
    edge_fields = ['file_names', 'edge_index', 'edge_probs']

    n_pred_rows = 0
    n_meta_rows = 0
    n_edge_rows = 0
    t_annotations = 0.0
    t_metadata = 0.0
    t_edges = 0.0

    with open(output_csv, 'w', newline='') as pred_f, open(edges_output_csv, 'w', newline='') as edge_f:
        # Added delimiter='\t' for all writers
        pred_writer = csv.DictWriter(pred_f, fieldnames=pred_fields, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
        edge_writer = csv.DictWriter(edge_f, fieldnames=edge_fields, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
        pred_writer.writeheader()
        edge_writer.writeheader()

        if skip_metadata:
            metadata_cm = None
            meta_writer = None
        else:
            metadata_cm = open(metadata_output_csv, 'w', newline='')
            meta_writer = csv.DictWriter(metadata_cm, fieldnames=meta_fields, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
            meta_writer.writeheader()

        total_files = len(file_names)
        assembly_log_every = max(1, total_files // 20)  # every 5%

        try:
            for i, fname in enumerate(file_names):
                pred = predictions.get(i, {})
                data_obj = graphs[i]

                node_probs = pred.get('node_probs', None)
                graph_probs = pred.get('graph_probs', None)
                
                graph_prob = None
                if graph_probs is not None and len(graph_probs) > 0:
                    graph_prob = float(graph_probs[0])

                roi_ids = data_obj.roi_values.tolist()
                component_id = roi_ids[0] if len(roi_ids) > 0 else -1
                num_peaks = data_obj.num_nodes
                node_features = data_obj.x

                if node_probs is not None:
                    if isinstance(node_probs, torch.Tensor):
                        node_probs_arr = node_probs.detach().cpu().reshape(-1).numpy()
                    else:
                        node_probs_arr = np.asarray(node_probs).reshape(-1)
                    if node_probs_arr.size == 1 and len(roi_ids) > 1:
                        node_probs_arr = np.repeat(node_probs_arr, len(roi_ids))
                else:
                    node_probs_arr = np.zeros(len(roi_ids))

                # align graph_prob to node_prob for singletons/isolated peaks
                if num_peaks == 1 and node_probs_arr.size > 0:
                    graph_prob = float(node_probs_arr[0])

                if graph_prob is not None and graph_prob < args.min_graph_prob:
                    continue

                ranked = []
                # Requirement 3: only components > 1 and <= max_annotation_size
                if (not skip_top_annotations) and data_type == 'real' and 1 < num_peaks <= args.max_annotation_size and node_probs is not None:
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
                            (fname, int(roi_ids[idx])): float(node_probs_arr[idx])
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
                    except Exception as e:
                        print(f"Annotation error in component {component_id}: {e}")
                    t_annotations += time.perf_counter() - ann_start

                if not skip_metadata and meta_writer is not None:
                    meta_start = time.perf_counter()
                    if data_type == 'real':
                        mz_vals = (node_features[:, 0] * 1000 + node_features[:, 6]).detach().cpu().numpy()
                        rt_vals = (node_features[:, 1] * 1800).detach().cpu().numpy()
                        into_vals = torch.expm1(node_features[:, 2]).detach().cpu().numpy()
                        maxo_vals = torch.expm1(node_features[:, 5]).detach().cpu().numpy() if node_features.shape[1] > 5 else torch.expm1(node_features[:, 2]).detach().cpu().numpy()
                        rtmin_vals = (node_features[:, 3] * 60).detach().cpu().numpy()
                        rtmax_vals = (node_features[:, 4] * 60).detach().cpu().numpy()
                    else:
                        mz_vals = node_features[:, 0].detach().cpu().numpy()
                        rt_vals = node_features[:, 1].detach().cpu().numpy()
                        into_vals = node_features[:, 2].detach().cpu().numpy()
                        maxo_vals = np.full(len(roi_ids), np.nan)
                        rtmin_vals = np.full(len(roi_ids), np.nan)
                        rtmax_vals = np.full(len(roi_ids), np.nan)

                    for idx, (roi, mz_v, rt_v, into_v, maxo_v, rtmin_v, rtmax_v, p_v) in enumerate(zip(
                        roi_ids, mz_vals, rt_vals, into_vals, maxo_vals, rtmin_vals, rtmax_vals, node_probs_arr
                    )):
                        # node-level filtering 
                        if p_v < args.min_node_prob:
                            continue

                        # Filtering threshold validation
                        keep_node = (p_v >= args.min_node_prob) or (graph_prob is not None and graph_prob >= args.min_graph_prob)
                        if not keep_node:
                            continue

                        # Requirement 1: Extract string annotations specifically for the current node under each hypothesis k
                        top_ann_cols = {}
                        for k in range(5):
                            if k < len(ranked):
                                top_ann_cols[f'top_annotation_{k+1}'] = ranked[k]['node_annotations'].get(int(roi), '')
                            else:
                                top_ann_cols[f'top_annotation_{k+1}'] = ''

                        meta_writer.writerow({
                            'mz': float(mz_v),
                            'mzmin': float(mz_v),
                            'mzmax': float(mz_v),
                            'rt': float(rt_v),
                            'rtmin': float(rtmin_v) if not np.isnan(rtmin_v) else float(rt_v),
                            'rtmax': float(rtmax_v) if not np.isnan(rtmax_v) else float(rt_v),
                            'into': float(into_v),
                            'maxo': float(maxo_v) if not np.isnan(maxo_v) else float(into_v),
                            'sample': file_to_sample_idx[fname],
                            'file_name': fname,
                            'component_id': component_id,
                            'roi_id': int(roi),
                            'node_prob': float(p_v),
                            'graph_prob': float(graph_prob) if graph_prob is not None else np.nan,
                            **top_ann_cols,
                        })
                        n_meta_rows += 1
                    t_metadata += time.perf_counter() - meta_start

                node_probs_str = str(node_probs_arr.tolist())
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

        pred_f.flush()
        os.fsync(pred_f.fileno())
        edge_f.flush()
        os.fsync(edge_f.fileno())

    written_pred_rows = _count_csv_data_rows(output_csv)
    if written_pred_rows != n_pred_rows:
        print(
            f'WARNING: predicted rows counter ({n_pred_rows}) does not match rows present in file ({written_pred_rows}).'
        )

    total_assembly = time.time() - assembly_start
    if not skip_metadata:
        print(f'Metadata saved to {metadata_output_csv} with {n_meta_rows} rows. Processed in {t_metadata:.2f}s.')
    else:
        print('Metadata export skipped (--skip_metadata).')
    print(f'Predictions saved to {output_csv} with {n_pred_rows} rows.')
    if n_edge_rows > 0:
        print(f'Edge predictions saved to {edges_output_csv} with {n_edge_rows} rows. Processed in {t_edges:.2f}s.')
    else:
        print('No edge predictions available to save.')
    print(f'Annotation total time: {t_annotations:.2f}s.')
    print(f'CSV assembly completed in {total_assembly:.2f} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with trained MassGAT or Baseline model')
    parser.add_argument('--folder', type=str, help='Folder containing .pt files or .npy files')
    parser.add_argument('--data_type', type=str, default='synthetic', choices=['synthetic', 'real'], help='Type of data')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth or .pt)')
    parser.add_argument('--output', type=str, default='predictions.tsv', help='Output CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--model_type', type=str, default='iterative', choices=['baseline', 'iterative', 'baseline_gnn'], help='Model type')
    parser.add_argument('--baseline_source', type=str, default='independent', choices=['independent', 'iterative_cnn'])
    parser.add_argument('--baseline_use_singleton_head', action='store_true', default=True)
    parser.add_argument('--baseline_use_regular_head', dest='baseline_use_singleton_head', action='store_false')
    parser.add_argument('--encoder_type', type=str, default='transformer', choices=['transformer', 'cnn'], help='Encoder type')
    parser.add_argument('--hidden_dim', type=int, default=48, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=24, help='Embedding dimension')
    parser.add_argument('--zero_metadata_all_nodes', action='store_true', default=False)
    parser.add_argument('--block_type', type=str, default='gate', choices=['gate', 'gat', 'gcn'])
    parser.add_argument('--skip_metadata', action='store_true', default=False)
    parser.add_argument('--skip_top_annotations', action='store_true', default=True)
    parser.add_argument('--with_top_annotations', dest='skip_top_annotations', action='store_false')
    
    # New options for filtering and optimization
    parser.add_argument('--min_node_prob', type=float, default=0.0, help='Keep peaks with node_prob >= threshold')
    parser.add_argument('--min_graph_prob', type=float, default=0.5, help='Keep peaks in components with graph_prob >= threshold')
    parser.add_argument('--max_annotation_size', type=int, default=200, help='Skip annotation for components larger than this to prevent bottleneck')
    
    args = parser.parse_args()
    main(args)
