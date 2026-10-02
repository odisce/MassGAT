import os
import time
import torch
import argparse
import glob
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from src.models import MassGAT, IndependentPeakClassifier
from src.data.loader import load_single_synthetic_file, load_mzml_file
from src.data.single_loader import SinglePeakDataset, extract_peaks_from_pt
from src.training.trainer import ChroMSTrainer
from src.utils.annotate_component import annotate_component
from src.utils.common import read_feather


def _validated_components(edge_index, valid_mask):
    """Connected components of the sub-graph left once rejected peaks are removed.

    This function aims to break large components by removing low-confidence nodes, helping with the downstream annotation step
    """
    n_nodes = valid_mask.shape[0]
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(n_nodes, n_nodes),
    )
    _, labels = sp.csgraph.connected_components(adj, connection='weak')

    pieces = {}
    for gidx in np.flatnonzero(valid_mask):
        pieces.setdefault(labels[gidx], set()).add(int(gidx))
    return [piece for piece in pieces.values() if len(piece) > 1]


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

def _load_roi_mzmin_mzmax_mapping(ionshed_output_dir, file_name):
    """Load mzmin/mzmax from ionshed feather file, matched by pt filename."""
    if ionshed_output_dir is None or not os.path.isdir(ionshed_output_dir):
        return {}
    
    # Remove .pt extension
    base_name = os.path.splitext(file_name)[0]
    
    # Strip "graph_" prefix if present (common in graph files)
    if base_name.startswith('graph_'):
        base_name = base_name[len('graph_'):]
    
    # If base_name ends with "_signal", replace it with "_peak"
    # Otherwise, just append "_peak"
    if base_name.endswith('_signal'):
        peak_base = base_name[:-len('_signal')] + '_peak'
    else:
        peak_base = base_name + '_peak'
    
    feather_path = os.path.join(ionshed_output_dir, f"{peak_base}.feather")
    
    if not os.path.isfile(feather_path):
        return {}
    
    try:
        df = read_feather(feather_path)
        
        # Create mapping: roi -> (mzmin, mzmax)
        roi_mapping = {}
        if 'roi' in df.columns and 'mzmin' in df.columns and 'mzmax' in df.columns:
            for _, row in df.iterrows():
                roi_id = int(row['roi'])
                roi_mapping[roi_id] = (float(row['mzmin']), float(row['mzmax']))
        return roi_mapping
    except Exception as e:
        print(f"[DEBUG] Failed to load ROI mapping from {feather_path}: {e}")
        return {}

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
    ionshed_output_dir = args.ionshed_output_dir
    
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
                    from src.config import BATCH_SIZE_INFERENCE
                    batch_size = BATCH_SIZE_INFERENCE
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
            from src.config import BATCH_SIZE_INFERENCE
            batch_size = BATCH_SIZE_INFERENCE
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

    # Pre-load ROI mappings from ionshed feather files
    file_to_roi_mapping = {}
    if data_type == 'real' and ionshed_output_dir:
        for fname in unique_files:
            file_to_roi_mapping[fname] = _load_roi_mzmin_mzmax_mapping(ionshed_output_dir, fname)

    # Instantiate model
    model_kwargs = dict(
        num_features=72,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        encoder_type=encoder_type,
        block_type=block_type,
        zero_metadata_all_nodes=zero_metadata_all_nodes,
    )
    
    model = MassGAT(**model_kwargs)


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
    output_per_file = args.output_per_file
    
    # XCMS compatible headers
    meta_fields = ['mz', 'mzmin', 'mzmax', 'rt', 'rtmin', 'rtmax', 'into', 'maxo', 'sample', 
                   'file_name', 'component_id', 'roi_id', 'node_prob', 'graph_prob'] + [f'top_annotation_{k+1}' for k in range(5)]

    n_meta_rows = 0
    t_annotations = 0.0
    t_metadata = 0.0

    # Setup output file handling
    if output_per_file:
        # One file per input file
        file_writers = {}  
        metadata_output_csv = None  
    else:
        # Combined output file
        metadata_output_csv = str(target_tsv.with_name(f"{target_tsv.stem}_peaks.tsv"))
        metadata_cm = open(metadata_output_csv, 'w', newline='')
        meta_writer = csv.DictWriter(metadata_cm, fieldnames=meta_fields, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
        meta_writer.writeheader()
        file_writers = None

    try:
        total_files = len(file_names)
        assembly_log_every = max(1, total_files // 20)  # every 5%
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

                annotations_by_roi = {}
                # The size limit applies to the validated parts of the component
                if (not skip_top_annotations) and data_type == 'real' and num_peaks > 1 and node_probs is not None:
                    ann_start = time.perf_counter()
                    try:
                        node_features_real = node_features.clone()
                        node_features_real[:, 0] = node_features[:, 0] * 1000 + node_features[:, 6]
                        node_features_real[:, 1] = node_features[:, 1] * 1800
                        node_features_real[:, 2] = torch.expm1(node_features[:, 2])
                        node_features_real[:, 3] = node_features[:, 3] * 60
                        node_features_real[:, 4] = node_features[:, 4] * 60

                        edge_index_np = data_obj.edge_index.cpu().numpy()
                        valid_mask = node_probs_arr >= args.min_node_prob
                        keep_edge = valid_mask[edge_index_np[0]] & valid_mask[edge_index_np[1]]
                        keep_edge_t = torch.from_numpy(keep_edge)

                        edge_index_roi = data_obj.roi_values[data_obj.edge_index[:, keep_edge_t]]
                        edge_attr_valid = data_obj.edge_attr[keep_edge_t]
                        node_probs_map = {
                            (fname, int(roi_ids[idx])): float(node_probs_arr[idx])
                            for idx in range(len(roi_ids))
                        }

                        for piece in _validated_components(edge_index_np[:, keep_edge], valid_mask):
                            if len(piece) > args.max_annotation_size:
                                continue
                            ranked = annotate_component(
                                component_gidx_set=piece,
                                node_features=node_features_real,
                                edge_index=edge_index_roi,
                                edge_attr=edge_attr_valid,
                                node_probs=node_probs_map,
                                file_name=fname,
                                roi_values=data_obj.roi_values,
                                prob_threshold=args.min_node_prob,
                                temperature=3.0,
                            )
                            for k, hypothesis in enumerate(ranked[:5]):
                                for roi_key, annotation in hypothesis['node_annotations'].items():
                                    annotations_by_roi.setdefault(int(roi_key), [''] * 5)[k] = annotation
                    except Exception as e:
                        print(f"Annotation error in component {component_id}: {e}")
                    t_annotations += time.perf_counter() - ann_start

                if meta_writer is not None:
                    meta_start = time.perf_counter()
                    roi_mzmin_mzmax_map = file_to_roi_mapping.get(fname, {})
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
                        top_ann_cols = {
                            f'top_annotation_{k+1}': annotation
                            for k, annotation in enumerate(annotations_by_roi.get(int(roi), [''] * 5))
                        }

                        mzmin_v = mz_v
                        mzmax_v = mz_v
                        if int(roi) in roi_mzmin_mzmax_map:
                            mzmin_v, mzmax_v = roi_mzmin_mzmax_map[int(roi)]

                        row_data = {
                            'mz': float(mz_v),
                            'mzmin': mzmin_v,
                            'mzmax': mzmax_v,
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
                        }
                        
                        if output_per_file:
                            # Initialize writer for this file if not already done
                            if fname not in file_writers:
                                output_fname = os.path.join(str(output_dir), fname.replace('.pt', '.tsv'))
                                fm = open(output_fname, 'w', newline='')
                                fw = csv.DictWriter(fm, fieldnames=meta_fields, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
                                fw.writeheader()
                                file_writers[fname] = (fm, fw)
                            file_writers[fname][1].writerow(row_data)
                        else:
                            meta_writer.writerow(row_data)
                        
                        n_meta_rows += 1
                    t_metadata += time.perf_counter() - meta_start

                done = i + 1
                if done % assembly_log_every == 0 or done == total_files:
                    elapsed = time.time() - assembly_start
                    rate = done / max(elapsed, 1e-9)
                    remaining = (total_files - done) / max(rate, 1e-9)
                    pct = done / total_files * 100
                    print(f"[assembly] {pct:.0f}% ({done}/{total_files}) | "
                        f"{rate:.0f} rows/s | elapsed {elapsed:.1f}s | eta {remaining:.1f}s")
                        
    finally:
        if output_per_file:
            # Close all per-file writers
            for fname, (fm, fw) in file_writers.items():
                fm.flush()
                os.fsync(fm.fileno())
                fm.close()
        else:
            # Close combined file
            metadata_cm.flush()
            os.fsync(metadata_cm.fileno())
            metadata_cm.close()

    total_assembly = time.time() - assembly_start
    if output_per_file:
        print(f'Peak data saved to {len(file_writers)} files in {output_dir} with {n_meta_rows} total rows. Processed in {t_metadata:.2f}s.')
    else:
        print(f'Peak data saved to {metadata_output_csv} with {n_meta_rows} rows. Processed in {t_metadata:.2f}s.')
    print(f'Annotation total time: {t_annotations:.2f}s.')
    print(f'CSV assembly completed in {total_assembly:.2f} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with trained MassGAT or Baseline model')
    parser.add_argument('--folder', type=str, help='Folder containing .pt files or .npy files')
    parser.add_argument('--data_type', type=str, default='real', choices=['synthetic', 'real'], help='Type of data')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth or .pt)')
    parser.add_argument('--output', type=str, default='predictions.tsv', help='Output CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--model_type', type=str, default='iterative', choices=['baseline', 'iterative', 'baseline_gnn'], help='Model type')
    parser.add_argument('--baseline_source', type=str, default='independent', choices=['independent', 'iterative_cnn'])
    parser.add_argument('--baseline_use_singleton_head', action='store_true', default=True)
    parser.add_argument('--baseline_use_regular_head', dest='baseline_use_singleton_head', action='store_false')
    parser.add_argument('--encoder_type', type=str, default='transformer', choices=['transformer', 'cnn'], help='Encoder type')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--zero_metadata_all_nodes', action='store_true', default=False)
    parser.add_argument('--block_type', type=str, default='gate', choices=['gate', 'gat', 'gcn'])
    parser.add_argument('--skip_metadata', action='store_true', default=False)
    parser.add_argument('--skip_top_annotations', action='store_true', default=True)
    parser.add_argument('--with_top_annotations', dest='skip_top_annotations', action='store_false')
    parser.add_argument('--output_per_file', action='store_true', default=False, help='Save one peak file per input file (named as input file) instead of combined output')
    
    # Options for filtering and optimization
    parser.add_argument('--min_node_prob', type=float, default=0.5, help='Keep peaks with node_prob >= threshold')
    parser.add_argument('--min_graph_prob', type=float, default=0.0, help='Keep peaks in components with graph_prob >= threshold')
    parser.add_argument('--max_annotation_size', type=int, default=200, help='Skip annotation for validated components larger than this to prevent bottleneck')
    parser.add_argument('--ionshed_output_dir', type=str, default=None, help='Path to ionshed output directory containing _peak.feather files')
    
    args = parser.parse_args()
    main(args)
