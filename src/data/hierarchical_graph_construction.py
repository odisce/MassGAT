"""
Hierarchical graph construction for MassGAT
"""

import numpy as np
import pandas as pd
import time
import networkx as nx
import os


from src.config import MIN_SIGNALS
try:
    from src import config
except ModuleNotFoundError:
    import config

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

# Global variable to control progress bars
_SHOW_PROGRESS = True

def set_progress_bars(enabled=True):
    """Enable or disable progress bars globally."""
    global _SHOW_PROGRESS
    _SHOW_PROGRESS = enabled

def maybe_tqdm(iterable, desc=""):
    """If tqdm is not available, print a message."""
    if _SHOW_PROGRESS:
        try:
            from tqdm import tqdm
            return tqdm(iterable, desc=desc)
        except ImportError:
            print(f"{desc}...")
            return iterable
    else:
        return iterable


log_file = os.getenv('CHROMS_GRAPH_LOG_FILE', 'logs/graph_processing_progress.log')

def log_progress(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f'[{timestamp}] {message}'
    print(log_msg, flush=True)
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
            f.flush()
    except Exception as e:
        print(f"[LOGGING ERROR] {e}")

def _table_rows(df, columns):
    """Rows of df as tuples, with the same values as iterrows."""
    if df.empty:
        return []
    return zip(*(df[c].tolist() for c in columns))


def _first_at_offset(mz, rows, offset, inclusive):
    """For each i of rows, first k with mz[k] - mz[i] >= offset (inclusive) or > offset, mz sorted.

    mz[k] - mz[i] is computed as in a loop over k, so that the bounds do not depend on rounding.
    """
    n = len(mz)
    k = np.searchsorted(mz, mz[rows] + offset - 1e-6, side='left')
    todo = np.arange(len(rows))
    while todo.size:
        todo = todo[k[todo] < n]
        d = mz[k[todo]] - mz[rows[todo]]
        behind = d < offset[todo] if inclusive else d <= offset[todo]
        todo = todo[behind]
        k[todo] += 1
    return k


def collect_candidate_peak_pairs(peaks_df, mz_deltas, dimer_specs=None, mz_tolerance=None, rt_tolerance=None):
    import time
    step_start = time.time()
    """
    Collect all (roi_source, roi_target, delta) pairs that match m/z and RT criteria.
    
    Args:
        peaks_df: DataFrame with peak information  
        mz_deltas: Dict of mass differences {name: delta_value}
        dimer_specs: List of (multiplier, shift) tuples for dimer detection
        mz_tolerance: m/z tolerance (defaults to MZ_TOLERANCE)
        rt_tolerance: RT tolerance (defaults to RT_WINDOW)
    
    Returns:
        peaks_df_sorted: Updated DataFrame with match columns
        candidate_pairs: List of (roi_source, roi_target, delta_name) tuples
    """
    if mz_tolerance is None:
        mz_tolerance = config.MZ_TOLERANCE
    if rt_tolerance is None:
        rt_tolerance = config.RT_WINDOW

    peaks_df["mzmed"] = peaks_df["mzmed"].astype("float64")
    peaks_df_sorted = peaks_df.copy()
    peaks_df_sorted["mzmed"] = peaks_df_sorted["mzmed"].astype("float64")
    peaks_df_sorted["rtmeanw"] = peaks_df_sorted["rtmeanw"].astype("float64")
    peaks_df_sorted = peaks_df_sorted.sort_values("mzmed").reset_index(drop=True)
    mz = peaks_df_sorted["mzmed"].astype("float64").values
    rt = peaks_df_sorted["rtmeanw"].astype("float64").values
    roi = peaks_df_sorted["roi"].values
    n = len(peaks_df_sorted)

    # Initialize match columns for mz_deltas
    for delta_name in mz_deltas.keys():
        peaks_df_sorted[f"match_{delta_name}"] = np.nan

    candidate_pairs = []

    # Detect regular mass differences: for each peak i, the peaks k with delta - tol <= mz[k] - mz[i] <= delta + tol
    first = np.arange(n - 1) if n > 1 else np.zeros(0, dtype=np.int64)
    for delta_name, delta in mz_deltas.items():
        if n < 2:
            continue
        start = np.maximum.accumulate(_first_at_offset(mz, first, np.full(n - 1, delta - mz_tolerance), inclusive=True))
        stop = np.maximum(start, _first_at_offset(mz, first, np.full(n - 1, delta + mz_tolerance), inclusive=False))
        count = stop - start
        ii = np.repeat(first, count)
        kk = np.repeat(start, count) + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))
        keep = np.abs(rt[kk] - rt[ii]) <= rt_tolerance
        ii, kk = ii[keep], kk[keep]
        match = peaks_df_sorted[f"match_{delta_name}"].to_numpy(dtype=np.float64, copy=True)
        for i, k, roi_i, roi_k in zip(ii.tolist(), kk.tolist(), roi[ii], roi[kk]):
            # Store both (i,k) and (k,i) to cover bidirectional matching
            candidate_pairs.append((roi_i, roi_k, delta_name))
            candidate_pairs.append((roi_k, roi_i, delta_name))
            # Update the DataFrame with the matched ROIs
            match[k] = roi_i
            match[i] = roi_k
        peaks_df_sorted[f"match_{delta_name}"] = match

    # Detect dimers (need to multiply by 2)
    if dimer_specs:
        for mult, shift in dimer_specs:
            target_col = f"dimer_{mult:.3f}x+{shift:.5f}"
            peaks_df_sorted[target_col] = np.nan
            
            i = 0
            j = 0
            while i < n - 1:
                target = mult * mz[i] - shift
                while j < n and mz[j] < target - mz_tolerance:
                    j += 1
                k = j
                while k < n and mz[k] <= target + mz_tolerance:
                    if abs(mz[k] - target) <= mz_tolerance and abs(rt[k] - rt[i]) <= rt_tolerance:
                        # Store both (i,k) and (k,i) to cover bidirectional matching
                        candidate_pairs.append((roi[i], roi[k], target_col))
                        candidate_pairs.append((roi[k], roi[i], target_col))
                        # Update the DataFrame with the matched ROIs
                        peaks_df_sorted.at[k, target_col] = roi[i]
                        peaks_df_sorted.at[i, target_col] = roi[k]
                    k += 1
                i += 1

    print(f"[TIMER] collect_candidate_peak_pairs took {time.time() - step_start:.4f}s")
    return peaks_df_sorted, candidate_pairs



# Profiles cover whole blocks of the grid, so that np.dot on an overlap sums as on the full grid (same bits)
PROFILE_BLOCK = 64


def precompute_intensity_profiles(signals_df, roi_list, rt_grid=None):
    import time
    step_start = time.time()
    """
    Fast profile precomputation using pre-grouped data.
    
    Args:
        signals_df: DataFrame with signal data (roi, rt, i columns)
        roi_list: List of ROI values to process
        rt_grid: Optional RT grid for interpolation (computed if None)
    
    Returns:
        rt_grid: Common RT grid used for interpolation
        profiles: Dict mapping roi -> (start, values), the interpolated intensity profile on
            rt_grid[start:start + len(values)], zero elsewhere on the grid
    """
    if not roi_list:
        return np.array([]), {}
    
    # Filter signals to only include ROIs we need
    filter_start = time.time()
    roi_set = set(roi_list)
    signals_filtered = signals_df[signals_df['roi'].isin(roi_set)]
    filter_time = time.time() - filter_start
    
    if signals_filtered.empty:
        return np.array([]), {roi: (0, np.array([])) for roi in roi_list}
    
    # Sort signals by RT once
    sort_start = time.time()
    signals_filtered = signals_filtered.sort_values("rt")
    sort_time = time.time() - sort_start

    # For a given roi id and rt, take max intensity if multiple intensities are present
    groupby_start = time.time()
    signals_filtered = signals_filtered.groupby(['roi', 'rt'], as_index=False).agg({'i': 'max'})
    groupby_time = time.time() - groupby_start
    
    # Create RT grid
    grid_start = time.time()
    if rt_grid is None:
        rt_grid = np.unique(signals_filtered['rt'].values)
    grid_time = time.time() - grid_start

    # Pre-group signals by ROI (the table is sorted by roi, then rt)
    pregroup_start = time.time()
    roi_arr = signals_filtered['roi'].to_numpy()
    rt_arr = signals_filtered['rt'].to_numpy()
    i_arr = signals_filtered['i'].to_numpy()
    bounds = np.flatnonzero(np.r_[True, roi_arr[1:] != roi_arr[:-1], True])
    roi_slices = dict(zip(roi_arr[bounds[:-1]].tolist(), zip(bounds[:-1].tolist(), bounds[1:].tolist())))
    pregroup_time = time.time() - pregroup_start
    
    print(f"  Preprocessing: filter={filter_time:.2f}s, sort={sort_time:.2f}s, groupby={groupby_time:.2f}s, grid={grid_time:.2f}s, pregroup={pregroup_time:.2f}s")

    # Interpolate on the blocks of the common RT grid covering each trace
    interp_start = time.time()
    profiles = {}
    interp_count = 0
    n_grid = len(rt_grid)
    for roi in roi_list:
        if roi in roi_slices:
            a, b = roi_slices[roi]
            start = np.searchsorted(rt_grid, rt_arr[a], side='left') // PROFILE_BLOCK * PROFILE_BLOCK
            stop = np.searchsorted(rt_grid, rt_arr[b - 1], side='right')
            stop = min(n_grid, (stop + PROFILE_BLOCK - 1) // PROFILE_BLOCK * PROFILE_BLOCK)
            profiles[roi] = (start, np.interp(rt_grid[start:stop], rt_arr[a:b], i_arr[a:b], left=0, right=0))
            interp_count += 1
        else:
            profiles[roi] = (0, np.array([]))
    
    interp_time = time.time() - interp_start
    print(f"  Interpolation: {interp_time:.2f}s for {interp_count}/{len(roi_list)} ROIs (RT grid size: {len(rt_grid)})")

    print(f"[TIMER] precompute_intensity_profiles took {time.time() - step_start:.4f}s")
    return rt_grid, profiles


def compute_profile_correlations(candidate_pairs, signals_df, min_corr=0.6):
    import time
    step_start = time.time()
    """
    Fast cosine similarity computation after precomputing intensity profiles.
    
    Args:
        candidate_pairs: List of (roi_i, roi_j, delta_name) tuples
        signals_df: DataFrame with signal data
        min_corr: Minimum correlation threshold
    
    Returns:
        DataFrame with correlation results
    """
    if not candidate_pairs:
        return pd.DataFrame(columns=["roi_i", "roi_k", "delta", "correlation"])
    
    # Get all unique ROIs (optimized using set comprehension)
    all_rois = {roi for pair in candidate_pairs for roi in pair[:2]}

    # Precompute profiles
    precompute_start = time.time()
    rt_grid, profiles = precompute_intensity_profiles(signals_df, all_rois)
    precompute_time = time.time() - precompute_start
    print(f"Profile precomputation: {precompute_time:.2f}s for {len(all_rois)} ROIs")

    # Precompute norms
    norms = {}
    for roi in all_rois:
        norms[roi] = np.linalg.norm(profiles[roi][1])

    # Compute correlations
    results = []
    failed_candidates = []
    for roi_i, roi_j, delta in candidate_pairs:
        norm_i = norms.get(roi_i, 0.0)
        norm_j = norms.get(roi_j, 0.0)
        if norm_i == 0 or norm_j == 0:
            continue
        # Correlation computed on the overlap of the two profiles
        start_i, profile_i = profiles[roi_i]
        start_j, profile_j = profiles[roi_j]
        lo = max(start_i, start_j)
        hi = min(start_i + len(profile_i), start_j + len(profile_j))
        dot = np.dot(profile_i[lo - start_i:hi - start_i], profile_j[lo - start_j:hi - start_j]) if hi > lo else 0.0
        corr = dot / (norm_i * norm_j)
        if corr >= min_corr:
            results.append({
                "roi_i": roi_i,
                "roi_k": roi_j,
                "delta": delta,
                "correlation": corr
            })
        else:
            failed_candidates.append((roi_i, roi_j, delta, corr))

    print(f"[TIMER] compute_profile_correlations took {time.time() - step_start:.4f}s")
    return pd.DataFrame(results)




def filter_duplicate_edge_types(edges, peaks_df):
    """
    Filters duplicate edge types symmetrically, allowing up to two edges 
    of the same type per node in the same direction
    """
    if not edges:
        return edges

    best_target = {}
    
    # Pre-extract as NumPy arrays for blazing fast O(1) lookups
    mz_vals = peaks_df['mzmed'].values
    into_vals = peaks_df['into'].values
    
    # 1. Find the best target for each node, edge type, AND direction
    for src, dst, edge_type, corr in edges:
        src_mz = mz_vals[src]
        dst_mz = mz_vals[dst]
        
        src_into = into_vals[src]
        dst_into = into_vals[dst]
        
        # Determine relative direction (+1 for higher m/z, -1 for lower m/z)
        if dst_mz > src_mz:
            src_dir, dst_dir = 1, -1
        elif dst_mz < src_mz:
            src_dir, dst_dir = -1, 1
        else:
            # Fallback tie-breaker if m/z is exactly identical
            src_dir = 1 if dst > src else -1
            dst_dir = -src_dir
            
        # Record what 'src' thinks is the best 'dst' in this direction
        key_src = (src, edge_type, src_dir)
        if key_src not in best_target:
            best_target[key_src] = (dst, dst_into)
        else:
            curr_dst, curr_dst_into = best_target[key_src]
            if dst_into > curr_dst_into or (dst_into == curr_dst_into and dst > curr_dst):
                best_target[key_src] = (dst, dst_into)
                
        # Record what 'dst' thinks is the best 'src' in its respective direction
        key_dst = (dst, edge_type, dst_dir)
        if key_dst not in best_target:
            best_target[key_dst] = (src, src_into)
        else:
            curr_src, curr_src_into = best_target[key_dst]
            if src_into > curr_src_into or (src_into == curr_src_into and src > curr_src):
                best_target[key_dst] = (src, src_into)
                
    # 2. Symmetrically filter edges
    filtered_edges = []
    for edge in edges:
        src, dst, edge_type, corr = edge
        
        src_mz = mz_vals[src]
        dst_mz = mz_vals[dst]
        
        # Re-determine direction
        if dst_mz > src_mz:
            src_dir, dst_dir = 1, -1
        elif dst_mz < src_mz:
            src_dir, dst_dir = -1, 1
        else:
            src_dir = 1 if dst > src else -1
            dst_dir = -src_dir
            
        src_wants = best_target[(src, edge_type, src_dir)][0]
        dst_wants = best_target[(dst, edge_type, dst_dir)][0]
        
        # Keep the edge ONLY if both nodes agree it's their best connection for that direction
        if src_wants == dst and dst_wants == src:
            filtered_edges.append(edge)
            
    return filtered_edges


def build_hierarchical_graph(peaks_df, raw_signals, signals_df, file_id=None, rt_window=None, include_profile_correlations=False, ionization_mode = None, 
                             show_progress=True, limit_component_size_value=None, debug_hanging=False,
                             eic_extraction_mode='uniform'):
    import time
    global_step_start = time.time()

    log_progress(f"build_hierarchical_graph: START (peaks={len(peaks_df)}, signals={len(signals_df)})")

    """
    Build graph using hierarchical approach:
    1. Isotope detection (similarity > 0.6, isotope m/z < main ion m/z)
    2. Adduct detection on main ions (similarity > 0.85)
    3. Dimer detection on main ions (similarity > 0.85)
    4. Profile correlation on remaining unconnected peaks (optional, similarity > 0.9)
    
    Args:
        peaks_df: DataFrame with peak information
        raw_signals: Dict mapping roi to raw (rt, intensity) signal data
        signals_df: DataFrame with original signal data (roi, rt, i columns)
        file_id: File identifier for ground truth loading (e.g., 1 for file_id_1)
        rt_window: Optional tuple (rt_min, rt_max) to filter peaks by RT range
        include_profile_correlations: Whether to add profile correlation edges (default False)
        show_progress: Whether to show progress bars (default True)
        limit_component_size_value: value of maximum component size allowed(default None)
        debug_hanging: Whether to enable debug logging for hanging detection (default False)
        eic_extraction_mode: EIC extraction strategy for node features.
            - 'uniform' (default): uniform subsampling with apex preservation
            - 'center_rt_window': keep original points around rtmeanw center (windowed)
    Returns:
        edge_index: Tensor of shape [2, num_edges]
        edge_attr: Tensor of edge attributes
        node_features: Tensor of node features
    """
    # Set progress bar visibility
    set_progress_bars(show_progress)
    
    start_time = time.time()
    
    # Detect ionization mode and configure mass differences
    if ionization_mode is None:
        true_peaks = pd.read_csv(os.getenv('CHROMS_TRUE_PEAKS_PATH', 'data/raw/true_peaks.tsv'), sep='\t')
        ionization_mode = config.detect_ionization_mode(true_peaks, file_id)
        config.configure_for_mode(ionization_mode)
    else:
        config.configure_for_mode(ionization_mode)
    
    if show_progress:
        print(f"Detected ionization mode: {ionization_mode}")

    # Filter peaks that have less than MIN_SIGNALS in their signal clusters
    if "sc_length" in peaks_df.columns:
                peaks_df = peaks_df[peaks_df["sc_length"] >= MIN_SIGNALS]
                filtered_rois = set(peaks_df["roi"])
                signals_df = signals_df[signals_df["roi"].isin(filtered_rois)]
    
    # Filter peaks by RT window if specified
    if rt_window is not None:
        rt_min, rt_max = rt_window
        rt_mask = (peaks_df['rtmeanw'] >= rt_min) & (peaks_df['rtmeanw'] <= rt_max)
        peaks_df = peaks_df[rt_mask].reset_index(drop=True)
        
        # Filter signals to match filtered peaks
        filtered_rois = set(peaks_df['roi'])
        signals_df = signals_df[signals_df['roi'].isin(filtered_rois)]
        
        print(f"RT window [{rt_min:.2f}, {rt_max:.2f}]: {len(peaks_df)} peaks remaining")
    
    if show_progress:
        print(f"Building hierarchical graph for {len(peaks_df)} peaks...")
    
    # Prepare data structures
    prep_start = time.time()
    rois = peaks_df['roi'].values
    roi_to_idx = {roi: idx for idx, roi in enumerate(rois)}
    rt_values = peaks_df['rtmeanw'].values
    
    # Check which ROIs have valid signal data
    valid_rois = set(raw_signals.keys())
    valid_eic_mask = np.array([roi in valid_rois for roi in rois])
    
    edges = []
    isotope_nodes = set()  # Track only isotopes 
    main_ions = set(range(len(rois)))  # Will be reduced as isotopes are found
    
    if show_progress:
        print("Step 1: Detecting isotope patterns...")
    log_progress("Starting isotope detection")

    iso_start = time.time()
    isotope_groups = detect_isotope_patterns(
        peaks_df, signals_df, show_progress=show_progress
    )
    print(f"[TIMER] detect_isotope_patterns took {time.time() - iso_start:.4f}s")
    log_progress(f"[TIMER] detect_isotope_patterns took {time.time() - iso_start:.4f}s")

    # Mark only isotopes as processed (main ions remain available for adduct/dimer detection)
    for group in isotope_groups:
        main_roi = group['main_roi']  # ROI value of main ion
        isotope_data = group['isotopes']  # List of dicts with roi and edge_type
        
        # Convert main ROI to DataFrame index for edge creation
        main_idx = roi_to_idx.get(main_roi, -1)
        if main_idx == -1:
            continue  # Skip if main ROI not found
        
        # Add isotope edges with correct edge type
        for iso_data in isotope_data:
            iso_roi = iso_data['roi']  # ROI value of isotope
            iso_idx = roi_to_idx.get(iso_roi, -1)
            if iso_idx != -1:  # Only add if isotope ROI found
                edges.append((main_idx, iso_idx, iso_data['edge_type'], iso_data['correlation']))
                isotope_nodes.add(iso_idx)  # Only isotopes are excluded from further processing
        
        # Only remove isotopes from main_ions set
        isotope_rois = [iso_data['roi'] for iso_data in isotope_data]
        isotope_indices = [roi_to_idx.get(roi, -1) for roi in isotope_rois]
        main_ions.difference_update([idx for idx in isotope_indices if idx != -1])
    
    iso_time = time.time() - iso_start
    if show_progress:
        print(f"Found {len(isotope_groups)} isotope groups, {len(main_ions)} main ions remaining")

    if show_progress:
        print("Step 2: Detecting adduct patterns...")
    

    adduct_start = time.time()
    adduct_edges = detect_adduct_patterns(
        peaks_df, raw_signals, valid_eic_mask, roi_to_idx, main_ions, signals_df, show_progress
    )
    print(f"[TIMER] detect_adduct_patterns took {time.time() - adduct_start:.4f}s")
    log_progress(f"[TIMER] detect_adduct_patterns took {time.time() - adduct_start:.4f}s")

    edges.extend(adduct_edges)
    adduct_time = time.time() - adduct_start
    
    if show_progress:
        print("Step 3: Detecting dimer patterns...")
    

    dimer_start = time.time()
    dimer_edges = detect_dimer_patterns(
        peaks_df, roi_to_idx, main_ions, signals_df, show_progress, ionization_mode=ionization_mode
    )
    print(f"[TIMER] detect_dimer_patterns took {time.time() - dimer_start:.4f}s")
    log_progress(f"[TIMER] detect_dimer_patterns took {time.time() - dimer_start:.4f}s")
    
    edges.extend(dimer_edges)
    dimer_time = time.time() - dimer_start


    edges = filter_duplicate_edge_types(edges, peaks_df)

    if show_progress:
        print(f"Total edges created: {len(edges)}")
    
    # Limit component sizes (optional)
    if limit_component_size_value is None:
        limit_component_size_value = config.MAX_COMPONENT_SIZE
    if limit_component_size_value is not None:
        print("Step 5: Limiting component sizes...")
        log_progress("Step 5: Limiting component sizes...")
        limit_start = time.time()
        edges, split_metadata = limit_component_size(edges, peaks_df, K=limit_component_size_value,
                                                     method=config.COMPONENT_SPLIT)
        limit_time = time.time() - limit_start
        print(f"Component size limiting: {limit_time:.2f}s")
        log_progress(f"Component size limiting: {limit_time:.2f}s")
    else:
        print("Step 5: Component size limiting disabled")
        limit_time = 0
        split_metadata = {'components_processed': 0, 'components_split': 0, 'max_component_size': 0}
    print(f"Final edges after limiting: {len(edges)}")
    log_progress(f"Final edges after limiting: {len(edges)}")
    

    # Convert to PyTorch format
    convert_start = time.time()
    result = convert_to_pytorch_format(
        edges,
        peaks_df,
        raw_signals,
        signals_df,
        debug_hanging=debug_hanging,
        file_id=file_id,
        show_progress=show_progress,
        eic_extraction_mode=eic_extraction_mode,
    )
    edge_index, edge_attr, node_features, roi_values = result
    

    total_time = time.time() - start_time

    # Print timing summary
    print(f"\nGraph construction completed in {total_time:.2f}s")
    print(f"[TIMER] build_hierarchical_graph total: {time.time() - global_step_start:.4f}s")
    print(f"Isotope: {iso_time:.2f}s | Adduct: {adduct_time:.2f}s | Dimer: {dimer_time:.2f}s")
    print(f"Optimized correlation computation enabled")
    

    return edge_index, edge_attr, node_features, roi_values


# m/z shift and isotope / monoisotopic intensity ratio per atom (IUPAC abundances)
HEAVY_ISOTOPE_SHIFTS = {'isotope_34S': 1.995796, 'isotope_37Cl': 1.997050}
K41_SHIFT = 1.998119
S34_RATIO, K41_RATIO = 0.0447, 0.0722


def heavy_isotope_label(shift, ratio, mz_tol, min_37cl_ratio=0.2):
    """Edge type of an M+2 pair near 2 Da from its m/z shift and intensity ratio, None when neither fits.

    37Cl (0.32 per Cl) above min_37cl_ratio, 34S from half the ratio of one S up to that threshold, and 41K
    (one K, within a factor 2) kept as a 34S edge, the type the published model was trained with for it.
    """
    def near(ref):
        return abs(shift - ref) <= mz_tol
    if near(HEAVY_ISOTOPE_SHIFTS['isotope_37Cl']) and ratio >= min_37cl_ratio:
        return 'isotope_37Cl'
    if near(HEAVY_ISOTOPE_SHIFTS['isotope_34S']) and S34_RATIO / 2 <= ratio < min_37cl_ratio:
        return 'isotope_34S'
    if near(K41_SHIFT) and K41_RATIO / 2 <= ratio <= K41_RATIO * 2:
        return 'isotope_34S'
    return None


def detect_isotope_patterns(peaks_df, signals_df, show_progress=True):
    import time
    step_start = time.time()
    """
    Detect isotopic patterns.
    
    Method:
    1. Use optimized candidate collection for all isotope mass differences
    2. Validate with intensity hierarchy and cosine similarity  
    3. Build isotope chains (13C2 -> 13C -> main ion)
    4. Return only main ions for further adduct/dimer processing
    """
    if show_progress:
        print("Collecting isotope candidates...")

    
    log_progress("Starting isotope candidate collection")
    # Time candidate collection step
    candidate_start = time.time()
    peaks_sorted, candidate_pairs = collect_candidate_peak_pairs(
        peaks_df, config.ACTIVE_MASS_DIFFS, mz_tolerance=config.MZ_TOLERANCE, rt_tolerance=config.RT_WINDOW
    )
    candidate_time = time.time() - candidate_start

    if show_progress:
        print(f"Found {len(candidate_pairs)} isotope candidate pairs in {candidate_time:.2f}s")
    log_progress(f"Found {len(candidate_pairs)} isotope candidate pairs in {candidate_time:.2f}s")
    
    # Time correlation computation step
    if show_progress:
        print("Computing correlations for isotope candidates...")
    correlation_start = time.time()
    correlation_results = compute_profile_correlations(
        candidate_pairs, signals_df, min_corr=config.ISOTOPE_CORR_THRESHOLD
    )
    correlation_time = time.time() - correlation_start
    
    if show_progress:
        print(f"Found {len(correlation_results)} isotope pairs above correlation threshold in {correlation_time:.2f}s")
    
    # Validate isotope pairs with intensity ratio
    validated_pairs = []
    min_37cl_ratio = getattr(config, 'ISOTOPE_37CL_MIN_RATIO', 0.2)
    ratio_rule = config.ISOTOPE_LABEL_RULE == 'ratio'
    heavy_pairs = set()
    
    roi_to_peak = {roi: {'mzmed': mz, 'into': into} for roi, mz, into in
                   zip(peaks_df['roi'].to_numpy(), peaks_df['mzmed'].to_numpy(), peaks_df['into'].to_numpy())}
    
    for roi_i, roi_k, delta_name, correlation in _table_rows(correlation_results, ['roi_i', 'roi_k', 'delta', 'correlation']):
        # Enforce isotope intensity hierarchy: lower m/z (main) must have higher intensity
        
        # Fast peak lookup using pre-built dictionary
        peak_i = roi_to_peak.get(roi_i)
        peak_k = roi_to_peak.get(roi_k)
        
        if peak_i is None or peak_k is None:
            continue
        
        # Determine if the ion with lower mz has a higher intensity than the isotope
        if peak_i["mzmed"] < peak_k["mzmed"]:
            if peak_i['into'] > peak_k['into']:
                main_roi, iso_roi = roi_i, roi_k
                main_into, iso_into = peak_i['into'], peak_k['into']
            else:
                continue  # Skip: isotope has higher intensity than main peak
        else:
            if peak_i['into'] < peak_k['into']:
                main_roi, iso_roi = roi_k, roi_i
                main_into, iso_into = peak_k['into'], peak_i['into']
            else:
                continue  # Skip: isotope has higher intensity than main peak

        if ratio_rule and delta_name in HEAVY_ISOTOPE_SHIFTS:
            # one decision per pair, whichever of the 34S / 37Cl rows came first
            if (main_roi, iso_roi) in heavy_pairs:
                continue
            heavy_pairs.add((main_roi, iso_roi))
            delta_name = heavy_isotope_label(abs(peak_k['mzmed'] - peak_i['mzmed']), iso_into / main_into,
                                             config.MZ_TOLERANCE, min_37cl_ratio)
            if delta_name is None:
                continue
        # Extra 37Cl filter (34S and 37Cl are close in m/z, so the intensity ratio can help distinguish them)
        elif delta_name == 'isotope_37Cl':
            if (iso_into / main_into) < min_37cl_ratio:
                # replace with 34S isotope
                delta_name = 'isotope_34S'

        validated_pairs.append({
            'main_roi': main_roi,
            'iso_roi': iso_roi,
            'iso_name': delta_name,
            'correlation': correlation
        })
    
    if show_progress:
        print(f"Validated {len(validated_pairs)} isotope pairs with intensity hierarchy")
    
    # Group pairs into isotope families (same logic as before)
    print(f"[TIMER] detect_isotope_patterns total took {time.time() - step_start:.4f}s")
    return group_isotope_pairs(validated_pairs)


def group_isotope_pairs(all_pairs):
    """Group isotope pairs into families with main ions.
    """
    # Build adjacency list using ROI values
    main_to_isotopes = {}
    used_isotopes = set()
    
    for pair in all_pairs:
        main_roi = pair['main_roi']  # ROI value of main ion
        iso_roi = pair['iso_roi']    # ROI value of isotope
        
        # Skip if isotope already used in another group
        if iso_roi in used_isotopes:
            continue
            
        if main_roi not in main_to_isotopes:
            main_to_isotopes[main_roi] = []
            
        main_to_isotopes[main_roi].append({
            'roi': iso_roi,  # Store ROI value instead of DataFrame index
            'edge_type': pair['iso_name'],
            'correlation': pair['correlation']
        })
        used_isotopes.add(iso_roi)
    
    # Convert to isotope groups format (now using ROI values)
    isotope_groups = []
    for main_roi, isotopes in main_to_isotopes.items():
        if isotopes:  # Only add groups with isotopes
            isotope_groups.append({
                'main_roi': main_roi,  # Use ROI value instead of DataFrame index
                'isotopes': isotopes
            })
    
    return isotope_groups



def detect_adduct_patterns(peaks_df, raw_signals, valid_eic_mask, roi_to_idx, main_ions, signals_df, show_progress=False):
    import time
    step_start = time.time()
    """
    Detect adduct patterns between main ions using optimized correlation computation.
    """
    edges = []
    main_ions_list = list(main_ions)
    
    if len(main_ions_list) < 2:
        return edges
    
    if show_progress:
        print(f"Detecting adduct patterns among {len(main_ions_list)} main ions...")
    
    # Time DataFrame subset creation
    subset_start = time.time()
    main_peaks_df = peaks_df.iloc[main_ions_list].copy().reset_index(drop=True)
    subset_time = time.time() - subset_start
    if show_progress:
        print(f"Created main ions subset in {subset_time:.2f}s")
    
    # Time candidate collection
    candidate_start = time.time()
    peaks_sorted, candidate_pairs = collect_candidate_peak_pairs(
        main_peaks_df, config.ACTIVE_ADDUCT_DIFFS, mz_tolerance=config.MZ_TOLERANCE, rt_tolerance=config.RT_WINDOW
    )
    candidate_time = time.time() - candidate_start

    
    if show_progress:
        print(f"Found {len(candidate_pairs)} adduct candidate pairs in {candidate_time:.2f}s")
    
    if not candidate_pairs:
        return edges
    
    # Time correlation computation
    correlation_start = time.time()
    correlation_results = compute_profile_correlations(
        candidate_pairs, signals_df, min_corr=config.ADDUCT_CORR_THRESHOLD
    )
    correlation_time = time.time() - correlation_start


    
    if show_progress:
        print(f"Found {len(correlation_results)} adduct pairs above correlation threshold in {correlation_time:.2f}s")
    
    # Convert to edge format
    for roi_i, roi_k, delta, correlation in _table_rows(correlation_results, ['roi_i', 'roi_k', 'delta', 'correlation']):
        # Convert ROI values back to original DataFrame indices
        idx_i = roi_to_idx.get(roi_i, -1)
        idx_k = roi_to_idx.get(roi_k, -1)
        
        if idx_i != -1 and idx_k != -1:
            edges.append((idx_i, idx_k, delta, correlation))
    
    print(f"[TIMER] detect_adduct_patterns total took {time.time() - step_start:.4f}s")
    return edges


def detect_dimer_patterns(peaks_df, roi_to_idx, main_ions, signals_df, show_progress=True, ionization_mode='positive'):
    import time
    step_start = time.time()
    """
    Detect dimer patterns using multiplication-based search.
    For dimers, the relationship is: m_dimer = 2 * m_monomer - correction
    """
    edges = []
    main_ions_list = list(main_ions)
    
    if len(main_ions_list) < 2:
        return edges
        
    # Create subset DataFrame for main ions only
    main_peaks_df = peaks_df.iloc[main_ions_list].copy().reset_index(drop=True)
    
    # Sort peaks by m/z for efficient binary search
    sorted_indices = main_peaks_df['mzmed'].argsort()
    sorted_peaks = main_peaks_df.iloc[sorted_indices].copy()
    sorted_mz = sorted_peaks['mzmed'].values
    sorted_roi = sorted_peaks['roi'].values
    
    sorted_rt = sorted_peaks['rtmeanw'].values
    monomers = np.arange(len(sorted_mz))
    adduct_h = config.ADDUCT_DIFFS.get('adduct_H', 1.00782)
    tol = config.MZ_TOLERANCE

    candidate_pairs = []  # List of tuples (roi_i, roi_k, delta_name)

    # For each dimer type, find monomer-dimer pairs
    for dimer_name, correction in config.ACTIVE_DIMER_DIFFS.items():
        if show_progress:
            print(f"  Searching for {dimer_name} (correction: {correction:.5f})")

        # Expected dimer m/z of every monomer: m_dimer = 2 * m_monomer + correction
        if ionization_mode == 'positive':
            expected = 2 * (sorted_mz - adduct_h) + correction
        elif ionization_mode == 'negative':
            expected = 2 * (sorted_mz + adduct_h) + correction
        else:
            raise ValueError(f"Unknown ionization mode: {ionization_mode}")

        # Peaks in the m/z range of each monomer (binary search), in the order of the monomers then of the m/z
        left = np.searchsorted(sorted_mz, expected - tol, side='left')
        right = np.searchsorted(sorted_mz, expected + tol, side='right')
        count = right - left
        ii = np.repeat(monomers, count)
        jj = np.repeat(left, count) + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))

        # Skip self-matches, then RT window and m/z tolerance
        keep = (sorted_roi[ii] != sorted_roi[jj]) & ~(np.abs(sorted_rt[jj] - sorted_rt[ii]) > config.RT_WINDOW) \
            & (np.abs(sorted_mz[jj] - expected[ii]) <= tol)
        # Add as tuple (roi_i, roi_k, delta_name) - format expected by compute_profile_correlations
        candidate_pairs.extend((a, b, dimer_name) for a, b in zip(sorted_roi[ii[keep]], sorted_roi[jj[keep]]))
    
    if show_progress:
        print(f"Found {len(candidate_pairs)} dimer candidate pairs")
    
    if not candidate_pairs:
        return edges
        
    # Compute correlations using optimized approach
    correlation_results = compute_profile_correlations(
        candidate_pairs, signals_df, min_corr=config.DIMER_CORR_THRESHOLD
    )

    print(f"Found {len(correlation_results)} dimer pairs above correlation threshold {config.DIMER_CORR_THRESHOLD}")
    
    # Convert to edge format
    for roi_i, roi_k, delta, correlation in _table_rows(correlation_results, ['roi_i', 'roi_k', 'delta', 'correlation']):
        # Convert ROI values back to original DataFrame indices  
        idx_i = roi_to_idx.get(roi_i, -1)
        idx_k = roi_to_idx.get(roi_k, -1)
        
        if idx_i != -1 and idx_k != -1:
            edges.append((idx_i, idx_k, delta, correlation))
    
    print(f"[TIMER] detect_dimer_patterns total took {time.time() - step_start:.4f}s")
    return edges


# The higher the value the lower chance the edge will be cut. np.inf protects isotope links
EDGE_CUT_PRIORITY = {'isotope': np.inf, 'adduct': 3.0, 'dimer': 2.0, 'mod': 1.0, 'loss': 1.0}


def edge_cut_score(edge_type, corr):
    """Score used to rank edges from the most to the least expendable."""
    return EDGE_CUT_PRIORITY.get(edge_type.split('_')[0], 1.0) * corr


def merge_under_size(edges, n_nodes, K):
    """
    Size-constrained single linkage: edges are taken from the strongest to the weakest and two groups are joined only
    if the result holds at most K peaks (isotope edges always join). Every edge whose two ends end up in the same group
    is kept, so oversized components are split along their weakest bridges instead of losing their peripheral peaks.
    """
    parent = list(range(n_nodes))
    size = [1] * n_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, edge_type, corr in sorted(edges, key=lambda e: -edge_cut_score(e[2], e[3])):
        a, b = find(i), find(j)
        if a == b or (size[a] + size[b] > K and np.isfinite(edge_cut_score(edge_type, corr))):
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    return [e for e in edges if find(e[0]) == find(e[1])]


def limit_component_size(edges, peaks_df, K=200, method='cut'):
    """
    Break up connected components larger than K.

    Isotope edges are never cut, while the remaining edges get a score combining the chemical priority of their type
    with the profile correlation. With method='cut' the lowest scoring edges inside oversized components are dropped
    in batches until every component holds at most K peaks; with method='merge' components are rebuilt from the
    strongest edges under the size limit (see merge_under_size).

    Args:
        edges: List of (idx_i, idx_j, edge_type_name, correlation) tuples
        peaks_df: DataFrame with peak information (kept for signature compatibility)
        K: Maximum component size
        method: 'cut' or 'merge'

    Returns:
        limited_edges: List of edges after cutting
        split_metadata: Dict with cutting statistics
    """
    if not edges or K is None:
        return edges, {'components_processed': 0, 'components_split': 0, 'max_component_size': 0}

    if method == 'merge':
        limited_edges = merge_under_size(edges, len(peaks_df), K)
        print(f"Cut {len(edges) - len(limited_edges)} edges")
        return limited_edges, {'cut_edges': len(edges) - len(limited_edges)}
    if method != 'cut':
        raise ValueError(f"Unknown component split method: {method}")

    G = nx.Graph()
    G.add_nodes_from(range(len(peaks_df)))
    for i, j, edge_type, corr in edges:
        G.add_edge(i, j, edge_type=edge_type, corr=corr, score=edge_cut_score(edge_type, corr))

    n_before = G.number_of_edges()
    components = list(nx.connected_components(G))
    n_oversized = sum(1 for comp in components if len(comp) > K)
    print(f"Found {len(components)} components, {n_oversized} above K={K}")

    # Only the oversized components can need a cut
    H = G.subgraph(set().union(*(comp for comp in components if len(comp) > K))).copy()
    while True:
        oversized_comps = [comp for comp in nx.connected_components(H) if len(comp) > K]
        if not oversized_comps:
            break
        oversized = set().union(*oversized_comps)
        # only edges sitting inside an oversized component are candidates
        cuttable = sorted(
            (d['score'], u, v) for u, v, d in G.edges(sorted(oversized), data=True)
            if np.isfinite(d['score'])
        )
        if not cuttable:
            print(f"WARNING: {len(oversized)} nodes remain in oversized components with no cuttable edge left")
            break
        # one cut per K peaks in excess
        batch = max(1, sum(len(comp) // K for comp in oversized_comps))
        cut = [(u, v) for _, u, v in cuttable[:batch]]
        G.remove_edges_from(cut)
        H.remove_edges_from(cut)

    limited_edges = [(u, v, d['edge_type'], d['corr']) for u, v, d in G.edges(data=True)]
    sizes = [len(comp) for comp in nx.connected_components(G)]
    print(f"Cut {n_before - len(limited_edges)} edges, max component size now {max(sizes) if sizes else 0}")

    return limited_edges, {
        'components_processed': len(components),
        'components_split': n_oversized,
        'max_component_size': max(sizes) if sizes else 0,
        'cut_edges': n_before - len(limited_edges),
    }


def _deduplicate_undirected_edges_numpy(edge_indices, similarities, edge_type_indices):
    """Fast deduplication on undirected pairs with identical attrs.

    Treats (u, v) and (v, u) as duplicates when edge_type and similarity bits match.
    Keeps the first occurrence in original order.
    """
    n_edges = similarities.shape[0]
    if n_edges <= 1:
        return edge_indices, similarities, edge_type_indices, 0

    src = edge_indices[0]
    dst = edge_indices[1]
    undirected_u = np.minimum(src, dst)
    undirected_v = np.maximum(src, dst)
    sim_bits = similarities.view(np.uint32)
    original_idx = np.arange(n_edges, dtype=np.int64)

    # Sort by key in C-speed; include original index as tie-breaker so we keep first-seen edges
    order = np.lexsort((original_idx, sim_bits, edge_type_indices, undirected_v, undirected_u))

    u_sorted = undirected_u[order]
    v_sorted = undirected_v[order]
    type_sorted = edge_type_indices[order]
    sim_bits_sorted = sim_bits[order]

    is_dup = (
        (u_sorted[1:] == u_sorted[:-1])
        & (v_sorted[1:] == v_sorted[:-1])
        & (type_sorted[1:] == type_sorted[:-1])
        & (sim_bits_sorted[1:] == sim_bits_sorted[:-1])
    )

    keep_sorted = np.ones(n_edges, dtype=bool)
    keep_sorted[1:] = ~is_dup

    keep_idx = order[keep_sorted]
    keep_idx.sort()  # Preserve first-seen order in the original edge list

    removed = n_edges - keep_idx.shape[0]
    return (
        edge_indices[:, keep_idx],
        similarities[keep_idx],
        edge_type_indices[keep_idx],
        removed,
    )


def convert_to_pytorch_format(edges, peaks_df, raw_signals, signals_df, debug_hanging=False, file_id=None,
                              show_progress=True, eic_extraction_mode='uniform'):
    """
    Convert edges to PyTorch format and create EIC profiles for node features.

    EIC extraction modes:
    - uniform: subsample to EIC_LEN with apex preservation
    - center_rt_window: keep original points around rtmeanw center (windowed to EIC_LEN max)
    """
    import torch
    
    # Get ROI values to use as node IDs
    roi_values = peaks_df['roi'].values
    n_nodes = len(peaks_df)
    
    # Create mapping from array index to ROI value
    idx_to_roi = {idx: roi for idx, roi in enumerate(roi_values)}
    
    if not edges:
        # Empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2))
    else:
        n_edges = len(edges)
        
        # Pre-allocate tensors for better performance
        edge_indices = np.zeros((2, n_edges), dtype=np.int64)
        similarities = np.zeros(n_edges, dtype=np.float32)
        edge_type_indices = np.zeros(n_edges, dtype=np.int16)
        
        # Create edge type encoding (do this once)
        edge_type_encoding = {}
        
        # Isotopes (0-10)
        for i, iso_name in enumerate(config.MASS_DIFFS.keys()):
            edge_type_encoding[iso_name] = i
        
        # Adducts (11-30) 
        for i, adduct_name in enumerate(config.ADDUCT_DIFFS.keys()):
            edge_type_encoding[adduct_name] = 11 + i
        
        # Dimers (31-33)
        for i, dimer_name in enumerate(config.DIMER_DIFFS.keys()):
            edge_type_encoding[dimer_name] = 31 + i
    
        # Fallback for unknown types
        edge_type_encoding['unknown'] = 35

        print(f"Edge type encoding: {edge_type_encoding}")
        
        # Fill arrays efficiently, converting array indices to ROI values.
        # Every edge goes from the lighter to the heavier ion, as in the graphs used to train the published model.
        mz_values = peaks_df['mzmed'].values
        for idx, (i, j, edge_type, similarity) in enumerate(edges):
            if mz_values[i] > mz_values[j]:
                i, j = j, i
            edge_indices[0, idx] = idx_to_roi[i]  # Convert to ROI value
            edge_indices[1, idx] = idx_to_roi[j]  # Convert to ROI value
            similarities[idx] = similarity
            edge_type_indices[idx] = edge_type_encoding.get(edge_type, 35)
        
        # Remove duplicate undirected edges with identical attributes:
        edge_indices, similarities, edge_type_indices, n_removed = _deduplicate_undirected_edges_numpy(
            edge_indices, similarities, edge_type_indices
        )
        if n_removed > 0:
            print(f"Removed {n_removed} duplicate undirected edges before tensor conversion")

        # Convert to tensors
        edge_index = torch.from_numpy(edge_indices)
        edge_attr = torch.stack([
            torch.from_numpy(similarities),
            torch.from_numpy(edge_type_indices.astype(np.float32))
        ], dim=1)
        
    
    valid_eic_modes = {'uniform', 'center_rt_window'}
    if eic_extraction_mode not in valid_eic_modes:
        raise ValueError(f"Unknown eic_extraction_mode='{eic_extraction_mode}'. Expected one of: {sorted(valid_eic_modes)}")

    if show_progress:
        print(f"Creating EIC profiles with {config.EIC_LEN} RT + {config.EIC_LEN} intensity points...")
        print("Using original scan RT values (no interpolation) - preserving RT-intensity correspondence")
        print(f"EIC extraction mode: {eic_extraction_mode}")
    
    # node feature structure: 8 basic + 32 RT + 32 intensity = 72 total
    available_cols = set(peaks_df.columns)
    
    # 8 meta core peak characteristics
    basic_feature_cols = [
        ('mzmed', 'mz', 0.0),           # m/z median
        ('rtmeanw', 'rt', 0.0),         # RT weighted mean  
        ('into', 'intensity', 0.0),     # integrated intensity
        ('rtmin', 'rtmin', 0.0),        # RT minimum
        ('rtmax', 'rtmax', 0.0),        # RT maximum
        ('imax', 'imax', 0.0),          # maximum intensity
        ('mzmeanw', 'mzmeanw', 0.0),              # m/z weighted mean
        ('sc_length', 'sc_length', 0.0), # scan length
    ]
    
    # Ensure exactly 8 basic features 
    n_basic_features = 8
    n_rt_features = config.EIC_LEN  # 32 RT values
    n_intensity_features = config.EIC_LEN  # 32 intensity values
    total_features = n_basic_features + n_rt_features + n_intensity_features  # 8 + 32 + 32 = 72
    rt_start_idx = n_basic_features
    intensity_start_idx = n_basic_features + n_rt_features
    
    
    # Pre-allocate node features array
    node_features = np.zeros((n_nodes, total_features), dtype=np.float32)
    roi_values_array = peaks_df['roi'].values
    rtmeanw_values = peaks_df['rtmeanw'].values
    
    # Fill basic features (exactly 8 columns)
    for feature_idx in range(n_basic_features):
        if feature_idx < len(basic_feature_cols):
            col_name, display_name, default_val = basic_feature_cols[feature_idx]
            if col_name in available_cols:
                node_features[:, feature_idx] = peaks_df[col_name].fillna(default_val).values
            else:
                node_features[:, feature_idx] = default_val  # Use default if column missing
        else:
            node_features[:, feature_idx] = 0.0  # Pad with zeros if needed
    
    # Create EIC profiles with subsampling/padding 
    for idx, roi in enumerate(roi_values_array):
        rt_profile = np.zeros(config.EIC_LEN)
        intensity_profile = np.zeros(config.EIC_LEN)
        
        if roi in raw_signals:
            rt_points, intensities = raw_signals[roi]
            
            if len(rt_points) > 0 and len(intensities) > 0:
                # Convert to numpy arrays
                rt_array = np.array(rt_points, dtype=np.float64)
                intensity_array = np.array(intensities, dtype=np.float64)
                
                # Sort by RT to ensure proper ordering
                sort_indices = np.argsort(rt_array)
                rt_sorted = rt_array[sort_indices]
                intensity_sorted = intensity_array[sort_indices]
                
                n_points = len(rt_sorted)
                
                if n_points <= config.EIC_LEN:
                    # Pad with zeros if fewer points than EIC_LEN
                    rt_profile[:n_points] = rt_sorted
                    intensity_profile[:n_points] = intensity_sorted
                else:
                    if eic_extraction_mode == 'center_rt_window':
                        # Keep original points around rtmeanw center (no subsampling interpolation)
                        center_rt = rtmeanw_values[idx]
                        center_idx = int(np.argmin(np.abs(rt_sorted - center_rt)))

                        if config.EIC_LEN == 1:
                            selected_idx = np.array([center_idx], dtype=int)
                        else:
                            half_window = config.EIC_LEN // 2
                            start_idx = center_idx - half_window
                            start_idx = max(0, min(start_idx, n_points - config.EIC_LEN))
                            end_idx = start_idx + config.EIC_LEN
                            selected_idx = np.arange(start_idx, end_idx, dtype=int)

                        rt_profile = rt_sorted[selected_idx]
                        intensity_profile = intensity_sorted[selected_idx]
                    else:
                        # subsampling: always preserve apex + uniform sampling
                        apex_idx = np.argmax(intensity_sorted)  # Find apex index

                        # generate uniform subsample indices
                        uniform_indices = np.linspace(0, n_points - 1, config.EIC_LEN, dtype=int)

                        # ensure apex is included by replacing closest uniform point
                        closest_uniform_idx = np.argmin(np.abs(uniform_indices - apex_idx))
                        uniform_indices[closest_uniform_idx] = apex_idx

                        # sort indices to maintain RT order
                        uniform_indices = np.sort(uniform_indices)

                        rt_profile = rt_sorted[uniform_indices]
                        intensity_profile = intensity_sorted[uniform_indices]
                

        # Fill feature arrays with EIC profiles
        node_features[idx, rt_start_idx:rt_start_idx + n_rt_features] = rt_profile
        node_features[idx, intensity_start_idx:intensity_start_idx + n_intensity_features] = intensity_profile
    
    node_features = torch.from_numpy(node_features)
    
    # Print feature summary
    print(f"Node feature dimensions: {node_features.shape}")
    basic_names = [display_name for col_name, display_name, _ in basic_feature_cols[:n_basic_features]]
    rt_names = [f'rt_{i:02d}' for i in range(n_rt_features)]
    intensity_names = [f'intensity_{i:02d}' for i in range(n_intensity_features)]
    
    all_feature_names = basic_names + rt_names + intensity_names
    print(f"Feature structure: {basic_names} + {len(rt_names)} RT + {len(intensity_names)} intensity")
    print(f"Total features: {len(all_feature_names)} (expected: {total_features})")
    
    # Verify EIC profiles are non-zero
    rt_nonzero = torch.sum(node_features[:, rt_start_idx:rt_start_idx + n_rt_features] != 0).item()
    intensity_nonzero = torch.sum(node_features[:, intensity_start_idx:intensity_start_idx + n_intensity_features] != 0).item()
    print(f"Non-zero EIC values: RT={rt_nonzero}/{n_nodes * n_rt_features}, Intensity={intensity_nonzero}/{n_nodes * n_intensity_features}")
    
    # Also return the ROI values for mapping
    return edge_index, edge_attr, node_features, roi_values

