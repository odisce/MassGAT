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

    # Detect regular mass differences
    for delta_name, delta in mz_deltas.items():
        i = 0
        j = 0
        while i < n - 1:
            #j = i + 1
            while j < n and mz[j] - mz[i] < delta - mz_tolerance:
                j += 1
            
            k = j

            while k < n and mz[k] - mz[i] <= delta + mz_tolerance:
                if abs(rt[k] - rt[i]) <= rt_tolerance:
                    # Store both (i,k) and (k,i) to cover bidirectional matching
                    candidate_pairs.append((roi[i], roi[k], delta_name))
                    candidate_pairs.append((roi[k], roi[i], delta_name))

                    # Update the DataFrame with the matched ROIs
                    peaks_df_sorted.at[k, f"match_{delta_name}"] = roi[i]
                    peaks_df_sorted.at[i, f"match_{delta_name}"] = roi[k]
                k += 1
            i += 1

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
        profiles: Dict mapping roi -> interpolated intensity profile
    """
    if not roi_list:
        return np.array([]), {}
    
    # Filter signals to only include ROIs we need
    filter_start = time.time()
    roi_set = set(roi_list)
    signals_filtered = signals_df[signals_df['roi'].isin(roi_set)]
    filter_time = time.time() - filter_start
    
    if signals_filtered.empty:
        return np.array([]), {roi: np.array([]) for roi in roi_list}
    
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

    # Pre-group signals by ROI (vectorized approach)
    pregroup_start = time.time()
    roi_groups = signals_filtered.groupby('roi')
    pregroup_time = time.time() - pregroup_start
    
    print(f"  Preprocessing: filter={filter_time:.2f}s, sort={sort_time:.2f}s, groupby={groupby_time:.2f}s, grid={grid_time:.2f}s, pregroup={pregroup_time:.2f}s")

    # Interpolate on common RT grid
    interp_start = time.time()
    profiles = {}
    interp_count = 0
    for roi in roi_list:
        if roi in roi_groups.groups:
            group = roi_groups.get_group(roi)
            rt_vals = group['rt'].values
            intensity_vals = group['i'].values
            profiles[roi] = np.interp(rt_grid, rt_vals, intensity_vals, left=0, right=0)
            interp_count += 1
        else:
            profiles[roi] = np.zeros_like(rt_grid)
    
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
        profile = profiles[roi]
        norms[roi] = np.linalg.norm(profile) if profile is not None else 0.0

    # Compute correlations
    results = []
    failed_candidates = []
    for roi_i, roi_j, delta in candidate_pairs:
        norm_i = norms.get(roi_i, 0.0)
        norm_j = norms.get(roi_j, 0.0)
        if norm_i == 0 or norm_j == 0:
            continue
        # Correlation computation
        profile_i = profiles[roi_i]
        profile_j = profiles[roi_j]
        corr = np.dot(profile_i, profile_j) / (norm_i * norm_j)
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
        print(f"Mode d'ionisation détecté: {ionization_mode}")

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
    
    if show_progress:
        print("Step 4: Adding profile correlations between main ions...")
    profile_start = time.time()
    
    include_profile_correlations = False
    if include_profile_correlations:
        # Collect main ions from isotope groups
        main_ion_rois = [group['main_roi'] for group in isotope_groups]
        if show_progress:
            print(f"Working with {len(main_ion_rois)} main ions from isotope groups")
        
        if len(main_ion_rois) < 2:
            profile_edges = []
        else:
            main_ion_rois_set = set(main_ion_rois)
            
            # Create a set of existing connections to avoid duplicates
            existing_pairs = set()
            for edge in edges:
                roi1, roi2 = rois[edge[0]], rois[edge[1]]
                if roi1 in main_ion_rois_set and roi2 in main_ion_rois_set:
                    existing_pairs.add((min(roi1, roi2), max(roi1, roi2)))
            
            print(f"Found {len(existing_pairs)} existing connections between main ions")
            log_progress(f"Found {len(existing_pairs)} existing connections between main ions")
            
            # Create DataFrame with just main ions, sorted by RT for efficient neighbor search
            main_ions_df = peaks_df[peaks_df['roi'].isin(main_ion_rois_set)].copy()
            main_ions_df = main_ions_df.sort_values('rtmeanw').reset_index(drop=True)
            
            # Optimized RT-proximity search using sorted data
            rt_values = main_ions_df['rtmeanw'].values
            roi_values_main = main_ions_df['roi'].values
            n = len(main_ions_df)
            
            candidate_pairs = []
            for i in range(n):
                # Only search forward in sorted RT list until beyond RT_WINDOW
                for j in range(i + 1, n):
                    rt_diff = rt_values[j] - rt_values[i]
                    if rt_diff > config.RT_WINDOW:
                        break  # Stop searching, all remaining peaks too far in RT
                    
                    roi_pair = (min(roi_values_main[i], roi_values_main[j]), 
                               max(roi_values_main[i], roi_values_main[j]))
                    
                    # Skip if already connected
                    if roi_pair in existing_pairs:
                        continue
                    
                    candidate_pairs.append((roi_values_main[i], roi_values_main[j]))
            
            print(f"Found {len(candidate_pairs)} new main ion pairs within RT window (after filtering existing connections and intensity ratio)")
            log_progress(f"Found {len(candidate_pairs)} new main ion pairs within RT window")

            if candidate_pairs:
                correlation_results = compute_profile_correlations(
                    candidate_pairs, signals_df, min_corr=config.PROFILE_CORR_THRESHOLD
                )
                
                # Convert to edge format
                profile_edges = []
                for _, row in correlation_results.iterrows():
                    roi_i, roi_k = row['roi_i'], row['roi_k']
                    idx_i = roi_to_idx.get(roi_i, -1)
                    idx_k = roi_to_idx.get(roi_k, -1)
                    if idx_i != -1 and idx_k != -1:
                        profile_edges.append((idx_i, idx_k, 'profile_correlation', row['correlation']))
            else:
                profile_edges = []
    else:
        print("Profile correlations disabled")
        profile_edges = []
                
    
    edges.extend(profile_edges)
    profile_time = time.time() - profile_start
    print(f"[TIMER] profile correlation step took {profile_time:.4f}s")
    log_progress(f"[TIMER] profile correlation step took {profile_time:.4f}s")
    
    if show_progress:
        print(f"Total edges created: {len(edges)}")
    
    # Limit component sizes (optional)
    if limit_component_size_value is not None:
        print("Step 5: Limiting component sizes...")
        log_progress("Step 5: Limiting component sizes...")
        limit_start = time.time()
        edges, split_metadata = limit_component_size(edges, peaks_df, K=limit_component_size_value)
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
    print(f"Isotope: {iso_time:.2f}s | Adduct: {adduct_time:.2f}s | Dimer: {dimer_time:.2f}s | Profile: {profile_time:.2f}s")
    print(f"Optimized correlation computation enabled")
    

    return edge_index, edge_attr, node_features, roi_values


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
        peaks_df, config.MASS_DIFFS, mz_tolerance=config.MZ_TOLERANCE, rt_tolerance=config.RT_WINDOW
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
    
    roi_to_peak = {row['roi']: row for _, row in peaks_df.iterrows()}
    
    for _, row in correlation_results.iterrows():
        # Enforce isotope intensity hierarchy: lower m/z (main) must have higher intensity
        roi_i, roi_k = row['roi_i'], row['roi_k']
        delta_name = row['delta']
        
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

        # Extra 37Cl filter (34S and 37Cl are close in m/z, so the intensity ratio can help distinguish them)
        if delta_name == 'isotope_37Cl':
            if (iso_into / main_into) < min_37cl_ratio:
                # replace with 34S isotope
                delta_name = 'isotope_34S'

        validated_pairs.append({
            'main_roi': main_roi,
            'iso_roi': iso_roi,
            'iso_name': delta_name,
            'correlation': row['correlation']
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
        main_peaks_df, config.ADDUCT_DIFFS, mz_tolerance=config.MZ_TOLERANCE, rt_tolerance=config.RT_WINDOW
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
    for _, row in correlation_results.iterrows():
        roi_i, roi_k = row['roi_i'], row['roi_k']
        
        # Convert ROI values back to original DataFrame indices
        idx_i = roi_to_idx.get(roi_i, -1)
        idx_k = roi_to_idx.get(roi_k, -1)
        
        if idx_i != -1 and idx_k != -1:
            edges.append((idx_i, idx_k, row['delta'], row['correlation']))
    
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
    
    candidate_pairs = []  # List of tuples (roi_i, roi_k, delta_name)

    # For each dimer type, find monomer-dimer pairs
    for dimer_name, correction in config.DIMER_DIFFS.items():
        if show_progress:
            print(f"  Searching for {dimer_name} (correction: {correction:.5f})")
        
        # For each potential monomer peak
        for i, monomer_row in sorted_peaks.iterrows():
            monomer_mz = monomer_row['mzmed']
            monomer_rt = monomer_row['rtmeanw']
            monomer_roi = monomer_row['roi']

            adduct_h = config.ADDUCT_DIFFS.get('adduct_H', 1.00782)
            
            
            # Calculate expected dimer m/z: m_dimer = 2 * m_monomer + correction
            if ionization_mode == 'positive':
                expected_dimer_mz = 2 * (monomer_mz - adduct_h) + correction
            elif ionization_mode == 'negative':
                expected_dimer_mz = 2 * (monomer_mz + adduct_h) + correction

            
            # Binary search for peaks near expected dimer m/z
            import bisect
            left_idx = bisect.bisect_left(sorted_mz, expected_dimer_mz - config.MZ_TOLERANCE)
            right_idx = bisect.bisect_right(sorted_mz, expected_dimer_mz + config.MZ_TOLERANCE)
            
            # Check all peaks in the m/z range
            for j in range(left_idx, right_idx):
                if j >= len(sorted_peaks):
                    break
                    
                dimer_mz = sorted_mz[j]
                dimer_roi = sorted_roi[j]
                dimer_rt = sorted_peaks.iloc[j]['rtmeanw']
                
                # Skip self-matches
                if monomer_roi == dimer_roi:
                    continue
                
                # Check RT window constraint
                rt_diff = abs(dimer_rt - monomer_rt)
                if rt_diff > config.RT_WINDOW:
                    continue
                
                # Check m/z tolerance
                mz_diff = abs(dimer_mz - expected_dimer_mz)
                if mz_diff <= config.MZ_TOLERANCE:
                    # Add as tuple (roi_i, roi_k, delta_name) - format expected by compute_profile_correlations
                    candidate_pairs.append((monomer_roi, dimer_roi, dimer_name))
    
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
    for _, row in correlation_results.iterrows():
        roi_i, roi_k = row['roi_i'], row['roi_k']
        
        # Convert ROI values back to original DataFrame indices  
        idx_i = roi_to_idx.get(roi_i, -1)
        idx_k = roi_to_idx.get(roi_k, -1)
        
        if idx_i != -1 and idx_k != -1:
            edges.append((idx_i, idx_k, row['delta'], row['correlation']))
    
    print(f"[TIMER] detect_dimer_patterns total took {time.time() - step_start:.4f}s")
    return edges


def limit_component_size(edges, peaks_df, K=100):
    """
    Split large connected components to ensure all components ≤ K nodes.
    Uses isotope contraction + MST-based cutting with chemical hierarchy preservation.
    
    Args:
        edges: List of (idx_i, idx_j, edge_type_name, correlation) tuples
        peaks_df: DataFrame with peak information (for RT, m/z data)
        K: Maximum component size (default 50)
    
    Returns:
        limited_edges: List of edges after component splitting
        split_metadata: Dict with splitting statistics
    """
    
    if not edges:
        print("No edges to process")
        return edges, {'components_processed': 0, 'components_split': 0}
    
    # Build NetworkX graph from edges
    G = nx.Graph()
    
    # Add all nodes from peaks_df
    for idx in range(len(peaks_df)):
        G.add_node(idx)
    
    # Edge type priorities (higher = more important, never cut)
    edge_type_priority = {
        # Isotopes - highest priority, never split
        'isotope_13C': 1000, 'isotope_15N': 1000, 'isotope_18O': 1000, 
        'isotope_34S': 1000, 'isotope_37Cl': 1000,
        # Dimers - high priority 
        'dimer_2M+H': 500, 'dimer_2M+Na': 500, 'dimer_2M+NH4': 500,
        # Adducts - medium priority
        'adduct_Na-H': 100, 'adduct_H2O': 100, 'adduct_NH3': 100, 'adduct_HCOOH': 100,
        'adduct_CO': 100, 'adduct_CO2': 100, 'adduct_K-Na': 100, 'adduct_2H': 100,
        'adduct_O': 100, 'adduct_H': 100,
        # Profile correlations - lowest priority (not used anymore)
        'profile_correlation': 10
    }
    
    # Add edges with attributes
    for edge_id, (i, j, edge_type, corr) in enumerate(edges):
        if i < len(peaks_df) and j < len(peaks_df):
            # Get peak properties for better scoring
            peak_i = peaks_df.iloc[i]
            peak_j = peaks_df.iloc[j]
            
            mz_i, mz_j = peak_i['mzmed'], peak_j['mzmed']
            rt_i, rt_j = peak_i['rtmeanw'], peak_j['rtmeanw']
            intensity_i, intensity_j = peak_i['into'], peak_j['into']
            
            # Intensity ratio factor (penalize very different intensities)
            intensity_ratio = max(intensity_i, intensity_j) / max(min(intensity_i, intensity_j), 1e-6)
            intensity_factor = 1.0 / (1.0 + max(0, intensity_ratio - 2.0))  # Penalty if ratio > 2
            
            # RT proximity factor (already captured in correlation but add direct penalty)
            rt_diff = abs(rt_i - rt_j)
            rt_factor = max(0.1, 1.0 - rt_diff / 30.0)  # Penalty for RT > 30s apart
            
            # Edge type priority (much more important than small differences)
            type_priority = edge_type_priority.get(edge_type, 50)
            
            # Base score: type_priority * correlation^2 
            base_score = type_priority * (corr ** 2)
            
            # Apply factors
            score = base_score * intensity_factor * rt_factor
            
            # Store additional debug info
            G.add_edge(i, j, 
                      edge_type=edge_type, 
                      corr=corr, 
                      mz_diff=abs(mz_i - mz_j),
                      rt_diff=rt_diff,
                      intensity_ratio=intensity_ratio,
                      intensity_factor=intensity_factor,
                      rt_factor=rt_factor,
                      base_score=base_score,
                      score=score,
                      edge_id=edge_id)
    
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Find connected components
    components = list(nx.connected_components(G))
    large_components = [comp for comp in components if len(comp) > K]
    
    print(f"Found {len(components)} components, {len(large_components)} need splitting (size > {K})")
    
    # Quick exit for small graphs
    if not large_components:
        return edges, {
            'components_processed': len(components),
            'components_split': 0,
            'max_component_size': max(len(comp) for comp in components) if components else 0
        }
    
    # Performance optimization: Skip component limiting for very large components (>500 nodes)
    # Use simple random splitting instead
    huge_components = [comp for comp in large_components if len(comp) > 500]
    manageable_components = [comp for comp in large_components if len(comp) <= 500]
    
    if huge_components:
        print(f"Found {len(huge_components)} huge components (>500 nodes), using fast random splitting")
    
    # Process each large component
    new_edges = []
    split_metadata = {
        'components_processed': len(components),
        'components_split': len(large_components),
        'original_sizes': [len(comp) for comp in large_components],
        'final_sizes': [],
        'cut_edges': []
    }
    
    # Keep edges from small components unchanged
    small_components = [comp for comp in components if len(comp) <= K]
    for comp in small_components:
        subgraph = G.subgraph(comp)
        for u, v, data in subgraph.edges(data=True):
            new_edges.append((u, v, data['edge_type'], data['corr']))
    
    # Fast splitting for huge components (random partitioning)
    for comp_idx, comp in enumerate(huge_components):
        comp_list = list(comp)
        print(f"Fast splitting huge component {comp_idx + 1}/{len(huge_components)} (size: {len(comp)})")
        
        # Calculate optimal partition size (aim for K*0.8 to avoid tiny fragments)
        target_size = max(K // 2, int(K * 0.6))  # Aim for 30-40 nodes per partition
        num_partitions = max(1, len(comp_list) // target_size)
        
        for i in range(num_partitions):
            start_idx = i * len(comp_list) // num_partitions
            end_idx = (i + 1) * len(comp_list) // num_partitions
            partition = comp_list[start_idx:end_idx]
            
            if len(partition) > K:  # Still too big, split further
                sub_partitions = [partition[j:j+K] for j in range(0, len(partition), K)]
                for sub_partition in sub_partitions:
                    subgraph = G.subgraph(sub_partition)
                    for u, v, data in subgraph.edges(data=True):
                        new_edges.append((u, v, data['edge_type'], data['corr']))
                    split_metadata['final_sizes'].append(len(sub_partition))
            else:
                subgraph = G.subgraph(partition)
                for u, v, data in subgraph.edges(data=True):
                    new_edges.append((u, v, data['edge_type'], data['corr']))
                split_metadata['final_sizes'].append(len(partition))
    
    # Sophisticated splitting for manageable components
    for comp_idx, comp in enumerate(manageable_components):
        print(f"Processing manageable component {comp_idx + 1}/{len(manageable_components)} (size: {len(comp)})")
        
        subgraph = G.subgraph(comp).copy()
        partitions = _split_component(subgraph, K)
        
        # Add edges within each partition
        for partition in partitions:
            partition_subgraph = subgraph.subgraph(partition)
            for u, v, data in partition_subgraph.edges(data=True):
                new_edges.append((u, v, data['edge_type'], data['corr']))
        
        partition_sizes = [len(p) for p in partitions]
        split_metadata['final_sizes'].extend(partition_sizes)
        print(f"Split into {len(partitions)} partitions: {partition_sizes}")
        
        # CRITICAL: Validate that all partitions are ≤ K
        oversized_partitions = [size for size in partition_sizes if size > K]
        if oversized_partitions:
            print(f"WARNING: Found oversized partitions after splitting: {oversized_partitions}")
            print(f"This should never happen! Partitions > {K}: {oversized_partitions}")
    
    # Final validation of all partition sizes
    all_final_sizes = split_metadata['final_sizes']
    max_final_size = max(all_final_sizes) if all_final_sizes else 0
    oversized_final = [size for size in all_final_sizes if size > K]
    
    print(f"Component splitting complete. Final edge count: {len(new_edges)}")
    print(f"Final validation - Max component size: {max_final_size}, Oversized components: {len(oversized_final)}")
    
    if oversized_final:
        print(f"ERROR: Component limiting failed! Found components larger than {K}: {oversized_final}")
        print("Applying emergency random splitting to oversized components...")
        
        # Emergency fallback: find and randomly split oversized components
        final_edges = []
        G_final = nx.Graph()
        G_final.add_edges_from([(u, v) for u, v, _, _ in new_edges])
        
        final_components = list(nx.connected_components(G_final))
        for comp in final_components:
            if len(comp) <= K:
                # Keep edges for small components
                subgraph = G_final.subgraph(comp)
                for u, v in subgraph.edges():
                    # Find original edge data
                    for orig_u, orig_v, edge_type, corr in new_edges:
                        if (orig_u == u and orig_v == v) or (orig_u == v and orig_v == u):
                            final_edges.append((orig_u, orig_v, edge_type, corr))
                            break
            else:
                # Emergency split: randomly partition oversized component
                comp_list = list(comp)
                num_partitions = (len(comp_list) + K - 1) // K  # Ceiling division
                for i in range(num_partitions):
                    start_idx = i * K
                    end_idx = min((i + 1) * K, len(comp_list))
                    partition = comp_list[start_idx:end_idx]
                    
                    # Add edges within this partition only
                    subgraph = G_final.subgraph(partition)
                    for u, v in subgraph.edges():
                        # Find original edge data
                        for orig_u, orig_v, edge_type, corr in new_edges:
                            if (orig_u == u and orig_v == v) or (orig_u == v and orig_v == u):
                                final_edges.append((orig_u, orig_v, edge_type, corr))
                                break
                
                print(f"Emergency split component of size {len(comp)} into {num_partitions} partitions")
        
        new_edges = final_edges
        print(f"After emergency splitting: {len(new_edges)} edges")
    
    return new_edges, split_metadata


def _split_component(G, K):
    """
    Split a single large component using isotope contraction + MST cutting.
    """
    # Step 1: Contract isotope families (never split these)
    contracted_G, supernode_map = _contract_isotope_families(G)
    
    if len(contracted_G) <= K:
        # After contraction, component is small enough
        return [set(G.nodes())]
    
    # Step 2: Build MST on contracted graph
    # Use negative scores for maximum spanning tree
    for u, v, data in contracted_G.edges(data=True):
        data['weight'] = -data['score']  # Negative for MST
    
    mst = nx.maximum_spanning_tree(contracted_G, weight='weight')
    
    # Step 3: Smart cutting to create balanced partitions
    # Target partition size: aim for 60-80% of K to avoid tiny fragments
    min_partition_size = max(K // 3, 10)  # Minimum 10 nodes or K/3
    target_partition_size = int(K * 0.7)   # Target 70% of K
    
    current_trees = [mst.copy()]
    
    while any(len(_expand_supernode_tree(tree, supernode_map)) > K for tree in current_trees):
        # Find tree with largest expanded size
        tree_sizes = [len(_expand_supernode_tree(tree, supernode_map)) for tree in current_trees]
        largest_tree_idx = max(range(len(tree_sizes)), key=lambda i: tree_sizes[i])
        largest_tree = current_trees[largest_tree_idx]
        
        if largest_tree.number_of_edges() == 0:
            break  # Can't split further
        
        # Smart edge selection: find edge that creates most balanced split
        best_edge = None
        best_balance_score = float('-inf')
        
        for edge in largest_tree.edges(data=True):
            u, v = edge[0], edge[1]
            # Temporarily remove edge and check resulting component sizes
            temp_tree = largest_tree.copy()
            temp_tree.remove_edge(u, v)
            
            components = list(nx.connected_components(temp_tree))
            if len(components) == 2:  # Should split into exactly 2 parts
                sizes = [len(_expand_supernode_tree(temp_tree.subgraph(comp), supernode_map)) 
                        for comp in components]
                
                # Calculate balance score (prefer more balanced splits)
                min_size, max_size = min(sizes), max(sizes)
                
                # Penalty for tiny components
                if min_size < min_partition_size:
                    balance_score = -1000  # Heavy penalty
                else:
                    # Reward balanced splits, with preference for target size
                    balance_score = min_size / max_size  # Higher is more balanced
                    
                    # Bonus if both parts are close to target size
                    if target_partition_size <= min_size <= K and target_partition_size <= max_size <= K:
                        balance_score += 0.5
                
                if balance_score > best_balance_score:
                    best_balance_score = balance_score
                    best_edge = edge
        
        # If no good balanced split found, fall back to weakest edge
        if best_edge is None:
            best_edge = min(largest_tree.edges(data=True),
                          key=lambda x: (x[2]['score'], x[2]['edge_id']))
        
        # Remove the selected edge
        largest_tree.remove_edge(best_edge[0], best_edge[1])
        
        # Split into connected components
        current_trees.pop(largest_tree_idx)
        new_components = list(nx.connected_components(largest_tree))
        for comp in new_components:
            current_trees.append(largest_tree.subgraph(comp).copy())
    
    # Step 4: Expand supernodes back to original nodes
    final_partitions = []
    for tree in current_trees:
        partition = _expand_supernode_tree(tree, supernode_map)
        final_partitions.append(partition)
    
    # Step 5: CRITICAL - Ensure no single nodes are isolated
    # (unless they were already isolated in the original graph)
    final_partitions = _ensure_no_isolated_singles(final_partitions, G, supernode_map)
    
    return final_partitions


def _contract_isotope_families(G):
    """Contract nodes connected by isotope edges into supernodes."""
    isotope_types = {'isotope_13C', 'isotope_15N', 'isotope_18O', 'isotope_34S', 'isotope_37Cl'}
    
    # Find isotope edges
    isotope_edges = [(u, v) for u, v, data in G.edges(data=True) 
                     if data['edge_type'] in isotope_types]
    
    # Find connected components of isotope edges
    isotope_graph = nx.Graph()
    isotope_graph.add_nodes_from(G.nodes())
    isotope_graph.add_edges_from(isotope_edges)
    
    isotope_families = list(nx.connected_components(isotope_graph))
    
    # Create contracted graph
    contracted_G = nx.Graph()
    supernode_map = {}  # supernode_id -> set of original nodes
    
    # Create supernodes for isotope families
    for family_idx, family in enumerate(isotope_families):
        if len(family) > 1:
            supernode_id = f"super_{family_idx}"
            contracted_G.add_node(supernode_id)
            supernode_map[supernode_id] = family
        else:
            # Single node families stay as is
            node = list(family)[0]
            contracted_G.add_node(node)
            supernode_map[node] = {node}
    
    # Add edges between supernodes
    for u, v, data in G.edges(data=True):
        # Find which supernodes u and v belong to
        u_supernode = None
        v_supernode = None
        
        for supernode, nodes in supernode_map.items():
            if u in nodes:
                u_supernode = supernode
            if v in nodes:
                v_supernode = supernode
        
        # Only add edge if different supernodes
        if u_supernode != v_supernode and not contracted_G.has_edge(u_supernode, v_supernode):
            contracted_G.add_edge(u_supernode, v_supernode, **data)
        elif u_supernode != v_supernode:
            # Keep highest scoring edge between supernodes
            existing_score = contracted_G[u_supernode][v_supernode]['score']
            if data['score'] > existing_score:
                contracted_G[u_supernode][v_supernode].update(data)
    
    return contracted_G, supernode_map


def _expand_supernode_tree(tree, supernode_map):
    """Expand a tree of supernodes back to original nodes."""
    expanded_nodes = set()
    for supernode in tree.nodes():
        expanded_nodes.update(supernode_map[supernode])
    return expanded_nodes


def _ensure_no_isolated_singles(partitions, original_graph, supernode_map):
    """
    Ensure no single nodes are isolated during partitioning.
    Isotope super-nodes are allowed to be isolated since they maintain internal connectivity.
    """
    # Find single-node partitions
    single_node_partitions = []
    multi_node_partitions = []
    
    for partition in partitions:
        if len(partition) == 1:
            node = list(partition)[0]
            
            # Check if this single node was part of an isotope family
            was_in_isotope_family = False
            for supernode, original_nodes in supernode_map.items():
                if node in original_nodes and len(original_nodes) > 1:
                    was_in_isotope_family = True
                    break
            
            if was_in_isotope_family:
                # This single node was extracted from an isotope family - it's fine to be isolated
                # because it maintains connectivity within the isotope group
                multi_node_partitions.append(partition)
            else:
                # This is a truly isolated single node - needs to be merged
                single_node_partitions.append(partition)
        else:
            multi_node_partitions.append(partition)
    
    if not single_node_partitions:
        return partitions  # No isolated singles to fix
    
    print(f"Found {len(single_node_partitions)} isolated single nodes, merging with neighbors...")
    
    # For each isolated single node, find the best partition to merge it with
    for single_partition in single_node_partitions:
        single_node = list(single_partition)[0]
        
        # Find all neighbors of this node in the original graph
        neighbors = set(original_graph.neighbors(single_node))
        
        if not neighbors:
            # Truly isolated node in original graph - keep as is
            multi_node_partitions.append(single_partition)
            continue
        
        # Find which existing partitions contain these neighbors
        neighbor_partitions = []
        for i, partition in enumerate(multi_node_partitions):
            if neighbors & partition:  # Intersection - partition contains some neighbors
                neighbor_partitions.append((i, len(neighbors & partition)))
        
        if neighbor_partitions:
            # Merge with partition that has the most neighbors
            best_partition_idx = max(neighbor_partitions, key=lambda x: x[1])[0]
            
            # Check if merging would exceed K limit
            target_partition = multi_node_partitions[best_partition_idx]
            if len(target_partition) < 50:  # Safe to merge
                multi_node_partitions[best_partition_idx] = target_partition | single_partition
                print(f"Merged isolated node {single_node} with partition of size {len(target_partition)}")
            else:
                # Target partition is full, find smallest partition instead
                smallest_partition_idx = min(range(len(multi_node_partitions)), 
                                           key=lambda i: len(multi_node_partitions[i]))
                if len(multi_node_partitions[smallest_partition_idx]) < 50:
                    multi_node_partitions[smallest_partition_idx] |= single_partition
                    print(f"Merged isolated node {single_node} with smallest partition")
                else:
                    # All partitions are full, keep as isolated (shouldn't happen)
                    multi_node_partitions.append(single_partition)
                    print(f"WARNING: Could not merge isolated node {single_node}, all partitions full")
        else:
            # No neighbors in any partition (shouldn't happen after proper splitting)
            multi_node_partitions.append(single_partition)
            print(f"WARNING: Isolated node {single_node} has no neighbors in any partition")
    
    return multi_node_partitions


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
        
        # Fill arrays efficiently, converting array indices to ROI values
        for idx, (i, j, edge_type, similarity) in enumerate(edges):
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

