#!/usr/bin/env python3
"""
Generate synthetic families, build hierarchical graphs,
and save them as .pt files.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

# Add src/ to path for direct execution; when installed via pip, imports will work directly
HERE = Path(__file__).resolve().parent
SRC = str(HERE.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Parse arguments early to get the ionization mode
_parser = argparse.ArgumentParser(description='Demo: generate synthetic families and save hierarchical graphs', add_help=False)
_parser.add_argument('--mode', '-m', default='negative', help='ionization mode: positive or negative')
_args_partial, _ = _parser.parse_known_args()

# Set environment variable BEFORE importing synthetic_generation modules
os.environ["IONIZATION_MODE"] = _args_partial.mode

import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool

try:
    from synthetic_generation.data import graph_construction as hgc
except ImportError as e:
    print(f'ERROR: could not import synthetic_generation.data.graph_construction: {e}')
    raise

from synthetic_generation.generators.peak_families import (
    generate_true_with_noise_family,
    generate_false_peak_family,
    generate_true_peak_family,
)
from synthetic_generation import sample_mz, sample_intensity, sample_rt_center
from synthetic_generation.config import configure_for_mode


def _to_safe_torch_object(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_safe_torch_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_safe_torch_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_safe_torch_object(item) for item in value)
    return value


def family_to_peaks_signals(family):
    # Minimal version of conversion used in tests — returns (peaks_df, signals_df)
    peaks = []
    signals = []
    true_keys = ['main_peak', 'isotope_peaks', 'adduct_peaks', 'dimer_peaks']
    false_keys = ['noise_peaks']
    roi_counter = 0  # Initialize ROI counter

    for key in true_keys + false_keys:
        peak_list = family.get(key, [])
        if isinstance(peak_list, dict):
            peak_list = [peak_list]
        for peak in peak_list:
            # Assign sequential ROI values
            roi = roi_counter
            roi_counter += 1

            rtmeanw = np.average(peak['rt_profile'], weights=peak['intensity_profile'])
            if peak["type"] == "true_peak":
                is_true = True
            elif peak["type"] == "false_peak":
                is_true = False
            else: 
                raise ValueError(f"Unknown peak type: {peak['type']}")
            peaks.append({
                'roi': roi,
                'mzmed': peak['mz'],
                'rtmeanw': rtmeanw,
                'into': peak.get('intensity', np.sum(peak['intensity_profile'])),
                'rtmin': min(peak['rt_profile']),
                'rtmax': max(peak['rt_profile']),
                'imax': peak['max_intensity'],
                'mzmeanw': peak['mz'],
                'sc_length': peak['sc_length'],
                'is_true': is_true,
            })
            for rt, inten in zip(peak['rt_profile'], peak['intensity_profile']):
                signals.append({'roi': roi, 'rt': rt, 'i': inten})
    return pd.DataFrame(peaks), pd.DataFrame(signals)



# Add single peak generation for true/false peaks
from synthetic_generation.generators.peak_families import generate_true_peak, generate_false_peak, DEFAULT_SCAN_TIMES

FAMILY_FUNCS = {
    'true_with_noise': lambda: generate_true_with_noise_family(sample_mz(), sample_intensity(), sample_rt_center()),
    'false': lambda: generate_false_peak_family(main_mz=sample_mz(), main_rt=sample_rt_center()),
    'true': lambda: generate_true_peak_family(sample_mz(), sample_intensity(), sample_rt_center()),
}

def generate_true_single():
    mz = sample_mz()
    imax = sample_intensity()
    rt = sample_rt_center()
    peak = generate_true_peak(mz, imax, rt, scan_times=DEFAULT_SCAN_TIMES)
    return peak

def generate_false_single():
    mz = sample_mz()
    intensity = sample_intensity()
    rt = sample_rt_center()
    peak = generate_false_peak(mz, intensity, rt, scan_times=DEFAULT_SCAN_TIMES)
    return peak


def main():
    p = argparse.ArgumentParser(description='Demo: generate synthetic families and save hierarchical graphs')
    p.add_argument('--family', '-f', default='true_with_noise', help='family type (true_with_noise, false, multi_compound, large_false_component, true, true_single, false_single)')
    p.add_argument('--num', '-n', type=int, default=3, help='number of graphs to generate')
    p.add_argument('--out', '-o', default='graphs/demo', help='output directory to save .pt files')
    p.add_argument('--workers', '-w', type=int, default=1, help='number of worker processes (default 1)')
    p.add_argument('--seed', '-s', type=int, default=None, help='random seed for reproducibility (default: None)')
    p.add_argument('--mode', '-m', default='negative', help='ionization mode: positive or negative')
    p.add_argument('--eic_sampling_strategy', choices=['uniform', 'center_rt_window'], default='uniform', help='EIC feature sampling strategy')
    # TEMP ILLUSTRATION (easy removal): delete this arg block + the marked TEMP block below.
    p.add_argument('--illustrate_true_evolution', action='store_true', help='temporary: save no-noise EMG + main-ion preview artifacts for true-family generation')
    p.add_argument('--illustrate_num', type=int, default=12, help='number of first generated true/true_with_noise families to illustrate (default: 12)')
    p.add_argument('--family_size_bonus', type=int, default=2, help='temporary: add this many peaks to true-family size when illustrating')
    args = p.parse_args()

    # Fix the random seed for reproducibility
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        np.random.seed(args.seed)  # For numpy
        torch.manual_seed(args.seed)  # For PyTorch
        random.seed(args.seed)  # For Python's random module

    family_type = args.family
    num_graphs = args.num
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)


    # Handle new single peak types
    if family_type not in FAMILY_FUNCS and family_type not in ['true_single', 'false_single']:
        raise SystemExit(f'Unknown family type: {family_type}')

    # function used for serial or parallel generation
    def _generate_and_save(i, family_type, out_dir_str):
        TARGET_LEN = 32
        # Optional: Use a per-worker seed for multiprocessing
        if args.seed is not None:
            worker_seed = args.seed + i
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        if family_type == 'true_single' or family_type == 'false_single':
            # Generate a single true or false peak and save with the same
            # graph-like .pt schema used in demo_generate_graphs_2.py
            if family_type == 'true_single':
                peak = generate_true_single()
                label_val = 1
            else:
                peak = generate_false_single()
                label_val = -1

            rt_profile = np.array(peak['rt_profile'], dtype=np.float32)
            intensity_profile = np.array(peak['intensity_profile'], dtype=np.float32)

            def resize_profile(arr, target_len):
                n = len(arr)
                if n == 0:
                    return np.zeros(target_len, dtype=np.float32)
                if n == target_len:
                    return arr.astype(np.float32)
                if n < target_len:
                    padded = np.zeros(target_len, dtype=np.float32)
                    padded[:n] = arr
                    return padded
                x_old = np.linspace(0, 1, n)
                x_new = np.linspace(0, 1, target_len)
                return np.interp(x_new, x_old, arr).astype(np.float32)

            rt_resized = resize_profile(rt_profile, TARGET_LEN)
            int_resized = resize_profile(intensity_profile, TARGET_LEN)

            mz_val = float(peak['mz'])
            rt_center = float(peak['rt_center'])
            into_val = float(peak['intensity'])
            rt_min = float(rt_resized[rt_resized != 0].min()) if (rt_resized != 0).any() else rt_center
            rt_max = float(rt_resized.max())
            imax_val = float(peak['max_intensity'])
            sc_len = int(peak['sc_length'])

            meta = np.array([
                mz_val,
                rt_center,
                into_val,
                rt_min,
                rt_max,
                imax_val,
                mz_val,
                float(sc_len),
            ], dtype=np.float32)

            arr = np.concatenate([meta, rt_resized, int_resized])
            node_features = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

            ground_truth_labels = torch.tensor([label_val], dtype=torch.long)
            positive_mask = torch.tensor([True], dtype=torch.bool)
            roi_values = torch.tensor([i], dtype=torch.long)

            raw_edge_attr_dim = 2
            empty_edge_index = torch.empty((2, 0), dtype=torch.long)
            empty_edge_attr = torch.empty((0, raw_edge_attr_dim), dtype=torch.float32)

            out = {
                'edge_index': empty_edge_index,
                'edge_attr': empty_edge_attr,
                'node_features': node_features,
                'roi_values': roi_values,
                'ground_truth_labels': ground_truth_labels,
                'positive_mask': positive_mask,
                'file_name': f'peak_{family_type}_{i}',
                'n_peaks': 1,
                'n_edges': 0,
            }

            out_path = Path(out_dir_str) / f'peak_{family_type}_{i}.pt'
            torch.save(out, out_path)
        else:
            # ===== TEMP ILLUSTRATION BLOCK (START) =====
            illustration_payload = None
            if (
                args.illustrate_true_evolution
                and family_type in ('true', 'true_with_noise')
                and i < args.illustrate_num
            ):
                main_mz = sample_mz()
                main_imax = sample_intensity()
                main_rt = sample_rt_center()

                np_state = np.random.get_state()
                torch_state = torch.get_rng_state()
                py_state = random.getstate()

                peak_no_noise = generate_true_peak(
                    main_mz,
                    main_imax,
                    main_rt,
                    noise_level=0.0,
                    scan_times=DEFAULT_SCAN_TIMES,
                )

                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                random.setstate(py_state)
                peak_main_preview = generate_true_peak(
                    main_mz,
                    main_imax,
                    main_rt,
                    scan_times=DEFAULT_SCAN_TIMES,
                )

                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                random.setstate(py_state)

                if family_type == 'true_with_noise':
                    fam = generate_true_with_noise_family(main_mz, main_imax, main_rt)
                else:
                    fam = generate_true_peak_family(main_mz, main_imax, main_rt, family_size=4 + max(0, args.family_size_bonus))

                illustration_payload = {
                    'peak_no_noise': peak_no_noise,
                    'peak_main_preview': peak_main_preview,
                }
            else:
                fam = FAMILY_FUNCS[family_type]()

            peaks_df, signals_df = family_to_peaks_signals(fam)
            if 'sc_length' in peaks_df.columns:
                peaks_df = peaks_df[peaks_df['sc_length'] >= 4]
                signals_df = signals_df[signals_df['roi'].isin(peaks_df['roi'])]
            raw_signals = {roi: (group['rt'].values, group['i'].values) for roi, group in signals_df.groupby('roi')}
            
            result = hgc.build_hierarchical_graph(peaks_df, raw_signals, signals_df, file_id=None, has_annotations=False, show_progress=False, include_ground_truth=True, ionization_mode=args.mode, eic_sampling_strategy=args.eic_sampling_strategy)
            if len(result) == 4:
                edge_index, edge_attr, node_features, roi_values = result
                ground_truth_labels = None
                positive_mask = None
            else:
                edge_index, edge_attr, node_features, roi_values, ground_truth_labels, positive_mask = result
            base_name = f'{family_type}_{i}'
            out = {
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'node_features': node_features,
                'roi_values': _to_safe_torch_object(roi_values),
                'ground_truth_labels': _to_safe_torch_object(ground_truth_labels) if ground_truth_labels is not None else None,
                'positive_mask': _to_safe_torch_object(positive_mask) if positive_mask is not None else None,
                'file_name': base_name,
                'n_peaks': len(peaks_df),
                'n_edges': edge_index.shape[1],
            }
            out = _to_safe_torch_object(out)
            out_path = Path(out_dir_str) / f'graph_{base_name}.pt'
            torch.save(out, str(out_path))

            if illustration_payload is not None:
                ill_dir = Path(out_dir_str) / 'illustrations' / family_type
                ill_dir.mkdir(parents=True, exist_ok=True)

                main_peak_family = fam.get('main_peak', None)

                sidecar_path = ill_dir / f'evolution_{base_name}.pt'
                torch.save(
                    _to_safe_torch_object({
                        'emg_no_noise': {
                            'rt_profile': np.array(illustration_payload['peak_no_noise']['rt_profile'], dtype=np.float32),
                            'intensity_profile': np.array(illustration_payload['peak_no_noise']['intensity_profile'], dtype=np.float32),
                        },
                        'main_peak_preview': {
                            'rt_profile': np.array(illustration_payload['peak_main_preview']['rt_profile'], dtype=np.float32),
                            'intensity_profile': np.array(illustration_payload['peak_main_preview']['intensity_profile'], dtype=np.float32),
                        },
                        'main_peak_family': {
                            'rt_profile': np.array(main_peak_family['rt_profile'], dtype=np.float32) if main_peak_family is not None else None,
                            'intensity_profile': np.array(main_peak_family['intensity_profile'], dtype=np.float32) if main_peak_family is not None else None,
                        },
                        'graph_path': str(out_path),
                    }),
                    sidecar_path,
                )

                try:
                    import matplotlib.pyplot as plt
                    plt.style.use('seaborn-v0_8-whitegrid')
                    peaks_for_plot = []
                    for key in ('main_peak', 'isotope_peaks', 'adduct_peaks', 'dimer_peaks', 'noise_peaks'):
                        peaks_for_plot.extend(fam.get(key, []) if isinstance(fam.get(key, []), list) else [fam.get(key)] if fam.get(key) is not None else [])
                    profiles = [
                        ('1) EMG no noise', illustration_payload['peak_no_noise']),
                        ('2) Main true peak', illustration_payload['peak_main_preview']),
                    ]
                    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
                    for ax, (title, peak) in zip(axes[:2], profiles):
                        ax.plot(peak['rt_profile'], peak['intensity_profile'], lw=2.2, color='#1f77b4')
                        ax.fill_between(peak['rt_profile'], peak['intensity_profile'], alpha=0.18, color='#1f77b4')
                        ax.set_title(title, fontweight='bold')
                        ax.set_xlabel('RT (min)')
                        ax.set_ylabel('Intensity')

                    for peak in peaks_for_plot:
                        rt = np.asarray(peak['rt_profile'], dtype=float)
                        intensity = np.asarray(peak['intensity_profile'], dtype=float)
                        peak_max = float(np.max(intensity)) if np.max(intensity) > 0 else 1.0
                        axes[2].plot(rt, intensity / peak_max, lw=1.8, alpha=0.65)
                    axes[2].set_title('3) Individual peaks: max-normalized', fontweight='bold')
                    axes[2].set_xlabel('RT (min)')
                    axes[2].set_ylabel('Normalized intensity')

                    fig.tight_layout()
                    fig_path = ill_dir / f'evolution_{base_name}.png'
                    fig.savefig(fig_path, dpi=320, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f'Warning: could not save illustration plot for {base_name}: {e}')


    if args.workers and args.workers > 1:
        with Pool(args.workers) as pool:
            list(tqdm(pool.istarmap(_generate_and_save, [(i, family_type, str(out_dir)) for i in range(num_graphs)]), total=num_graphs, desc="Generating graphs"))
    else:
        for i in tqdm(range(num_graphs), desc="Generating graphs"):
            _generate_and_save(i, family_type, str(out_dir))

if __name__ == '__main__':
    main()
