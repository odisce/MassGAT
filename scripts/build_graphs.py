#!/usr/bin/env python3
"""Build graph .pt files from Ionshed feather outputs."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from src.data.hierarchical_graph_construction import build_hierarchical_graph
    from src.config import MIN_SIGNALS
    from src.utils.config_utils import config_changed, save_config_metadata
    from src.utils.common import read_feather
except ModuleNotFoundError:
    # Allow direct execution of this file (python path/to/script.py)
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.data.hierarchical_graph_construction import build_hierarchical_graph
    from src.config import MIN_SIGNALS
    from src.utils.config_utils import config_changed, save_config_metadata
    from src.utils.common import read_feather


def log(message):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def signals_by_roi(signals_df):
    """{roi: (rt, intensity)} with the rows of each ROI in table order, as signals_df.groupby('roi') gives them."""
    roi = signals_df["roi"].to_numpy()
    rows = np.flatnonzero(~pd.isna(roi))
    if rows.size == 0:
        return {}
    rows = rows[np.argsort(roi[rows], kind="stable")]
    roi, rt, intensity = roi[rows], signals_df["rt"].to_numpy()[rows], signals_df["i"].to_numpy()[rows]
    bounds = np.flatnonzero(np.r_[True, roi[1:] != roi[:-1], True])
    return {roi[a]: (rt[a:b], intensity[a:b]) for a, b in zip(bounds[:-1], bounds[1:])}


def find_matching_signal(peak_file):
    peak_name = peak_file.stem
    candidates: list[Path] = [
        peak_file.parent / f"{peak_name.replace('peak', 'signal')}.feather",
        peak_file.parent / f"{peak_name.replace('_peak', '_signal')}.feather",
        peak_file.parent / f"{peak_name.replace('_peak', '_roi')}.feather",
    ]

    all_signal_like = list(peak_file.parent.glob("*signal*.feather")) + list(
        peak_file.parent.glob("*roi*.feather")
    )

    peak_base = peak_name.replace("_peak", "").replace("peak", "")
    for cand in all_signal_like:
        signal_base = cand.stem.replace("_signal", "").replace("signal", "")
        if peak_base == signal_base:
            candidates.append(cand)

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def build_one(peak_file, signal_file, out_file, config, ionization_mode, eic_extraction_mode, log_fn=log):
    """Build and save the graph of one peak/signal pair; returns 'processed', 'skipped' or 'failed'."""
    try:
        peaks_df = read_feather(peak_file)
        signals_df = read_feather(signal_file)

        if "sc_length" in peaks_df.columns:
            peaks_df = peaks_df[peaks_df["sc_length"] >= MIN_SIGNALS]
            filtered_rois = set(peaks_df["roi"])
            signals_df = signals_df[signals_df["roi"].isin(filtered_rois)]

        if peaks_df.empty:
            log_fn(f"SKIP: No peaks left after filtering for {peak_file.name}")
            return "skipped"

        raw_signals = signals_by_roi(signals_df)

        limit_comp_size = None
        if isinstance(config, dict):
            limit_comp_size = config.get('graph', {}).get('max_component_size', None)

        edge_index, edge_attr, node_features, roi_values = build_hierarchical_graph(
            peaks_df,
            raw_signals,
            signals_df,
            file_id=None,
            show_progress=False,
            ionization_mode=ionization_mode,
            limit_component_size_value=limit_comp_size,
            eic_extraction_mode=eic_extraction_mode,
        )

        torch.save(
            {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "node_features": node_features,
                "roi_values": torch.from_numpy(roi_values),
                "ground_truth_labels": None,
                "positive_mask": None,
                "file_name": peak_file.name,
                "n_peaks": len(peaks_df),
                "n_edges": edge_index.shape[1],
                "ionization_mode": ionization_mode,
            },
            str(out_file),
        )
        if config:
            save_config_metadata(out_file, config, ['graph', 'thresholds'])
        log_fn(f"Saved: {out_file}")
        return "processed"

    except Exception as exc:
        log_fn(f"ERROR processing {peak_file.name}: {exc}")
        return "failed"


def _init_worker():
    torch.set_num_threads(1)


def _build_one_worker(*args):
    """build_one in a child process; the log lines are returned to the parent instead of printed."""
    lines = []

    def collect(message):
        lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    try:
        status = build_one(*args, log_fn=collect)
    except BaseException as exc:  # never let one file stop the pool
        collect(f"ERROR processing {args[0].name}: {exc}")
        status = "failed"
    return status, lines


def build_graphs(input_dir, output_dir, ionization_mode, eic_extraction_mode, force=False, n_jobs=1):

    start = time.time()
    peak_files = sorted(input_dir.glob("*peak*.feather"))
    signal_files = list(input_dir.glob("*signal*.feather")) + list(input_dir.glob("*roi*.feather"))
    log(f"Found {len(peak_files)} peak files and {len(signal_files)} signal files")

    if not peak_files:
        log("ERROR: No *peak*.feather files found")
        return 1
    
    config_path = Path(os.environ.get('MASSGAT_CONFIG') or Path(__file__).resolve().parents[1] / 'config.yaml')
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    file_pairs: list[tuple[Path, Path]] = []
    for peak_file in peak_files:
        signal_file = find_matching_signal(peak_file)
        if signal_file is None:
            log(f"WARNING: No matching signal file for {peak_file.name}")
            continue
        file_pairs.append((peak_file, signal_file))

    log(f"Found {len(file_pairs)} peak-signal pairs")
    if not file_pairs:
        log("ERROR: No matching pairs")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in output_dir.glob("graph_*.pt")}

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    n_jobs = max(1, min(n_jobs, (os.cpu_count() or 1) - 1))
    jobs = []

    for idx, (peak_file, signal_file) in enumerate(file_pairs, start=1):
        log(f"--- Processing {idx}/{len(file_pairs)}: {peak_file.name} ---")

        base_name = peak_file.stem.replace("_peak", "").replace("peak", "")
        out_name = f"graph_{base_name}"
        out_file = output_dir / f"{out_name}.pt"

        skip = out_name in existing and not force
        if skip and config and config_changed(out_file, config, ['graph', 'thresholds']):
            log(f"CONFIG CHANGED: Reprocessing {out_file.name}")
            skip = False
        
        if skip:
            log(f"SKIP: {out_file.name} already exists")
            counts["skipped"] += 1
            continue

        job = (peak_file, signal_file, out_file, config, ionization_mode, eic_extraction_mode)
        if n_jobs == 1:
            counts[build_one(*job)] += 1
        else:
            jobs.append(job)

    if jobs:
        # One file per task, one BLAS/torch thread per process; each child writes its own .pt file.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"
        n_workers = min(n_jobs, len(jobs))
        log(f"Building {len(jobs)} graphs with {n_workers} processes")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker) as pool:
            futures = {pool.submit(_build_one_worker, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                peak_file = futures[future]
                try:
                    status, lines = future.result()
                except Exception as exc:
                    status, lines = "failed", []
                    log(f"ERROR processing {peak_file.name}: {exc}")
                log(f"--- Done: {peak_file.name} ---")
                for line in lines:
                    print(line)
                counts[status] += 1

    processed, skipped, failed = counts["processed"], counts["skipped"], counts["failed"]
    total_min = (time.time() - start) / 60.0
    log("==================================================")
    log("General processing completed")
    log(
        f"Total pairs: {len(file_pairs)}, processed: {processed}, skipped: {skipped}, failed: {failed}, time_min: {total_min:.1f}"
    )
    return 0 if failed == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Build graph files from feather peak/signal files")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with feather files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write graph_*.pt files")
    parser.add_argument(
        "--ionization_mode",
        choices=["positive", "negative"],
        default="positive",
        help="Ionization mode used for graph construction",
    )
    parser.add_argument(
        "--eic_extraction_mode",
        choices=["uniform", "center_rt_window"],
        default="uniform",
        help="How to reduce EIC points to fixed length",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force reprocessing even if output files already exist",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of files processed in parallel, one process per file (at most cpu_count - 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_dir.exists():
        log(f"ERROR: Input directory not found: {args.input_dir}")
        return 1
    return build_graphs(
        args.input_dir,
        args.output_dir,
        args.ionization_mode,
        args.eic_extraction_mode,
        force=args.force,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    sys.exit(main())
