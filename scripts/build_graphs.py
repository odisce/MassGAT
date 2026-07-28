#!/usr/bin/env python3
"""Build graph .pt files from Ionshed feather outputs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

try:
    from src.data.hierarchical_graph_construction import build_hierarchical_graph
    from src.config import MIN_SIGNALS
    from src.utils.config_utils import config_changed, save_config_metadata
except ModuleNotFoundError:
    # Allow direct execution of this file (python path/to/script.py)
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.data.hierarchical_graph_construction import build_hierarchical_graph
    from src.config import MIN_SIGNALS
    from src.utils.config_utils import config_changed, save_config_metadata


def log(message):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


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


def build_graphs(input_dir, output_dir, ionization_mode, eic_extraction_mode, force=False):

    start = time.time()
    peak_files = sorted(input_dir.glob("*peak*.feather"))
    signal_files = list(input_dir.glob("*signal*.feather")) + list(input_dir.glob("*roi*.feather"))
    log(f"Found {len(peak_files)} peak files and {len(signal_files)} signal files")

    if not peak_files:
        log("ERROR: No *peak*.feather files found")
        return 1
    
    config_path = Path(__file__).resolve().parents[1] / 'config.yaml'
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

    processed = 0
    skipped = 0
    failed = 0

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
            skipped += 1
            continue

        try:
            peaks_df = pd.read_feather(peak_file)
            signals_df = pd.read_feather(signal_file)

            if "sc_length" in peaks_df.columns:
                peaks_df = peaks_df[peaks_df["sc_length"] >= MIN_SIGNALS]
                filtered_rois = set(peaks_df["roi"])
                signals_df = signals_df[signals_df["roi"].isin(filtered_rois)]

            if peaks_df.empty:
                log(f"SKIP: No peaks left after filtering for {peak_file.name}")
                skipped += 1
                continue

            raw_signals = {
                roi: (g["rt"].values, g["i"].values)
                for roi, g in signals_df.groupby("roi")
            }

            edge_index, edge_attr, node_features, roi_values = build_hierarchical_graph(
                peaks_df,
                raw_signals,
                signals_df,
                file_id=None,
                show_progress=False,
                ionization_mode=ionization_mode,
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
                },
                str(out_file),
            )
            if config:
                save_config_metadata(out_file, config, ['graph', 'thresholds'])
            log(f"Saved: {out_file}")
            processed += 1

        except Exception as exc:
            log(f"ERROR processing {peak_file.name}: {exc}")
            failed += 1

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
    )


if __name__ == "__main__":
    sys.exit(main())
