"""Checks that the faster graph building and peak table code gives exactly what the former loops gave.

Run with: python -m pytest tests/test_t45_speedups.py   (or python tests/...py)
"""

import ast
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src import config  # noqa: E402
from src.data import hierarchical_graph_construction as hgc  # noqa: E402


def peaks_table(n=600, seed=0):
    """Peaks on a coarse m/z grid (many ties and exact isotope / adduct spacings) and a coarse RT grid."""
    rng = np.random.default_rng(seed)
    base = np.round(rng.uniform(80, 400, n // 3), 3)
    mz = np.concatenate([base, base + config.ISOTOPE_DIFFS['isotope_13C'], base + rng.choice([21.98194, 1.00336, 0.0], n - 2 * (n // 3))])
    rng.shuffle(mz)
    return pd.DataFrame({
        'roi': np.arange(1, n + 1, dtype=np.float64),
        'mzmed': mz.astype(np.float64),
        'rtmeanw': np.round(rng.uniform(0, 60, n), 1),
        'into': np.round(rng.lognormal(10, 2, n), 0),
        'sc_length': rng.integers(1, 40, n).astype(np.float64),
        'rtmin': np.zeros(n), 'rtmax': rng.uniform(1, 20, n),
        'mzmin': mz - 0.001, 'mzmax': mz + 0.001,
    })


def former_collect(peaks_df, mz_deltas, mz_tolerance, rt_tolerance):
    """collect_candidate_peak_pairs as it was written with loops (regular mass differences only)."""
    peaks_df_sorted = peaks_df.copy().sort_values("mzmed").reset_index(drop=True)
    mz = peaks_df_sorted["mzmed"].astype("float64").values
    rt = peaks_df_sorted["rtmeanw"].astype("float64").values
    roi = peaks_df_sorted["roi"].values
    n = len(peaks_df_sorted)
    for delta_name in mz_deltas:
        peaks_df_sorted[f"match_{delta_name}"] = np.nan
    pairs = []
    for delta_name, delta in mz_deltas.items():
        i = j = 0
        while i < n - 1:
            while j < n and mz[j] - mz[i] < delta - mz_tolerance:
                j += 1
            k = j
            while k < n and mz[k] - mz[i] <= delta + mz_tolerance:
                if abs(rt[k] - rt[i]) <= rt_tolerance:
                    pairs.append((roi[i], roi[k], delta_name))
                    pairs.append((roi[k], roi[i], delta_name))
                    peaks_df_sorted.at[k, f"match_{delta_name}"] = roi[i]
                    peaks_df_sorted.at[i, f"match_{delta_name}"] = roi[k]
                k += 1
            i += 1
    return peaks_df_sorted, pairs


def test_candidate_pairs_are_the_same_in_the_same_order():
    deltas = {'isotope_13C': config.ISOTOPE_DIFFS['isotope_13C'], 'adduct_Na-H': 21.98194, 'zero': 0.0, 'big': 150.0}
    for seed in range(4):
        df = peaks_table(seed=seed)
        for tol in (0.001, 0.005):
            ref_df, ref = former_collect(df, deltas, tol, 5.0)
            new_df, new = hgc.collect_candidate_peak_pairs(df.copy(), deltas, mz_tolerance=tol, rt_tolerance=5.0)
            assert new == ref and len(ref) > 0
            assert [type(a) for a, _, _ in new[:5]] == [type(a) for a, _, _ in ref[:5]]
            pd.testing.assert_frame_equal(new_df, ref_df)


def former_limit(edges, n_nodes, K):
    """limit_component_size(method='cut') with the cut loop on the whole graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i, j, edge_type, corr in edges:
        G.add_edge(i, j, edge_type=edge_type, corr=corr, score=hgc.edge_cut_score(edge_type, corr))
    while True:
        oversized_comps = [c for c in nx.connected_components(G) if len(c) > K]
        if not oversized_comps:
            break
        oversized = set().union(*oversized_comps)
        cuttable = sorted((d['score'], u, v) for u, v, d in G.edges(data=True) if u in oversized and np.isfinite(d['score']))
        if not cuttable:
            break
        batch = max(1, sum(len(c) // K for c in oversized_comps))
        G.remove_edges_from([(u, v) for _, u, v in cuttable[:batch]])
    return [(u, v, d['edge_type'], d['corr']) for u, v, d in G.edges(data=True)]


def test_component_cut_keeps_the_same_edges_in_the_same_order():
    rng = np.random.default_rng(1)
    for _ in range(5):
        n = 400
        edges = []
        for _ in range(900):
            i, j = rng.integers(0, n, 2)
            if i != j:
                etype = rng.choice(['isotope_13C', 'adduct_Na-H', 'loss_H2O', 'dimer_2M+H'])
                edges.append((int(i), int(j), etype, float(rng.choice([0.9, 0.95, 1.0]))))   # many equal scores
        edges += [(k, k + 1, 'adduct_Na-H', 0.9) for k in range(0, 60)]                       # a long chain
        for K in (5, 20, 50):
            new, _ = hgc.limit_component_size(edges, pd.DataFrame(index=range(n)), K=K, method='cut')
            assert new == former_limit(edges, n, K)


def test_profiles_are_the_same():
    rng = np.random.default_rng(2)
    rows = [(float(r), float(t), float(rng.uniform(0, 1e5))) for r in range(1, 300) for t in rng.choice(np.arange(0, 100, 0.5), 15)]
    sig = pd.DataFrame(rows, columns=['roi', 'rt', 'i'])
    sig = pd.concat([sig, sig.iloc[:50].assign(i=7.0)])                                          # duplicated (roi, rt)
    rois = set(np.arange(1, 320, dtype=np.float64))
    grid, prof = hgc.precompute_intensity_profiles(sig, rois)
    f = sig.sort_values('rt').groupby(['roi', 'rt'], as_index=False).agg({'i': 'max'})
    groups = f.groupby('roi')
    for roi in rois:
        ref = np.interp(grid, groups.get_group(roi)['rt'].values, groups.get_group(roi)['i'].values, left=0, right=0) \
            if roi in groups.groups else np.zeros_like(grid)
        assert np.array_equal(prof[roi], ref)


def test_signals_by_roi_matches_groupby():
    import build_graphs
    rng = np.random.default_rng(3)
    sig = pd.DataFrame({'roi': rng.integers(1, 50, 2000).astype(np.float64), 'rt': rng.uniform(0, 100, 2000),
                        'i': rng.uniform(0, 1, 2000)})
    sig.loc[5, 'roi'] = np.nan
    new = build_graphs.signals_by_roi(sig)
    ref = {roi: (g['rt'].values, g['i'].values) for roi, g in sig.groupby('roi')}
    assert list(new) == list(ref)
    assert all(np.array_equal(new[r][0], ref[r][0]) and np.array_equal(new[r][1], ref[r][1]) for r in ref)
    assert build_graphs.signals_by_roi(sig.iloc[:0]) == {}


def test_roi_table_of_predict_matches_iterrows(tmp_path):
    tree = ast.parse((ROOT / 'scripts' / 'predict.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_load_roi_mzmin_mzmax_mapping')
    namespace = {'os': __import__('os'), 'pd': pd, 'np': np, 'read_feather': pd.read_feather}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'predict.py', 'exec'), namespace)
    df = peaks_table()
    df.reset_index(drop=True).to_feather(tmp_path / 'f_peak.feather')
    new = namespace['_load_roi_mzmin_mzmax_mapping'](str(tmp_path), 'graph_f.pt')
    ref = {int(row['roi']): (float(row['mzmin']), float(row['mzmax'])) for _, row in df.iterrows()}
    assert list(new.items()) == list(ref.items())


def former_dimer_candidates(sorted_peaks, ionization_mode):
    """Candidate pairs of detect_dimer_patterns as written with loops."""
    import bisect
    sorted_mz, sorted_roi = sorted_peaks['mzmed'].values, sorted_peaks['roi'].values
    pairs = []
    for dimer_name, correction in config.ACTIVE_DIMER_DIFFS.items():
        for _, row in sorted_peaks.iterrows():
            adduct_h = config.ADDUCT_DIFFS.get('adduct_H', 1.00782)
            sign = -1 if ionization_mode == 'positive' else 1
            expected = 2 * (row['mzmed'] + sign * adduct_h) + correction
            tol = config.MZ_TOLERANCE
            for j in range(bisect.bisect_left(sorted_mz, expected - tol), bisect.bisect_right(sorted_mz, expected + tol)):
                if row['roi'] == sorted_roi[j] or abs(sorted_peaks.iloc[j]['rtmeanw'] - row['rtmeanw']) > config.RT_WINDOW:
                    continue
                if abs(sorted_mz[j] - expected) <= tol:
                    pairs.append((row['roi'], sorted_roi[j], dimer_name))
    return pairs


def test_dimer_candidates_are_the_same():
    captured = {}
    former_corr = hgc.compute_profile_correlations
    hgc.compute_profile_correlations = lambda pairs, *a, **k: captured.setdefault('pairs', pairs) and pd.DataFrame([])
    try:
        for mode in ('positive', 'negative'):
            config.configure_for_mode(mode)
            df = peaks_table(seed=5)
            adduct_h = config.ADDUCT_DIFFS.get('adduct_H', 1.00782)
            sign = -1 if mode == 'positive' else 1
            name, corr = next(iter(config.ACTIVE_DIMER_DIFFS.items()))
            dimers = df.iloc[:150].assign(roi=lambda d: d.roi + 10000, mzmed=lambda d: 2 * (d.mzmed + sign * adduct_h) + corr)
            df = pd.concat([df, dimers], ignore_index=True)
            captured.clear()
            hgc.detect_dimer_patterns(df, {r: i for i, r in enumerate(df.roi.values)}, set(range(len(df))), None,
                                      show_progress=False, ionization_mode=mode)
            main = df.copy().reset_index(drop=True)
            ref = former_dimer_candidates(main.iloc[main['mzmed'].argsort()].copy(), mode)
            assert captured['pairs'] == ref and len(ref) >= 150
    finally:
        hgc.compute_profile_correlations = former_corr
        config.ADDUCT_DIFFS, config.DIMER_DIFFS = {}, {}


def test_iterrows_of_a_table_with_text_gives_python_floats():
    # the edge lists keep the same values and types as with iterrows
    df = pd.DataFrame([{'roi_i': np.float64(1.0), 'roi_k': np.float64(2.0), 'delta': 'x', 'correlation': np.float64(0.91)}])
    row = next(df.iterrows())[1]
    new = next(iter(hgc._table_rows(df, ['roi_i', 'roi_k', 'delta', 'correlation'])))
    assert new == (row['roi_i'], row['roi_k'], row['delta'], row['correlation'])
    assert [type(v) for v in new] == [type(row[c]) for c in ['roi_i', 'roi_k', 'delta', 'correlation']]
    assert list(hgc._table_rows(pd.DataFrame([]), ['roi_i'])) == []


def test_lazy_imports_keep_the_package_names():
    import src.data
    import src.utils
    from src.data import load_mzml_file, build_hierarchical_graph  # noqa: F401
    from src.utils import set_seed  # noqa: F401
    assert callable(src.data.reindex_graph) and src.MIN_SIGNALS == config.MIN_SIGNALS


if __name__ == '__main__':
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            if 'tmp_path' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f'{name}: ok')
