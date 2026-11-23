"""Tests of the fixes made for the revision: ionization mode of the annotation, training on real data, edge orientation,
edge loss weight, main-ion hypotheses and electron mass, MASSGAT_CONFIG, the dimer mass differences, and the annotation rules
of the article (plausible ions, dimers after a loss, 41K, explained ions, charge states, component size, M+2 isotopes).

Run with: python -m pytest tests/test_revision.py
"""

import argparse
import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.data.hierarchical_graph_construction import (convert_to_pytorch_format, heavy_isotope_label,  # noqa: E402
                                                      limit_component_size)
from src.data.loader import GraphDataset  # noqa: E402
from src.utils.annotate_component import annotate_component, doubly_charged, plausible_ion, simplify_mods  # noqa: E402

spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

PROTON = 1.007276
M = 180.06339                                   # neutral hexose
NA_ION, K_ION, NH4_ION = 22.98922, 38.96316, 18.03383
POS = list(config.MOD_DIFFS_POSITIVE)
ISO_TYPE = list(config.ISOTOPE_DIFFS).index('isotope_13C')
C13 = config.ISOTOPE_DIFFS['isotope_13C']


def component(mz, intensities, edges, file_name='graph.pt'):
    """Nodes in real units, as predict.py passes them; edges = [(i, j, edge type number)]."""
    n = len(mz)
    x = torch.zeros(n, 7, dtype=torch.float64)
    x[:, 0] = torch.tensor(mz, dtype=torch.float64)
    x[:, 2] = torch.tensor(intensities, dtype=torch.float64)
    return dict(component_gidx_set=set(range(n)), node_features=x,
                edge_index=torch.tensor([[i for i, _, _ in edges], [j for _, j, _ in edges]]),
                edge_attr=torch.tensor([[0.95, float(t)] for _, _, t in edges]),
                node_probs={(file_name, i): 0.9 for i in range(n)}, file_name=file_name, roi_values=torch.arange(n),
                prob_threshold=0.5, temperature=3.0)


def pos_type(name):
    return 11 + POS.index(name)


def negative_component():
    """[M-H]-, [M+Cl]- and the 13C isotope of [M-H]-."""
    m_h = M - PROTON
    cl_type = 11 + list(config.MOD_DIFFS_NEGATIVE).index('adduct_Cl+H')
    return component([m_h, m_h + config.MOD_DIFFS_NEGATIVE['adduct_Cl+H'], m_h + C13], [1e6, 3e5, 7e4],
                     [(0, 1, cl_type), (0, 2, ISO_TYPE)])


def sodiated_component():
    """[M+Na]+ (most intense), [M+K]+ and the 13C isotope of [M+Na]+, without [M+H]+."""
    return component([M + NA_ION, M + K_ION, M + NA_ION + C13], [1e6, 2e5, 7e4],
                     [(0, 1, pos_type('adduct_K-Na')), (0, 2, ISO_TYPE)])


def test_negative_mode_gives_negative_annotations():
    ann = annotate_component(**negative_component(), ionization_mode='negative')[0]['node_annotations']
    assert ann == {0: '[M-H]-', 1: '[M+Cl]-', 2: '[M-H]-_13C'}


def test_user_mode_wins_over_config():
    config.configure_for_mode('positive')
    try:
        ann = annotate_component(**negative_component(), ionization_mode='negative')[0]['node_annotations']
    finally:
        config.ADDUCT_DIFFS, config.DIMER_DIFFS = {}, {}
    assert all(a.split('_')[0].endswith('-') for a in ann.values())


def test_unknown_mode_is_rejected():
    try:
        annotate_component(**negative_component(), ionization_mode='neg')
    except ValueError:
        return
    raise AssertionError('an unknown ionization mode should raise ValueError')


def test_predict_passes_the_mode_to_the_annotation():
    tree = ast.parse((ROOT / 'scripts' / 'predict.py').read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'annotate_component']
    assert calls and all(any(k.arg == 'ionization_mode' for k in c.keywords) for c in calls)
    options = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, 'attr', None) == 'add_argument' and n.args and isinstance(n.args[0], ast.Constant)]
    assert '--ionization_mode' in options


def _predict_function(name):
    """One function of scripts/predict.py, without importing the script (and the model) itself."""
    tree = ast.parse((ROOT / 'scripts' / 'predict.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'os': os, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'predict.py', 'exec'), namespace)
    return namespace[name]


def test_mode_stored_in_the_graph_is_used_when_the_option_is_missing():
    modes = _predict_function('_ionization_modes')
    with tempfile.TemporaryDirectory() as d:
        torch.save({'edge_index': torch.zeros(2, 0), 'ionization_mode': 'negative'}, os.path.join(d, 'graph_a.pt'))
        torch.save({'edge_index': torch.zeros(2, 0)}, os.path.join(d, 'graph_old.pt'))
        assert modes([os.path.join(d, 'graph_a.pt')]) == {'graph_a.pt': 'negative'}
        assert modes([os.path.join(d, 'graph_old.pt')], 'negative') == {'graph_old.pt': 'negative'}
        try:
            modes([os.path.join(d, 'graph_old.pt')])
        except ValueError:
            pass
        else:
            raise AssertionError('a graph without stored mode should not be annotated in positive mode silently')


def test_build_graphs_stores_the_mode():
    tree = ast.parse((ROOT / 'scripts' / 'build_graphs.py').read_text(encoding='utf-8'))
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'save']
    keys = {k.value for s in saves for d in s.args if isinstance(d, ast.Dict) for k in d.keys if isinstance(k, ast.Constant)}
    assert 'ionization_mode' in keys


def test_sodiated_root_with_a_single_hypothesis_is_inconsistent():
    top = annotate_component(**sodiated_component())[0]
    assert top['node_annotations'][1] == '[M+H+K-Na]+'
    assert top['n_inconsistent'] == 1


def test_alternative_hypotheses_recover_the_sodium_adduct():
    top = annotate_component(**sodiated_component(), ion_hypotheses='default')[0]
    assert top['ion_hypothesis'] == '[M+Na]+'
    assert top['node_annotations'] == {0: '[M+Na]+', 1: '[M+K]+', 2: '[M+Na]+_13C'}
    assert abs(top['implied_neutral_mass'] - M) / M * 1e6 < 1


def test_protonated_root_is_unchanged_by_the_hypotheses():
    comp = component([M + PROTON, M + NA_ION], [1e6, 3e5], [(0, 1, pos_type('adduct_Na-H'))])
    single = annotate_component(**comp)
    multi = annotate_component(**comp, ion_hypotheses='default')
    assert single[0]['node_annotations'] == multi[0]['node_annotations'] == {0: '[M+H]+', 1: '[M+Na]+'}
    assert len(single) == len(multi)
    assert abs(single[0]['implied_neutral_mass'] - M) < 1e-5


def test_ammonium_adduct_has_one_notation():
    comp = component([M + PROTON, M + NH4_ION], [1e6, 3e5], [(0, 1, pos_type('mod_NH3'))])
    assert annotate_component(**comp)[0]['node_annotations'] == {0: '[M+H]+', 1: '[M+NH4]+'}
    assert simplify_mods(['+NH4', '-NH3']) == '+H'


def test_neutral_mass_regression_includes_the_electron():
    comp = component([M + PROTON, M + NA_ION], [1e6, 3e5], [(0, 1, pos_type('adduct_Na-H'))])
    reg = annotate_component(**comp)[0]['mass_regression']
    assert abs(reg['neutral_mass'] - M) < 1e-4 and reg['std'] < 1e-4


def test_edges_go_from_the_lighter_to_the_heavier_ion():
    config.configure_for_mode('positive')
    peaks = pd.DataFrame({'roi': [10, 20, 30], 'mzmed': [181.0707, 180.0673, 203.0526], 'rtmeanw': [60.0, 60.0, 60.0]})
    edges = [(0, 1, 'isotope_13C', 0.9), (2, 1, 'adduct_Na-H', 0.95), (1, 2, 'adduct_Na-H', 0.8)]
    edge_index, _, _, _ = convert_to_pytorch_format(edges, peaks, {}, None, show_progress=False)
    assert edge_index.t().tolist() == [[20, 10], [20, 30], [20, 30]]


def write_component(path, n, is_true, seed):
    """A labelled component in the per-component format of the training files (times in seconds)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, 72)
    x[:, 0] = 150 + 100 * torch.rand(n, generator=g)
    x[:, 1] = 300 + 600 * torch.rand(n, generator=g)
    x[:, 2] = 1e5 * (1 + 9 * torch.rand(n, generator=g))
    x[:, 3], x[:, 4] = x[:, 1] - 8, x[:, 1] + 9
    x[:, 5] = x[:, 2] / 5
    x[:, 6] = x[:, 0]
    x[:, 7] = 12
    x[:, 8:20] = x[:, 3:4] + torch.arange(12) * 1.5
    x[:, 40:52] = torch.rand(n, 12, generator=g) + 0.1
    edge_index = torch.stack([torch.zeros(n - 1, dtype=torch.long), torch.arange(1, n)])
    edge_attr = torch.stack([0.6 + 0.4 * torch.rand(n - 1, generator=g), torch.full((n - 1,), 11.)], dim=1)
    labels = torch.zeros(n, dtype=torch.long) if is_true else torch.full((n,), -1, dtype=torch.long)
    torch.save(dict(node_features=x, edge_index=edge_index, edge_attr=edge_attr, roi_values=torch.arange(n),
                    ground_truth_labels=labels, positive_mask=torch.ones(n, dtype=torch.bool)), path)


def real_folder(root, n_components=12):
    folder = Path(root) / "real_graphs"
    folder.mkdir()
    for i in range(n_components):
        write_component(folder / f"real_{i}.pt", n=3 + i % 3, is_true=i % 2 == 0, seed=i)
    return str(folder)


def train_args(**kwargs):
    defaults = dict(learning_rate=3e-4, num_epochs=1, batch_size=4, hidden_dim=8, embedding_dim=8, alpha=1.0, beta=1.0,
                    augmentation='none', seed=42, use_synthetic=False, synthetic_path='does/not/exist', real_path=None,
                    min_graphs=1, load_from_cache=False, model_type='hierarchical', model_choice='baseline',
                    encoder_type='transformer', block_type='gate', zero_metadata_all_nodes=False)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_real_folder_is_loaded_for_training():
    with tempfile.TemporaryDirectory() as d:
        graphs, num_classes = train.load_training_graphs(train_args(real_path=[real_folder(d)]))
        assert len(graphs) == 12 and num_classes == 2


def test_no_training_data_is_a_clear_error():
    for args in (train_args(), train_args(use_synthetic=True), train_args(real_path=['does/not/exist'])):
        try:
            train.load_training_graphs(args)
        except (ValueError, FileNotFoundError) as e:
            assert 'data' in str(e)
            continue
        raise AssertionError('a missing data source should raise an error')


def test_one_epoch_on_real_data():
    noop = lambda *a, **k: None
    patched = dict(setup_mlflow=lambda **k: 'test', log_training_params=noop, log_training_metrics=noop, end_mlflow_run=noop)
    saved = {k: getattr(train, k) for k in patched}
    cuda, log_artifact, cwd = torch.cuda.is_available, train.mlflow.log_artifact, os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        try:
            for k, v in patched.items():
                setattr(train, k, v)
            torch.cuda.is_available = lambda: False
            train.mlflow.log_artifact = noop
            os.chdir(d)
            _, path, metrics = train.main(train_args(real_path=[real_folder(d)]))
            assert Path(path).is_file() and metrics['total_epochs'] == 1
        finally:
            os.chdir(cwd)
            torch.cuda.is_available, train.mlflow.log_artifact = cuda, log_artifact
            for k, v in saved.items():
                setattr(train, k, v)


def test_edge_loss_weight_reaches_the_loss():
    seen = []

    def loss(*args, **kwargs):
        seen.append(kwargs.get('delta'))
        return train.full_hierarchical_loss(*args, **kwargs)

    with tempfile.TemporaryDirectory() as d:
        graphs, _ = train.load_training_graphs(train_args(real_path=[real_folder(d)]))
    model = train.MassGAT(num_features=72, hidden_dim=8, embedding_dim=8)
    trainer = train.ChroMSTrainer(model, device='cpu', delta=3.0, loss_fn=loss)
    trainer.train(graphs[:8], graphs[8:], num_epochs=1, batch_size=4, augmentation='none')
    assert seen and set(seen) == {3.0}
    assert train.ChroMSTrainer(model).delta == 1.0


def write_real_file(path):
    """A whole-file real graph: components {0, 1, 2} (true) and {3, 4} (false), edges given by ROI id."""
    n = 5
    x = torch.zeros(n, 72)
    x[:, 0] = torch.tensor([180.06, 181.07, 202.05, 300.10, 301.11])
    x[:, 1] = torch.tensor([310., 311., 310., 620., 621.])
    x[:, 2], x[:, 5] = 1e5, 2e4
    x[:, 3], x[:, 4] = x[:, 1] - 8, x[:, 1] + 9
    x[:, 7] = 12
    x[:, 8:20] = x[:, 3:4] + torch.arange(12) * 1.5
    x[:, 40:52] = 1.0
    torch.save(dict(node_features=x, edge_index=torch.tensor([[10, 10, 13], [11, 12, 14]]),
                    edge_attr=torch.tensor([[0.9, 0.], [0.8, 11.], [0.95, 0.]]), roi_values=torch.tensor([10, 11, 12, 13, 14]),
                    ground_truth_labels=torch.tensor([0, 0, -2, -1, -1]), positive_mask=torch.ones(n, dtype=torch.bool)), path)


def test_real_graph_files_are_loaded_with_component_labels():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "real"
        folder.mkdir()
        write_real_file(folder / "graph_a.pt")
        write_real_file(folder / "graph_b.pt")
        ds = GraphDataset(folder, cache_file=str(Path(d) / "real.pt"), load_from_cache=False, load_synthetic=False)
        assert sorted((g.num_nodes, int(g.graph_label)) for g in ds.graphs) == [(2, 0), (2, 0), (3, 1), (3, 1)]
        assert sorted(ds.graph_labels) == [0, 0, 1, 1]


def test_a_graph_file_without_edges_keeps_the_edge_attribute_width():
    from src.data.loader import load_mzml_file
    with tempfile.TemporaryDirectory() as d:
        write_real_file(Path(d) / "graph_a.pt")
        g = torch.load(Path(d) / "graph_a.pt", weights_only=False)
        width = load_mzml_file(Path(d) / "graph_a.pt")[0][0].edge_attr.size(1)
        g["edge_index"], g["edge_attr"] = g["edge_index"][:, :0], g["edge_attr"][:0]
        torch.save(g, Path(d) / "graph_b.pt")
        comps, _ = load_mzml_file(Path(d) / "graph_b.pt")
    assert len(comps) == 5 and all(c.edge_attr.shape == (0, width) for c in comps)


def test_massgat_config_selects_another_file():
    text = (ROOT / 'config.yaml').read_text()
    text = text.replace('min_signals:', 'min_signals: 4 #').replace('adduct_corr:', 'adduct_corr: 0.8 #')
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / 'config.yaml'
        cfg.write_text(text + '\n')
        out = subprocess.run([sys.executable, '-c', 'from src import config; print(config.MIN_SIGNALS, config.ADDUCT_CORR_THRESHOLD)'],
                             cwd=ROOT, env=dict(os.environ, MASSGAT_CONFIG=str(cfg)), capture_output=True, text=True, check=True)
    assert out.stdout.split()[-2:] == ['4', '0.8']


def test_dimer_mass_differences_match_the_graph_builder():
    # The builder looks for 2 * (mz - H) + d in positive mode and 2 * (mz + H) + d in negative mode (H = 1.00782, neutral atom),
    # so the table holds neutral masses: each dimer is then found within 0.6 mDa of its exact m/z, inside the 1 mDa tolerance.
    h, electron = 1.00782, 0.00054858
    exact = {'dimer_2M+H': 2 * M + PROTON, 'dimer_2M+Na': 2 * M + 22.989218, 'dimer_2M+NH4': 2 * M + 18.033823,
             'dimer_2M-H': 2 * M - PROTON, 'dimer_2M+Cl': 2 * M + 34.968853 + electron, 'dimer_2M+HCOO': 2 * M + 44.997654 + electron}
    for name, diff in {**config.DIMER_DIFFS_POSITIVE, **config.DIMER_DIFFS_NEGATIVE}.items():
        found = 2 * M + 2 * PROTON - 2 * h + diff if name in config.DIMER_DIFFS_POSITIVE else 2 * M - 2 * PROTON + 2 * h + diff
        assert abs(found - exact[name]) < 6e-4, name
    # the NH4+ ion mass (18.03382) would be 1.1 mDa away, outside the tolerance of Table S2
    assert abs(2 * M + 2 * PROTON - 2 * h + 18.03382 - exact['dimer_2M+NH4']) > 1e-3


def named_component(mz, intensities, edges):
    """component() with the edges given by their names (isotope_13C, adduct_Na-H, dimer_2M+H, ...)."""
    def number(e):
        if e.startswith('isotope_'):
            return list(config.ISOTOPE_DIFFS).index(e)
        if e.startswith('dimer_'):
            return 31 + list(config.DIMER_DIFFS_POSITIVE).index(e)
        return pos_type(e)
    return component(mz, intensities, [(i, j, number(e)) for i, j, e in edges])


def root_of(comp, candidate, **options):
    return next(r for r in annotate_component(**comp, **options) if r['candidate_roi'] == candidate)


def test_plausible_ion_rules():
    for mods in ('+H', '+4Na+4Cl-H', '+2Na-H-H2O', '+H-CO2-2H2O', '+H+C2H3N', '+Na+K-H'):
        assert plausible_ion(mods), mods
    for mods in ('+H+H2O', '+H+C2H3N+H2O', '+H+2C2H3N', '+H+C2H3N-H2O', '+Na+NH3-H2O'):
        assert not plausible_ion(mods), mods


def test_a_neutral_gain_makes_the_precursor_the_root():
    # in-source fragment [M+H-H2O]+ more intense than [M+H]+: without the rule it is taken for the root and [M+H]+ for +H2O
    m = 103.06333
    comp = named_component([m + PROTON - 18.010565, m + PROTON], [2.7e5, 1.5e5], [(0, 1, 'mod_H2O')])
    assert annotate_component(**comp)[0]['node_annotations'] == {0: '[M+H]+', 1: '[M+H+H2O]+'}
    top = annotate_component(**comp, plausibility=True)[0]
    assert top['node_annotations'] == {1: '[M+H]+', 0: '[M+H-H2O]+'}
    assert abs(top['implied_neutral_mass'] - m) / m * 1e6 < 1


def test_an_implausible_ion_leaves_the_group_with_its_isotope():
    m = 146.06914
    mz = [m + PROTON, m + PROTON + 41.026549, m + PROTON + 41.026549 + 18.010565, m + PROTON + 41.026549 + 18.010565 + C13]
    comp = named_component(mz, [1e6, 3e5, 1e5, 6e3], [(0, 1, 'adduct_C2H3N'), (1, 2, 'mod_H2O'), (2, 3, 'isotope_13C')])
    assert root_of(comp, 0)['node_annotations'][2] == '[M+H+C2H3N+H2O]+'
    kept = root_of(comp, 0, plausibility=True)
    assert kept['node_annotations'] == {0: '[M+H]+', 1: '[M+H+C2H3N]+'}
    assert kept['n_inconsistent'] == 2 and kept['n_propagated'] == 4


def test_no_dimer_edge_from_an_ion_with_a_loss():
    # [M+H]+, its fragment [M+H-HCOOH]+, the "dimer" of the fragment with its 13C, and the dimer of [M+H]+
    m, hcooh = 176.0321, 46.00548
    frag = m + PROTON - hcooh
    mz = [m + PROTON, frag, 2 * (frag - PROTON) + PROTON, 2 * (frag - PROTON) + PROTON + C13, 2 * m + PROTON]
    comp = named_component(mz, [1e6, 1e4, 3e4, 2e3, 5e4], [(1, 0, 'loss_HCOOH'), (1, 2, 'dimer_2M+H'),
                                                           (2, 3, 'isotope_13C'), (0, 4, 'dimer_2M+H')])
    assert {2, 3} <= set(root_of(comp, 0)['node_annotations'])
    kept = root_of(comp, 0, dimer_after_loss=False)['node_annotations']
    assert 2 not in kept and 3 not in kept
    assert kept[1] == '[M+H-HCOOH]+' and kept[4].startswith('[2M')


def test_41K_label_on_potassium_adducts_only():
    for ion, hypotheses, label in ((K_ION, [{'K': 1}], '[M+K]+_41K'), (PROTON, None, '[M+H]+_34S')):
        comp = named_component([M + ion, M + ion + 1.998119], [1e6, 7e4], [(0, 1, 'isotope_34S')])
        assert root_of(comp, 0, ion_hypotheses=hypotheses)['node_annotations'][1] == label


def test_explained_ions_follow_chains_of_relations():
    # [M+H]+ -> loss of H2O -> loss of CO, and the 13C isotope of [M+H]+: four ions explained by the root [M+H]+
    h2o, co = config.MOD_DIFFS_POSITIVE['mod_H2O'], config.MOD_DIFFS_POSITIVE['loss_CO']
    mz = [M + PROTON, M + PROTON - h2o, M + PROTON - h2o - co, M + PROTON + C13]
    comp = named_component(mz, [1e6, 5e5, 2e5, 7e4], [(1, 0, 'mod_H2O'), (2, 1, 'loss_CO'), (0, 3, 'isotope_13C')])
    top = annotate_component(**comp, score_mode='explained')[0]
    assert top['node_annotations'] == {0: '[M+H]+', 1: '[M+H-H2O]+', 2: '[M+H-CO-H2O]+', 3: '[M+H]+_13C'}
    assert top['n_children'] == 2 and top['n_explained'] == 4
    try:
        annotate_component(**comp, score_mode='both')
    except ValueError:
        return
    raise AssertionError('an unknown score mode should raise ValueError')


def test_isotope_spacing_gives_the_charge():
    mz = [307.0832, 307.5849, 613.1593, 614.1627]
    rt = [254.0, 253.8, 253.7, 253.7]
    assert doubly_charged(mz, rt, [1.9e7, 5.2e6, 8.2e6, 2.1e6]).tolist() == [True, False, False, False]
    assert not doubly_charged(mz, [254.0, 270.0, 253.7, 253.7], [1.9e7, 5.2e6, 8.2e6, 2.1e6])[0]   # not co-eluting


def test_charge_state_separates_a_doubly_charged_ion_from_a_dimer():
    # [M+2H]2+ (most intense) and [M+H]+ of oxidized glutathione, read by the graph as a monomer and its [2M+H]+ dimer
    big = 612.15197
    mz = [(big + 2 * PROTON) / 2, big + PROTON, big + PROTON + C13]
    comp = named_component(mz, [1.9e7, 8.2e6, 2.1e6], [(0, 1, 'dimer_2M+H'), (1, 2, 'isotope_13C')])
    plain = annotate_component(**comp, score_mode='explained')[0]
    assert abs(plain['implied_neutral_mass'] - big / 2) < 0.01            # the half mass wins without the charge
    checked = annotate_component(**comp, score_mode='explained', charge_states={0: 2})[0]
    assert abs(checked['implied_neutral_mass'] - big) < 0.01
    assert checked['node_annotations'][0] == '[M+2H]2+'


def test_predict_annotates_with_the_rules_of_the_article():
    tree = ast.parse((ROOT / 'scripts' / 'predict.py').read_text(encoding='utf-8'))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'annotate_component')
    options = {k.arg: k.value for k in call.keywords}
    assert options['ion_hypotheses'].value == 'default' and options['score_mode'].value == 'explained'
    assert options['plausibility'].value is True and options['dimer_after_loss'].value is False
    assert 'charge_states' in options


def edge_sizes(edges, n):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((i, j) for i, j, _, _ in edges)
    return sorted(len(c) for c in nx.connected_components(G))


def test_merge_splits_a_large_component_on_its_weakest_bridge():
    # two stars of 4 peaks joined by one bridge, each leaf linked by a weaker edge than the bridge
    edges = [(0, 1, 'adduct_Na-H', 0.80), (0, 2, 'adduct_K-H', 0.81), (0, 3, 'adduct_NH4-H', 0.82),
             (4, 5, 'adduct_Na-H', 0.80), (4, 6, 'adduct_K-H', 0.81), (4, 7, 'adduct_NH4-H', 0.82), (3, 4, 'loss_H2O', 0.95)]
    peaks = pd.DataFrame({'roi': range(8)})
    limited, _ = limit_component_size(edges, peaks, K=4, method='merge')
    assert edge_sizes(limited, 8) == [4, 4] and (3, 4, 'loss_H2O', 0.95) not in limited
    cut, _ = limit_component_size(edges, peaks, K=4, method='cut')    # the first version isolates a leaf
    assert edge_sizes(cut, 8) == [1, 3, 4]
    iso = [(0, 1, 'isotope_13C', 0.9), (1, 2, 'isotope_13C2', 0.9), (0, 3, 'adduct_Na-H', 0.99)]
    assert edge_sizes(limit_component_size(iso, pd.DataFrame({'roi': range(4)}), K=2, method='merge')[0], 4) == [1, 3]


def test_heavy_isotope_label():
    tol = 0.0015
    assert heavy_isotope_label(1.99710, 0.38, tol) == 'isotope_37Cl'   # NaCl cluster: inside both windows
    assert heavy_isotope_label(1.99710, 0.09, tol) == 'isotope_34S'    # two S
    assert heavy_isotope_label(1.99760, 0.0045, tol) is None           # neither S, Cl nor K
    assert heavy_isotope_label(1.99812, 0.07, tol) == 'isotope_34S'    # 41K, kept as the published edge type
    assert heavy_isotope_label(1.99440, 0.30, tol) is None             # 34S shift, chlorine ratio


def test_the_ratio_rule_is_applied_whatever_the_order_of_the_mass_differences():
    import numpy as np
    from src.data.hierarchical_graph_construction import detect_isotope_patterns

    def labels(rule, shift, ratio):
        t = np.arange(0, 20, 0.5)
        shape = np.exp(-0.5 * ((t - 10) / 1.5) ** 2)
        peaks = pd.DataFrame({'roi': [1, 2], 'mzmed': [180.9678, 180.9678 + shift], 'rtmeanw': [10.0, 10.0], 'into': [1e6, 1e6 * ratio]})
        signals = pd.concat([pd.DataFrame({'roi': r, 'rt': t, 'i': shape * s}) for r, s in ((1, 1e5), (2, 1e5 * ratio))])
        config.configure_for_mode('negative')
        old = config.ISOTOPE_LABEL_RULE, config.MZ_TOLERANCE
        config.ISOTOPE_LABEL_RULE, config.MZ_TOLERANCE = rule, 0.0015
        try:
            groups = detect_isotope_patterns(peaks, signals, show_progress=False)
        finally:
            config.ISOTOPE_LABEL_RULE, config.MZ_TOLERANCE = old
        return [iso['edge_type'] for g in groups for iso in g['isotopes']]

    assert labels('legacy', 1.99710, 0.38) == ['isotope_34S']   # the 34S row is seen first and never checked
    assert labels('ratio', 1.99710, 0.38) == ['isotope_37Cl']
    assert labels('ratio', 1.99760, 0.0045) == []
