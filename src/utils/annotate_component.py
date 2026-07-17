import numpy as np
import re

from src import config as project_config

PROTON = 1.007276
PPM_TOL = 5.0

ATOM_MASSES = {
    'H':  1.0072765,  'Na': 22.9897693, 'K':  38.9637068,
    'N':  14.0030740, 'C':  12.0000000, 'O':  15.9949146,
    'S':  31.9720707, 'Cl': 34.9688527, 'F':  18.9984032,
    'P':  30.973762,
}

ATOM_ORDER = ['H', 'Na', 'K', 'N', 'C', 'O', 'S', 'Cl', 'F', 'P']

MH_DELTA = {'H': +1}
M_MINUS_H_DELTA = {'H': -1}
MNa_DELTA = {'Na': +1}
MK_DELTA = {'K': +1}

def build_reverse_encoding(mass_diffs, adduct_diffs, dimer_diffs):
    reverse = {}
    for i, name in enumerate(mass_diffs.keys()):
        reverse[i] = (name, 'isotope')
    for i, name in enumerate(adduct_diffs.keys()):
        reverse[11 + i] = (name, 'adduct')
    for i, name in enumerate(dimer_diffs.keys()):
        reverse[31 + i] = (name, 'dimer')
    return reverse

ALL_MASS_DIFFS = {
    **project_config.ISOTOPE_DIFFS,
    **project_config.MOD_DIFFS_POSITIVE,
    **project_config.MOD_DIFFS_NEGATIVE,
    **project_config.DIMER_DIFFS_POSITIVE,
    **project_config.DIMER_DIFFS_NEGATIVE
}

ISOTOPE_MASS_SHIFTS = {}
for name, diff in project_config.ISOTOPE_DIFFS.items():
    iso_name = name.split('_')[1]  # Extract '13C' from 'isotope_13C'
    ISOTOPE_MASS_SHIFTS[iso_name] = diff

HEAVY_ISOTOPES = set(ISOTOPE_MASS_SHIFTS.keys())

def parse_formula(formula_str):
    """Parse formula string and return atom counts (e.g., 'H2O' -> {'H': 2, 'O': 1})."""
    counts = {}
    m = re.match(r'^([0-9]+)(.*)', formula_str)
    if m:
        multiplier = int(m.group(1))
        formula_str = m.group(2)
    else:
        multiplier = 1
        
    matches = re.findall(r'([A-Z][a-z]*)([0-9]*)', formula_str)
    for elem, count in matches:
        c = int(count) if count else 1
        counts[elem] = counts.get(elem, 0) + (c * multiplier)
        
    return counts

def parse_edge_name(edge_name):
    """Convert edge name to atom delta dict (e.g., 'adduct_Na-H' -> {'Na': 1, 'H': -1})."""
    if edge_name.startswith('isotope_'):
        return {edge_name.split('_')[1]: 1}
        
    chem_part = edge_name
    for prefix in ['adduct_', 'mod_', 'loss_']:
        if chem_part.startswith(prefix):
            chem_part = chem_part[len(prefix):]
            break
            
    if chem_part.startswith('dimer_2M'):
        chem_part = chem_part[len('dimer_2M'):]
        if not chem_part:
            return {}
    if chem_part and chem_part[0] not in ['+', '-']:
        chem_part = '+' + chem_part
        
    parts = re.findall(r'([+-])([A-Za-z0-9]+)', chem_part)
    
    delta = {}
    for sign, formula in parts:
        multiplier = 1 if sign == '+' else -1
        f_counts = parse_formula(formula)
        for elem, count in f_counts.items():
            delta[elem] = delta.get(elem, 0) + count * multiplier
            
    return {k: v for k, v in delta.items() if v != 0}

ADDUCT_DELTAS = {}
DIRECTION = {}
EDGE_ATOM_DELTAS = {}

for name, diff in ALL_MASS_DIFFS.items():
    ADDUCT_DELTAS[name] = diff
    
    if name.startswith(('isotope_', 'adduct_', 'dimer_')):
        DIRECTION[name] = 1
    elif name.startswith('loss_'):
        DIRECTION[name] = -1
    elif name.startswith('mod_'):
        DIRECTION[name] = 0
    else:
        DIRECTION[name] = 1 # Fallback
        
    EDGE_ATOM_DELTAS[name] = parse_edge_name(name)

POSITIVE_EDGE_NAMES = set(project_config.MOD_DIFFS_POSITIVE) | set(project_config.DIMER_DIFFS_POSITIVE)
NEGATIVE_EDGE_NAMES = set(project_config.MOD_DIFFS_NEGATIVE) | set(project_config.DIMER_DIFFS_NEGATIVE)

def _apply_delta(atoms, delta, sign=1):
    updated = dict(atoms)
    for atom, count in delta.items():
        new_val = updated.get(atom, 0) + sign * count
        if new_val == 0:
            updated.pop(atom, None)
        else:
            updated[atom] = new_val
    return updated

def _infer_mode_from_names(names):
    names = set(names)
    is_neg = bool(names & NEGATIVE_EDGE_NAMES)
    is_pos = bool(names & POSITIVE_EDGE_NAMES)
    
    if is_neg and not is_pos:
        return 'negative'
    if is_pos and not is_neg:
        return 'positive'
    return None

def _resolve_mode_settings(reverse_encoding=None, ionization_mode=None, base_delta=None):
    mode = ionization_mode
    if mode is None and reverse_encoding:
        mode = _infer_mode_from_names(name for name, _ in reverse_encoding.values())

    config_adducts = getattr(project_config, 'ADDUCT_DIFFS', {}) or {}
    config_dimers = getattr(project_config, 'DIMER_DIFFS', {}) or {}
    
    config_mode = None
    if config_adducts or config_dimers:
        config_mode = _infer_mode_from_names(set(config_adducts) | set(config_dimers))
        
    mode = mode or config_mode or 'positive'

    default_adducts = project_config.MOD_DIFFS_NEGATIVE if mode == 'negative' else project_config.MOD_DIFFS_POSITIVE
    default_dimers = project_config.DIMER_DIFFS_NEGATIVE if mode == 'negative' else project_config.DIMER_DIFFS_POSITIVE
    
    if reverse_encoding is None:
        reverse_encoding = build_reverse_encoding(
            project_config.ISOTOPE_DIFFS,
            config_adducts if config_mode == mode else default_adducts,
            config_dimers if config_mode == mode else default_dimers,
        )

    charge = '-' if mode == 'negative' else '+'
    return mode, reverse_encoding, (base_delta or (M_MINUS_H_DELTA if mode == 'negative' else MH_DELTA)), charge

def _ppm_ok(observed, expected, ppm=PPM_TOL):
    return abs(observed - expected) / (expected + 1e-9) * 1e6 < ppm

def apply_edge_delta(parent_atoms, edge_name, parent_is_lighter):
    delta = EDGE_ATOM_DELTAS.get(edge_name)
    if delta is None:
        return None
    return _apply_delta(parent_atoms, delta, +1 if parent_is_lighter else -1)

def atoms_to_string(atoms, is_dimer=False, charge='+'):
    """Format atom dict to string: {'C': 6, 'H': 12, 'O': 6} -> '[M+C6+H12+O6]+'."""
    regular  = {k: v for k, v in atoms.items() if k not in HEAVY_ISOTOPES}
    isotopes = {k: v for k, v in atoms.items() if k in HEAVY_ISOTOPES}

    pos_terms, neg_terms = [], []
    sorted_regular = sorted(
        regular.items(),
        key=lambda x: (ATOM_ORDER.index(x[0]) if x[0] in ATOM_ORDER else 99, x[0])
    )
    for atom, count in sorted_regular:
        if count > 0:
            pos_terms.append(f"{atom}{count if abs(count) > 1 else ''}")
        elif count < 0:
            neg_terms.append(f"{atom}{abs(count) if abs(count) > 1 else ''}")

    base   = '2M' if is_dimer else 'M'
    result = f'[{base}'
    if pos_terms:
        result += '+' + '+'.join(pos_terms)
    if neg_terms:
        result += '-' + '-'.join(neg_terms)
    result += f']{charge}'

    if isotopes:
        iso_parts = []
        for iso, count in sorted(isotopes.items()):
            iso_parts.extend([iso] * count)
        result += '_' + '_'.join(iso_parts)

    return result

def get_raw_chem_string(ename, category):
    """Extract formula from edge name (e.g., 'loss_H2O' -> 'H2O')."""
    part = ename
    if category == 'isotope':
        return part.replace('isotope_', '')
    for pfx in ['adduct_', 'mod_', 'loss_', 'dimer_']:
        if part.startswith(pfx):
            part = part[len(pfx):]
            break
    return part

def format_chem_part(chem_str, is_gain):
    """Add or flip sign based on direction (+Na-H -> -Na+H if loss)."""
    if is_gain:
        if not chem_str.startswith(('+', '-')):
            return '+' + chem_str
        return chem_str
    else:
        if not chem_str.startswith(('+', '-')):
            chem_str = '+' + chem_str
        flipped = ''
        for char in chem_str:
            if char == '+': flipped += '-'
            elif char == '-': flipped += '+'
            else: flipped += char
        return flipped

def build_annotation_string(base, mods, isos, charge):
    """Build final annotation string with algebraic simplification."""
    simplified_mods = simplify_mods(mods)
    inner = base + simplified_mods
    result = f"[{inner}]{charge}"
    if isos:
        result += "_" + "_".join(isos)
    return result
    
def get_base_mod_str(base_delta):
    """Extract base adduct string (+H, -H, etc)."""
    if base_delta == MH_DELTA: return '+H'
    if base_delta == M_MINUS_H_DELTA: return '-H'
    res = ''
    for atom, count in base_delta.items():
        if count > 0: res += f"+{atom}{count if count>1 else ''}"
        else: res += f"-{atom}{abs(count) if abs(count)>1 else ''}"
    return res

def simplify_mods(mods_list):
    """Algebraically combine mods: ['+H', '+Na-H'] -> '+Na'."""
    combined = "".join(mods_list)
    if not combined:
        return ""
        
    terms = re.findall(r'([+-])([A-Za-z0-9]+)', combined)
    counts = {}
    for sign, term in terms:
        multiplier = 1 if sign == '+' else -1
        m = re.match(r'^([0-9]*)([A-Za-z].*)$', term)
        if m:
            num_str, species = m.groups()
            num = int(num_str) if num_str else 1
        else:
            num = 1
            species = term
            
        counts[species] = counts.get(species, 0) + num * multiplier
        
    pos_terms = []
    neg_terms = []
    
    def sort_key(s):
        if s == 'H': return 0
        if s == 'Na': return 1
        if s == 'K': return 2
        if s == 'NH4': return 3
        if s == 'Cl': return 4
        return 5
        
    sorted_species = sorted(counts.keys(), key=lambda x: (sort_key(x), x))
    
    for species in sorted_species:
        count = counts[species]
        if count == 0: continue
            
        abs_count = abs(count)
        prefix = str(abs_count) if abs_count > 1 else ""
        
        if count > 0:
            pos_terms.append(f"+{prefix}{species}")
        else:
            neg_terms.append(f"-{prefix}{species}")
            
    return "".join(pos_terms) + "".join(neg_terms)

def atoms_to_mass_addition(atoms: dict) -> float:
    total = 0.0
    for atom, count in atoms.items():
        if atom in HEAVY_ISOTOPES:
            total += ISOTOPE_MASS_SHIFTS[atom] * count
        else:
            total += ATOM_MASSES.get(atom, 0.0) * count
    return total


def compute_neutral_mass_regression(atom_deltas, gidx_to_mz, gidx_to_prob, prob_threshold):
    estimates = []
    for gidx, (atoms, is_dimer, t_base, t_mods, t_isos, t_charge) in atom_deltas.items():
        if is_dimer or t_charge in ["2+", "2-"]:
            continue
        prob = gidx_to_prob.get(gidx, 0.0)
        if prob >= prob_threshold:
            observed_mz = gidx_to_mz[gidx]
            addition = atoms_to_mass_addition(atoms)
            M_i = observed_mz - addition
            estimates.append((gidx, M_i, prob))

    if not estimates:
        return None

    M_vals  = np.array([e[1] for e in estimates])
    weights = np.array([e[2] for e in estimates])
    weights = weights / weights.sum()
    neutral_mass = np.sum(weights * M_vals)
    std          = np.sqrt(np.sum(weights * (M_vals - neutral_mass) ** 2))
    return {'neutral_mass': neutral_mass, 'std': std, 'n_estimates': len(estimates)}

# Propagate annotations for a given candidate
def propagate_annotations(candidate_gidx, component_gidx_set, component_edges,
                           gidx_to_mz, gidx_to_prob, prob_threshold=0.5,
                           base_delta=MH_DELTA, charge='+'):
    
    adduct_edges  = [(s, d, n, c) for s, d, n, c in component_edges if c == 'adduct']
    dimer_edges   = [(s, d, n, c) for s, d, n, c in component_edges if c == 'dimer']
    isotope_edges = [(s, d, n, c) for s, d, n, c in component_edges if c == 'isotope']

    base_mod_str = get_base_mod_str(base_delta)
    
    # Establish known multicharged pairs dynamically based on polarity
    if charge == '+':
        known_multicharged_pairs = ["2M", "2M+H", "2M+Na", "2M+NH4"]
    else:
        known_multicharged_pairs = ["2M-H", "2M+H", "2M+HCOO", "2M+Cl"]
    
    node_state = {
        candidate_gidx: (dict(base_delta), False, 'M', [base_mod_str], [], charge)
    }
    queue   = [candidate_gidx]
    visited = {candidate_gidx}

    while queue:
        cur = queue.pop(0)
        cur_atoms, cur_is_dimer, cur_base, cur_mods, cur_isos, cur_charge = node_state[cur]
        cur_mz = gidx_to_mz[cur]

        for src, dst, ename, cat in (adduct_edges + dimer_edges):
            if src != cur and dst != cur: continue
            other = dst if src == cur else src
            if other in visited: continue
            
            other_mz          = gidx_to_mz[other]
            parent_is_lighter = cur_mz < other_mz
            
            direction = DIRECTION.get(ename, 0)
            is_multicharged_dimer = False
            
            chem_part = get_raw_chem_string(ename, cat)
            
            # MULTICHARGED HEURISTIC: Match known hallucinated chemical strings
            if cat == 'dimer' and not parent_is_lighter:
                if chem_part in known_multicharged_pairs:
                    is_multicharged_dimer = True
            
            if not is_multicharged_dimer:
                if direction == 1 and not parent_is_lighter:
                    continue 
                if direction == -1 and parent_is_lighter:
                    continue

            if cat == 'dimer':
                if is_multicharged_dimer:
                    custom_charge = "2+" if charge == "+" else "2-"
                    custom_mods = list(cur_mods) + [base_mod_str] 
                    
                    # --- CHEMISTRY RULE ---
                    # Reversing the algorithm's dimer math (mz_heavy = 2*(mz_light - base) + dimer_adduct)
                    # to deduce the exact neutral loss that occurred on the z=2 node.
                    if charge == "+":
                        if chem_part == "2M+NH4":
                            custom_mods.append("-NH3")
                        elif chem_part == "2M+Na":
                            custom_mods.append("-Na+H")
                    elif charge == "-":
                        if chem_part == "2M+HCOO":
                            custom_mods.append("-HCOOH")
                        elif chem_part == "2M+Cl":
                            custom_mods.append("-HCl")
                        elif chem_part == "2M+H":
                            custom_mods.append("-2H") 
                        
                    node_state[other] = (dict(cur_atoms), cur_is_dimer, cur_base, custom_mods, list(cur_isos), custom_charge)
                    visited.add(other)
                    continue
                    
                if cur_is_dimer: continue
                
                neutral_atoms = _apply_delta(cur_atoms, base_delta, -1)
                doubled_atoms = {a: c * 2 for a, c in neutral_atoms.items()}
                dimer_edge_atoms = EDGE_ATOM_DELTAS.get(ename, {})
                child_atoms = _apply_delta(doubled_atoms, dimer_edge_atoms)
                
                m = re.match(r'^([0-9]+M)(.*)$', chem_part)
                if m:
                    d_base, d_mod = m.groups()
                else:
                    d_base, d_mod = '2M', ''
                
                mod_list = [format_chem_part(d_mod, True)] if d_mod else []
                node_state[other] = (child_atoms, True, d_base, mod_list, list(cur_isos), charge)
                visited.add(other)
                queue.append(other)

            elif cat == 'adduct':
                child_atoms = apply_edge_delta(cur_atoms, ename, parent_is_lighter)
                if child_atoms is None: continue
                
                mod_str = format_chem_part(chem_part, parent_is_lighter)
                node_state[other] = (child_atoms, cur_is_dimer, cur_base, list(cur_mods) + [mod_str], list(cur_isos), charge)
                visited.add(other)
                queue.append(other)

    iso_queue   = list(node_state.keys())
    iso_visited = set(node_state.keys())

    while iso_queue:
        cur = iso_queue.pop(0)
        cur_atoms, cur_is_dimer, cur_base, cur_mods, cur_isos, cur_charge = node_state[cur]
        cur_mz = gidx_to_mz[cur]

        for src, dst, ename, cat in isotope_edges:
            if src != cur and dst != cur: continue
            other = dst if src == cur else src
            if other in iso_visited: continue
            
            other_mz = gidx_to_mz[other]
            if other_mz <= cur_mz: continue
            
            child_atoms = apply_edge_delta(cur_atoms, ename, parent_is_lighter=True)
            if child_atoms is None: continue
            
            chem_part = get_raw_chem_string(ename, cat)
            node_state[other] = (child_atoms, cur_is_dimer, cur_base, list(cur_mods), list(cur_isos) + [chem_part], cur_charge)
            iso_visited.add(other)
            iso_queue.append(other)

    return node_state


def annotate_component(component_gidx_set, node_features, edge_index, edge_attr,
                        node_probs, file_name, roi_values, reverse_encoding=None,
                        prob_threshold=0.5, temperature=2.0, base_delta=None, ionization_mode=None):
    
    ionization_mode, reverse_encoding, base_delta, charge = _resolve_mode_settings(
        reverse_encoding=reverse_encoding,
        ionization_mode=ionization_mode,
        base_delta=base_delta,
    )

    gidx_to_roi  = {i: roi_values[i].item() for i in range(len(roi_values))}
    roi_to_gidx  = {v: k for k, v in gidx_to_roi.items()}
    gidx_to_mz   = {i: node_features[i, 0].item() for i in component_gidx_set}
    gidx_to_int  = {i: node_features[i, 2].item() for i in component_gidx_set}
    gidx_to_prob = {
        i: node_probs.get((file_name, gidx_to_roi[i]), 0.0)
        for i in component_gidx_set
    }

    valid_nodes = {i for i in component_gidx_set if gidx_to_prob[i] >= prob_threshold}
    
    component_edges = []
    for ei in range(edge_index.shape[1]):
        src_roi  = edge_index[0, ei].item()
        dst_roi  = edge_index[1, ei].item()
        src_gidx = roi_to_gidx.get(src_roi)
        dst_gidx = roi_to_gidx.get(dst_roi)
        
        if src_gidx in valid_nodes and dst_gidx in valid_nodes:
            etype_int = int(edge_attr[ei, 1].item())
            if etype_int in reverse_encoding:
                etype_name, category = reverse_encoding[etype_int]
                component_edges.append((src_gidx, dst_gidx, etype_name, category))

    isotope_child_set = set()
    for src_gidx, dst_gidx, etype_name, category in component_edges:
        if category == 'isotope':
            heavier = src_gidx if gidx_to_mz[src_gidx] > gidx_to_mz[dst_gidx] else dst_gidx
            isotope_child_set.add(heavier)

    if ionization_mode == 'positive':
        known_multicharged_pairs = ["2M", "2M+H", "2M+Na", "2M+NH4"]
    else:
        known_multicharged_pairs = ["2M-H", "2M+H", "2M+HCOO", "2M+Cl"]

    dimer_gidx_set = set()
    for src_gidx, dst_gidx, etype_name, category in component_edges:
        if category == 'dimer':
            lighter = src_gidx if gidx_to_mz[src_gidx] < gidx_to_mz[dst_gidx] else dst_gidx
            heavier = dst_gidx if lighter == src_gidx else src_gidx
            
            chem_part = get_raw_chem_string(etype_name, category)
            
            # BYPASS 1: Known multicharged illusions
            if chem_part in known_multicharged_pairs:
                continue 
                
            # BYPASS 2: Allow true dimers to be candidate roots
            continue 
            
            dimer_gidx_set.add(heavier)

    dimer_and_children = set(dimer_gidx_set)
    for src_gidx, dst_gidx, etype_name, category in component_edges:
        if category == 'isotope':
            lighter = src_gidx if gidx_to_mz[src_gidx] < gidx_to_mz[dst_gidx] else dst_gidx
            heavier = dst_gidx if lighter == src_gidx else src_gidx
            if lighter in dimer_gidx_set:
                dimer_and_children.add(heavier)

    excluded = isotope_child_set | dimer_and_children
    candidates = [i for i in valid_nodes if i not in excluded]

    if not candidates:
        candidates = list(valid_nodes)
    if not candidates:
        return []

    raw_ints = [gidx_to_int[i] for i in candidates]
    max_int = max(raw_ints) if raw_ints and max(raw_ints) > 0 else 1.0
    max_prob = max([gidx_to_prob[i] for i in candidates]) if candidates else 1.0

    results = []

    for candidate_gidx in candidates:
        candidate_mz  = gidx_to_mz[candidate_gidx]
        candidate_int = gidx_to_int[candidate_gidx]
        implied_M     = candidate_mz - atoms_to_mass_addition(base_delta)
        children      = {}

        for src_gidx, dst_gidx, etype_name, category in component_edges:
            if src_gidx != candidate_gidx and dst_gidx != candidate_gidx:
                continue
            
            other_gidx = dst_gidx if src_gidx == candidate_gidx else src_gidx

            delta     = ADDUCT_DELTAS.get(etype_name)
            if delta is None: continue
            direction = DIRECTION.get(etype_name, 0)
            other_mz  = gidx_to_mz[other_gidx]
            other_int = gidx_to_int[other_gidx]
            is_child  = False

            if category == 'isotope':
                if other_mz > candidate_mz and _ppm_ok(other_mz, candidate_mz + delta):
                    is_child = True
            elif category == 'dimer':
                chem_part = get_raw_chem_string(etype_name, category)
                if other_mz > candidate_mz and _ppm_ok(other_mz, 2.0 * implied_M + delta):
                    is_child = True
                elif other_mz < candidate_mz and chem_part in known_multicharged_pairs:
                    is_child = True
            elif category == 'adduct':
                if direction == +1:
                    if other_mz > candidate_mz and _ppm_ok(other_mz, candidate_mz + delta):
                        is_child = True
                elif direction == -1:
                    if (other_mz < candidate_mz
                            and _ppm_ok(other_mz, candidate_mz - delta)
                            and other_int < candidate_int):
                        is_child = True
                else:
                    if (_ppm_ok(other_mz, candidate_mz + delta) or
                            _ppm_ok(other_mz, candidate_mz - delta)):
                        is_child = True

            if is_child:
                children[other_gidx] = etype_name

        n_children         = len(children)
        n_isotope_children = sum(1 for n in children.values() if n.startswith('isotope'))
        
        norm_intensity = candidate_int / max_int
        norm_prob = gidx_to_prob[candidate_gidx] / max_prob

        atom_deltas = propagate_annotations(
            candidate_gidx, component_gidx_set, component_edges,
            gidx_to_mz, gidx_to_prob, prob_threshold,
            base_delta=base_delta, charge=charge
        )

        mass_regression = compute_neutral_mass_regression(atom_deltas, gidx_to_mz, gidx_to_prob, prob_threshold)

        node_annotations = {}
        for gidx, (atoms, is_dimer, t_base, t_mods, t_isos, t_charge) in atom_deltas.items():
            annotation = build_annotation_string(t_base, t_mods, t_isos, t_charge)
            node_annotations[gidx_to_roi[gidx]] = annotation

        if n_children >= 2 and n_isotope_children >= 1:
            confidence = 'high'
        elif n_children >= 1:
            confidence = 'medium'
        else:
            confidence = 'low'

        results.append({
            'candidate_roi':        gidx_to_roi[candidate_gidx],
            'candidate_gidx':       candidate_gidx,
            'implied_neutral_mass': round(implied_M, 6),
            'mass_regression':      mass_regression,
            'node_annotations':     node_annotations,
            'n_children':           n_children,
            'n_isotope_children':   n_isotope_children,
            'norm_intensity':       round(float(norm_intensity), 4),
            'norm_prob':            round(float(norm_prob), 4),
            'confidence':           confidence,
        })

    # Normalized Scoring
    if results:
        max_n_c = max(r['n_children'] for r in results) or 1
        max_n_iso = max(r['n_isotope_children'] for r in results) or 1
        for r in results:
            r['score'] = (r['n_children'] / max_n_c * 1.0 + 
                          r['n_isotope_children'] / max_n_iso * 0.5 +
                          r['norm_intensity'] * 2.0 + 
                          r['norm_prob'] * 0.5)

    results.sort(key=lambda x: -x['score'])

    if len(results) > 1:
        raw_scores = np.array([r['score'] for r in results])
        raw_scores = raw_scores - raw_scores.max()
        exp_scores = np.exp(raw_scores / temperature)
        softmax    = exp_scores / exp_scores.sum()
        for r, p in zip(results, softmax):
            r['annotation_prob'] = float(p)
    elif results:
        results[0]['annotation_prob'] = 1.0

    return results
if __name__ == "__main__":
    print("Mods, adducts, and dimers loaded from config:")
    print("Isotopes:", list(project_config.ISOTOPE_DIFFS.keys()))
    print("Mods positive:", list(project_config.MOD_DIFFS_POSITIVE.keys()))