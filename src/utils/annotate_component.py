import numpy as np
import re

from src import config as project_config

PROTON = 1.007276
ELECTRON = 0.00054858
PPM_TOL = 5.0

# Neutral monoisotopic masses; the electron is accounted for with the charge
ATOM_MASSES = {
    'H':  1.0078250,  'Na': 22.9897693, 'K':  38.9637068,
    'N':  14.0030740, 'C':  12.0000000, 'O':  15.9949146,
    'S':  31.9720707, 'Cl': 34.9688527, 'F':  18.9984032,
    'P':  30.973762,
}

ATOM_ORDER = ['H', 'Na', 'K', 'N', 'C', 'O', 'S', 'Cl', 'F', 'P']

MH_DELTA = {'H': +1}
M_MINUS_H_DELTA = {'H': -1}
MNa_DELTA = {'Na': +1}
MK_DELTA = {'K': +1}
MNH4_DELTA = {'N': +1, 'H': +4}
MCl_DELTA = {'Cl': +1}
MHCOO_DELTA = {'H': +1, 'C': +1, 'O': +2}

# Main ions tried for each candidate root when ion_hypotheses='default'
POSITIVE_ION_HYPOTHESES = [MH_DELTA, MNa_DELTA, MNH4_DELTA, MK_DELTA]
NEGATIVE_ION_HYPOTHESES = [M_MINUS_H_DELTA, MCl_DELTA, MHCOO_DELTA]

# Atoms a metabolite cannot lose: [M+H+K-Na]+ is not a real ion
NON_REMOVABLE_ATOMS = ('Na', 'K', 'Cl')

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
    """The ionization mode given by the caller wins over the one inferred from the config or the edge names."""
    if ionization_mode not in (None, 'positive', 'negative'):
        raise ValueError(f"Unknown ionization mode: {ionization_mode}")
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

BASE_LABELS = {'+H': MH_DELTA, '-H': M_MINUS_H_DELTA, '+Na': MNa_DELTA, '+K': MK_DELTA,
               '+NH4': MNH4_DELTA, '+Cl': MCl_DELTA, '+HCOO': MHCOO_DELTA}

def get_base_mod_str(base_delta):
    """Extract base adduct string (+H, -H, +NH4, etc)."""
    for label, delta in BASE_LABELS.items():
        if base_delta == delta:
            return label
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

    # One notation for the ammonium adduct: +H+NH3 -> +NH4, +NH4-NH3 -> +H
    n = min(counts.get('H', 0), counts.get('NH3', 0))
    if n > 0:
        counts['H'] -= n
        counts['NH3'] -= n
        counts['NH4'] = counts.get('NH4', 0) + n
    n = min(counts.get('NH4', 0), -counts.get('NH3', 0))
    if n > 0:
        counts['NH4'] -= n
        counts['NH3'] += n
        counts['H'] = counts.get('H', 0) + n

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

def atoms_to_mass_addition(atoms: dict, charge='+') -> float:
    """Mass added to M by the atom deltas of a singly charged ion (electron included)."""
    total = 0.0
    for atom, count in atoms.items():
        if atom in HEAVY_ISOTOPES:
            total += ISOTOPE_MASS_SHIFTS[atom] * count
        else:
            total += ATOM_MASSES.get(atom, 0.0) * count
    return total + (ELECTRON if charge.endswith('-') else -ELECTRON)


def is_consistent_ion(atoms):
    """False when the annotation removes an atom that the metabolite does not carry (Na, K, Cl)."""
    return all(atoms.get(a, 0) >= 0 for a in NON_REMOVABLE_ATOMS)


# Charge carriers and salt exchanges are not restricted; C2H3N is the solvent adduct
IONIC_TERMS = ('H', 'Na', 'K', 'NH4', 'Cl', 'HCOO')
SOLVENT_TERMS = ('C2H3N',)


def plausible_ion(mods):
    """False when the ion gains a neutral molecule, carries two solvent adducts, or a solvent adduct and a loss."""
    gained = solvent = lost = 0
    for sign, num, term in re.findall(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)', mods):
        if term in IONIC_TERMS:
            continue
        n = int(num or 1)
        if sign == '-':
            lost += n
        elif term in SOLVENT_TERMS:
            solvent += n
        else:
            gained += n
    return gained == 0 and solvent <= 1 and not (solvent and lost)


def carries_loss(mods):
    """True when the modifications of an ion include a lost neutral (a removed H, Na, K or Cl is not one)."""
    return any(sign == '-' and term not in ('H', 'Na', 'K', 'Cl')
               for sign, _, term in re.findall(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)', simplify_mods(mods)))


K41_SHIFT = 1.998119


def isotope_label(parent_mz, child_mz, chem_part, parent_atoms):
    """'41K' for the M+2 isotope of an ion carrying K when the shift is closer to 41K than to the edge type."""
    if chem_part in ('34S', '37Cl') and parent_atoms.get('K', 0) > 0:
        shift = child_mz - parent_mz
        if abs(shift - K41_SHIFT) < abs(shift - ISOTOPE_MASS_SHIFTS.get(chem_part, K41_SHIFT)):
            return '41K'
    return chem_part


EXPLAIN_PPM = 10.0
C13_SHIFT = 1.00336


def doubly_charged(mz, rt, intensity, mz_tol=0.0015, rt_tol=5.0):
    """Nodes with a co-eluting, less intense partner at +1.00336/2 m/z: their isotope spacing says z = 2."""
    mz, rt, intensity = (np.asarray(a, dtype=float) for a in (mz, rt, intensity))
    order = np.argsort(mz)
    smz = mz[order]
    lo = np.searchsorted(smz, mz + C13_SHIFT / 2 - mz_tol, 'left')
    hi = np.searchsorted(smz, mz + C13_SHIFT / 2 + mz_tol, 'right')
    out = np.zeros(len(mz), dtype=bool)
    for i in np.where(hi > lo)[0]:
        cand = order[lo[i]:hi[i]]
        out[i] = np.any((np.abs(rt[cand] - rt[i]) <= rt_tol) & (intensity[cand] < intensity[i]))
    return out


def explains(observed_mz, atoms, is_dimer, charge, neutral_mass, ppm=EXPLAIN_PPM):
    """True when the annotation of a node gives back the neutral mass of the root."""
    if charge in ('2+', '2-'):
        # one more proton added (removed) than in the singly charged ion
        sign = 1 if charge == '2+' else -1
        implied = 2.0 * observed_mz - atoms_to_mass_addition(atoms, charge[1]) - sign * PROTON
    else:
        implied = observed_mz - atoms_to_mass_addition(atoms, charge)
    if is_dimer:
        implied /= 2.0
    return abs(implied - neutral_mass) / (neutral_mass + 1e-9) * 1e6 <= ppm


def compute_neutral_mass_regression(atom_deltas, gidx_to_mz, gidx_to_prob, prob_threshold):
    """Probability-weighted mean of the neutral masses implied by the annotated nodes."""
    estimates = []
    for gidx, (atoms, is_dimer, t_base, t_mods, t_isos, t_charge) in atom_deltas.items():
        if is_dimer or t_charge in ["2+", "2-"]:
            continue
        prob = gidx_to_prob.get(gidx, 0.0)
        if prob >= prob_threshold:
            observed_mz = gidx_to_mz[gidx]
            addition = atoms_to_mass_addition(atoms, t_charge)
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
                           base_delta=MH_DELTA, charge='+', dimer_after_loss=True):
    """Annotation of the nodes reached from the candidate. dimer_after_loss=False: a dimer edge is not followed from
    an ion carrying a neutral loss (a dimer forms from the intact molecule; doubly charged ions are not concerned)."""
    
    adduct_edges  = [(s, d, n, c) for s, d, n, c in component_edges if c == 'adduct']
    dimer_edges   = [(s, d, n, c) for s, d, n, c in component_edges if c == 'dimer']
    isotope_edges = [(s, d, n, c) for s, d, n, c in component_edges if c == 'isotope']

    base_mod_str = get_base_mod_str(base_delta)
    
    # Establish known multicharged pairs dynamically based on polarity
    if charge == '+':
        known_multicharged_pairs = ["2M+H", "2M+Na", "2M+NH4"]
    else:
        known_multicharged_pairs = ["2M-H", "2M+HCOO", "2M+Cl"]
    
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
                    
                    # Reversing the algorithm's dimer computation (mz_heavy = 2*(mz_light - base) + dimer_adduct)
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
                if not dimer_after_loss and carries_loss(cur_mods): continue
                
                neutral_atoms = _apply_delta(cur_atoms, base_delta, -1)
                doubled_atoms = {a: c * 2 for a, c in neutral_atoms.items()}
                dimer_edge_atoms = EDGE_ATOM_DELTAS.get(ename, {})
                child_atoms = _apply_delta(doubled_atoms, dimer_edge_atoms)
                
                m = re.match(r'^([0-9]+M)(.*)$', chem_part)
                if m:
                    d_base, d_mod = m.groups()
                else:
                    d_base, d_mod = '2M', ''
                
                # --- NEW TRUE DIMER STRING ALGEBRA ---
                # 1. Reverse the base ionizing atom (e.g., +H becomes -H)
                rev_base = format_chem_part(base_mod_str, False)
                
                # 2. Duplicate parent mods, subtract ionizing atom twice, add the dimer adduct (X)
                mod_list = list(cur_mods) + list(cur_mods) + [rev_base, rev_base]
                if d_mod:
                    mod_list.append(format_chem_part(d_mod, True))
                    
                # 3. Double the isotopes (if a 13C molecule dimerizes, it has two 13C atoms)
                new_isos = list(cur_isos) + list(cur_isos)
                
                node_state[other] = (child_atoms, True, d_base, mod_list, new_isos, charge)
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
            
            chem_part = isotope_label(cur_mz, other_mz, get_raw_chem_string(ename, cat), cur_atoms)
            node_state[other] = (child_atoms, cur_is_dimer, cur_base, list(cur_mods), list(cur_isos) + [chem_part], cur_charge)
            iso_visited.add(other)
            iso_queue.append(other)

    return node_state


def annotate_component(component_gidx_set, node_features, edge_index, edge_attr,
                        node_probs, file_name, roi_values, reverse_encoding=None,
                        prob_threshold=0.5, temperature=2.0, base_delta=None, ionization_mode=None,
                        ion_hypotheses=None, score_mode='children', charge_states=None, plausibility=False,
                        dimer_after_loss=True):
    """Rank the candidate main ions of a component.

    ion_hypotheses: None (each root is the base_delta ion, [M+H]+ or [M-H]- by default),
    'default' (also [M+Na]+, [M+NH4]+, [M+K]+ or [M+Cl]-, [M+HCOO]-) or a list of atom deltas.
    An alternative hypothesis is kept for a root only when it gives fewer inconsistent
    annotations (e.g. [M+H+K-Na]+) than the first one; the score is then weighted by the share
    of consistent annotations, and ties keep the order of the list.
    score_mode: 'children' (number of direct neighbours, first version) or 'explained' (number of consistent ions
    whose annotation gives back the neutral mass of the root, see explains).
    charge_states: {roi: 2} for doubly charged nodes (see doubly_charged); a singly charged annotation of such a
    node is inconsistent.
    plausibility: an implausible ion (see plausible_ion) is inconsistent and left out of the node annotations,
    with its isotopes.
    dimer_after_loss: see propagate_annotations.
    """
    if score_mode not in ('children', 'explained'):
        raise ValueError(f"Unknown score mode: {score_mode}")
    ionization_mode, reverse_encoding, base_delta, charge = _resolve_mode_settings(
        reverse_encoding=reverse_encoding,
        ionization_mode=ionization_mode,
        base_delta=base_delta,
    )
    if ion_hypotheses is None:
        hypotheses = [base_delta]
    elif ion_hypotheses == 'default':
        hypotheses = NEGATIVE_ION_HYPOTHESES if ionization_mode == 'negative' else POSITIVE_ION_HYPOTHESES
    else:
        hypotheses = list(ion_hypotheses)

    # called once per validated component to generate candidate annotations
    gidx_to_roi  = dict(enumerate(roi_values.tolist()))
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

    for candidate_gidx, hypothesis in ((c, h) for c in candidates for h in hypotheses):
        candidate_mz  = gidx_to_mz[candidate_gidx]
        candidate_int = gidx_to_int[candidate_gidx]
        implied_M     = candidate_mz - atoms_to_mass_addition(hypothesis, charge)
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
            base_delta=hypothesis, charge=charge, dimer_after_loss=dimer_after_loss
        )

        mass_regression = compute_neutral_mass_regression(atom_deltas, gidx_to_mz, gidx_to_prob, prob_threshold)

        node_annotations = {}
        n_inconsistent = 0
        n_explained = 0
        for gidx, (atoms, is_dimer, t_base, t_mods, t_isos, t_charge) in atom_deltas.items():
            if plausibility and not plausible_ion(simplify_mods(t_mods)):
                n_inconsistent += 1
                continue
            annotation = build_annotation_string(t_base, t_mods, t_isos, t_charge)
            node_annotations[gidx_to_roi[gidx]] = annotation
            consistent = is_consistent_ion(atoms)
            if charge_states and charge_states.get(gidx_to_roi[gidx], 1) == 2 and t_charge in ('+', '-'):
                consistent = False
            n_inconsistent += not consistent
            n_explained += consistent and explains(gidx_to_mz[gidx], atoms, is_dimer, t_charge, implied_M)

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
            'ion_hypothesis':       build_annotation_string('M', [get_base_mod_str(hypothesis)], [], charge),
            'n_inconsistent':       n_inconsistent,
            'n_explained':          n_explained,
            'n_propagated':         len(atom_deltas),
        })
        if hypothesis is not hypotheses[0]:
            first = next(r for r in results if r['candidate_gidx'] == candidate_gidx)
            if n_inconsistent >= first['n_inconsistent']:
                results.pop()

    # Normalized Scoring
    if results:
        count = 'n_children' if score_mode == 'children' else 'n_explained'
        max_n_c = max(r[count] for r in results) or 1
        max_n_iso = max(r['n_isotope_children'] for r in results) or 1
        for r in results:
            r['score'] = (r[count] / max_n_c * 1.0 + 
                          r['n_isotope_children'] / max_n_iso * 0.5 +
                          r['norm_intensity'] * 2.0 + 
                          r['norm_prob'] * 0.5)
            if len(hypotheses) > 1 or charge_states or plausibility:
                r['score'] *= 1.0 - r['n_inconsistent'] / max(r['n_propagated'], 1)

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