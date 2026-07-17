from __future__ import annotations
import math
from typing import Any
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objs as go
import sys
from pathlib import Path

# Setup imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src import config

def build_edge_names(diff_dicts):
    """Build mapping from edge type numbers to names."""
    names = {}
    # Isotopes (0-4)
    for i, name in enumerate(diff_dicts['isotope'].keys()):
        names[i] = name
    # Adducts/Modifications (11-30)
    for i, name in enumerate(diff_dicts['adduct'].keys()):
        names[11 + i] = name
    # Dimers (31-33)
    for i, name in enumerate(diff_dicts['dimer'].keys()):
        names[31 + i] = name
    names[35] = 'unknown'
    return names

EDGE_TYPE_NAMES_POSITIVE = build_edge_names({
    'isotope': config.ISOTOPE_DIFFS,
    'adduct': config.MOD_DIFFS_POSITIVE,
    'dimer': config.DIMER_DIFFS_POSITIVE
})


EDGE_TYPE_NAMES_NEGATIVE = build_edge_names({
    'isotope': config.ISOTOPE_DIFFS,
    'adduct': config.MOD_DIFFS_NEGATIVE,
    'dimer': config.DIMER_DIFFS_NEGATIVE
})


EDGE_COLORS = {
    'isotope': '#4CAF50',      # green
    'adduct': '#2196F3',       # blue
    'loss': "#2E15EA",         # red
    'mod': '#2E15EA',          # purple
    'dimer': '#FF9800',          # orange
    'unknown': '#9E9E9E'       # gray
}
def get_edge_category(edge_type):
    """Categorize edge type for coloring."""
    if 0 <= edge_type <= 4: return 'isotope'
    if 11 <= edge_type <= 30:
        name = EDGE_TYPE_NAMES_POSITIVE.get(edge_type, '')
        if 'adduct' in name: return 'adduct'
        if 'mod' in name: return 'mod'
        if 'loss' in name: return 'loss'
        return 'adduct'  # fallback
    if 31 <= edge_type <= 33: return 'dimer'
    return 'unknown'

def get_edge_color(edge_type):
    """Get color for an edge type."""
    return EDGE_COLORS.get(get_edge_category(edge_type), '#888')

def short_label(edge_name):
    """Strip prefix for compact display."""
    prefixes = ['isotope_', 'adduct_', 'mod_', 'loss_', 'dimer_']
    for prefix in prefixes:
        if edge_name.startswith(prefix):
            return edge_name[len(prefix):]
    return edge_name[:6]

def get_multicharged_labels(u, v, G, edge_type, edge_names):
    """Helper to dynamically rename multicharged edges."""
    edge_name = edge_names.get(edge_type, f'unknown_{edge_type}')
    cat = get_edge_category(edge_type)
    short = short_label(edge_name)
    
    # Our universal list of multicharged pairs
    multi_pairs = ["2M", "2M+H", "2M+Na", "2M+NH4", "2M-H", "2M+HCOO", "2M+Cl"]
    
    if cat == 'dimer' and short in multi_pairs:
        # You can change these strings to whatever you prefer!
        new_short = "Doubly charged"
        new_hover = f"<b>Doubly charged </b><br><i>Algorithm originally matched: {edge_name}</i>"
        return new_short, new_hover
        
    return short, f"{edge_name} (type {edge_type})"

def create_nx_graph(graph, ionization_mode='positive', node_probs=None, 
                    file_name=None, prob_threshold=0.5):
    """
    Convert PyTorch graph to NetworkX graph, filtering out nodes 
    with probability < prob_threshold.
    """
    edge_names = EDGE_TYPE_NAMES_POSITIVE if ionization_mode == 'positive' else EDGE_TYPE_NAMES_NEGATIVE
    G = nx.Graph()
    
    roi_values = graph['roi_values']
    roi_to_idx = {roi_values[i].item(): i for i in range(len(roi_values))}
    
    # 1. Identify Valid Nodes first
    valid_nodes = set()
    node_features = graph['node_features']
    for i in range(len(roi_values)):
        roi = roi_values[i].item()
        
        # Check probability if provided
        if node_probs is not None and file_name is not None:
            prob = node_probs.get((file_name, roi), 0.0)
            if prob < prob_threshold:
                continue
        
        valid_nodes.add(i)
        features = node_features[i, :8].numpy()
        G.add_node(i, 
            roi=roi, mz=features[0], rt=features[1], intensity=features[2],
            rt_min=features[3], rt_max=features[4], max_intensity=features[5],
            mzmean=features[6], scan_length=features[7]
        )
    
    # 2. Add edges only if BOTH nodes are in valid_nodes
    if 'edge_index' in graph and 'edge_attr' in graph:
        edge_index = graph['edge_index'].numpy()
        edge_attr = graph['edge_attr'].numpy()
        
        for i in range(edge_index.shape[1]):
            source_roi, target_roi = edge_index[0, i].item(), edge_index[1, i].item()
            source_idx = roi_to_idx.get(source_roi)
            target_idx = roi_to_idx.get(target_roi)
            
            # Only add edge if both endpoints are valid
            if source_idx in valid_nodes and target_idx in valid_nodes:
                cosine_sim = edge_attr[i, 0].item() if edge_attr.shape[1] > 0 else None
                edge_type = edge_attr[i, 1].item() if edge_attr.shape[1] > 1 else 0
                edge_name = edge_names.get(edge_type, f'unknown_{edge_type}')
                G.add_edge(source_idx, target_idx, type=edge_type, name=edge_name, cosine_similarity=cosine_sim)
    
    return G

def get_midpoint(pos, u, v, layout, intensities_are_log):
    """Calculate midpoint of an edge."""
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    mid_x = (x0 + x1) / 2
    
    if layout == 'spatial':
        mid_y = math.sqrt(max(y0, 0.1) * max(y1, 0.1))  # geometric mean in log space
    else:
        mid_y = (y0 + y1) / 2
    return mid_x, mid_y

def pick_edge_labels(subgraph, pos, layout, intensities_are_log, max_labels=24, min_sep=0.075):
    """Greedy edge label placement to avoid overlap."""
    if max_labels <= 0:
        return []
    
    # Calculate spans for normalization
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(max(xs) - min(xs), 1e-9)
    y_span = max(max(ys) - min(ys), 1e-9)
    
    # Score and sort edges
    candidates = []
    for u, v, data in subgraph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx = (x1 - x0) / x_span
        dy = (y1 - y0) / y_span
        length_score = math.hypot(dx, dy)
        mx, my = get_midpoint(pos, u, v, layout, intensities_are_log)
        candidates.append((length_score, (u, v, data), (mx, my)))
    
    candidates.sort(reverse=True)
    
    # Greedy selection
    selected = []
    selected_midpoints = []
    seen_types = set()
    
    for _, (u, v, data), (mx, my) in candidates:
        edge_type = data['type']
        
        # Always include first of each type
        if edge_type not in seen_types:
            selected.append((u, v, data, mx, my))
            selected_midpoints.append((mx, my))
            seen_types.add(edge_type)
            if len(selected) >= max_labels:
                break
            continue
        
        # Check distance to existing labels
        too_close = False
        for sx, sy in selected_midpoints:
            d = math.hypot((mx - sx) / x_span, (my - sy) / y_span)
            if d < min_sep:
                too_close = True
                break
        
        if not too_close:
            selected.append((u, v, data, mx, my))
            selected_midpoints.append((mx, my))
            if len(selected) >= max_labels:
                break
    
    return selected

def style_config(style='presentation'):
    """Rendering settings for static exports."""
    if style == 'article':
        return {
            'figsize': (16, 6),
            'title_size': 20, 'axis_label_size': 15, 'tick_size': 15,
            'legend_size': 15, 'legend_title_size': 15, 'label_size': 14,
            'label_pad': 0.12, 'node_scale': 1.0, 'edge_scale': 0.55,
            'max_edge_labels': 24, 'min_sep': 0.075
        }
    return {
        'figsize': (14, 8),
        'title_size': 14, 'axis_label_size': 11, 'tick_size': 10,
        'legend_size': 9, 'legend_title_size': 10, 'label_size': 14,
        'label_pad': 0.18, 'node_scale': 1.25, 'edge_scale': 0.75,
        'max_edge_labels': 80, 'min_sep': 0.045
    }

def compute_node_profiles(node_features, nodes):
    """Extract normalized RT/intensity profiles for a set of nodes.

    node_features: full (or component-sliced) node feature array/tensor,
        where columns [-64:-32] are RT bins and [-32:] are intensity bins,
        zero-padded.
    nodes: iterable of node indices into node_features (graph-level indices,
        i.e. matching the indices used elsewhere in the NetworkX graph G).

    Returns: {node_id: {"rt": [...], "intensity": [...]}} with intensity
        normalized to its own max and zero-padding stripped.
    """
    if hasattr(node_features, 'numpy'):
        node_features = node_features.numpy()

    nodes = list(nodes)
    rt_block = node_features[nodes, -64:-32]
    intensity_block = node_features[nodes, -32:]

    profiles = {}
    for i, node in enumerate(nodes):
        rt_row = rt_block[i]
        mask = rt_row != 0
        rt_no_padding = rt_row[mask]
        intensity_no_padding = intensity_block[i][mask]
        if intensity_no_padding.size == 0:
            continue
        intensity_normalized = intensity_no_padding / np.max(intensity_no_padding)
        profiles[node] = {
            "rt": rt_no_padding.tolist(),
            "intensity_raw": intensity_no_padding.tolist(),
            "intensity_norm": intensity_normalized.tolist()
        }
    return profiles


def get_coherent_nodes(G, root_node, edge_names):
    """Standalone BFS to find directionally coherent nodes from a root."""
    if root_node not in G: 
        return None
        
    coherent = {root_node}
    queue = [root_node]
    
    while queue:
        cur = queue.pop(0)
        cur_mz = G.nodes[cur]['mz']
        
        for nxt in G.neighbors(cur):
            if nxt in coherent: 
                continue
                
            edge_data = G.edges[cur, nxt]
            etype = edge_data.get('type', 0)
            ename = edge_names.get(etype, '')
            
            # Deduce category from name/type
            cat = 'adduct'
            if 'isotope' in ename or (0 <= etype <= 4): cat = 'isotope'
            elif 'dimer' in ename or (31 <= etype <= 33): cat = 'dimer'
            elif 'loss' in ename: cat = 'loss'
            elif 'mod' in ename: cat = 'mod'
            
            nxt_mz = G.nodes[nxt]['mz']
            parent_is_lighter = cur_mz < nxt_mz
            
            valid = False
            if cat == 'isotope':
                if parent_is_lighter: valid = True
            elif cat == 'dimer':
                if parent_is_lighter: valid = True
                elif 0.40 <= (nxt_mz % 1.0) <= 0.90: valid = True 
            elif cat == 'loss':
                if not parent_is_lighter: valid = True
            elif cat == 'adduct':
                if parent_is_lighter: valid = True
            elif cat == 'mod':
                valid = True # Bidirectional
                
            if valid:
                coherent.add(nxt)
                queue.append(nxt)
                
    return coherent


def save_static(G, subgraph, nodes, pos, low_prob_nodes, node_data, 
                layout, intensities_are_log, title, edge_names,
                use_prob_styling, low_prob_opacity, prob_colorscale,
                highlight_color, normal_color, node_size, edge_width,
                filepath, style='article', xlim=None, 
                root_idx=None, coherent_nodes=None, incoherent_alpha=0.2):
    """Save figure as PDF/PNG/SVG, with support for directional coherence fading."""
    
    cfg = style_config(style)
    fig, ax = plt.subplots(figsize=cfg['figsize'])
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    if layout == 'spatial':
        ax.set_yscale('log')
        ax.set_ylabel('Intensity (log scale)', fontsize=cfg['axis_label_size'])
        ax.set_xlabel('m/z', fontsize=cfg['axis_label_size'])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    for u, v, data in subgraph.edges(data=True):
        color = get_edge_color(data['type'])
        
        is_incoherent = coherent_nodes and (u not in coherent_nodes or v not in coherent_nodes)
        if is_incoherent:
            alpha = incoherent_alpha
        elif use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes):
            alpha = low_prob_opacity
        else:
            alpha = 1.0
            
        if alpha > 0:
            ax.plot(
                [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, lw=max(0.6, edge_width * cfg['edge_scale']),
                alpha=alpha, zorder=1, solid_capstyle='round'
            )
    
    for node, color_val, prob, is_low in node_data:
        x, y = pos[node]
        
        is_incoherent = coherent_nodes and node not in coherent_nodes
        alpha = incoherent_alpha if is_incoherent else 1.0
        
        if alpha <= 0:
            continue
            
        marker_sym = '*' if node == root_idx else 'o'
        n_size = (node_size * cfg['node_scale'] * (1.5 if node == root_idx else 1.0))**2
        
        if isinstance(color_val, float):
            cmap_name = prob_colorscale if prob_colorscale in plt.colormaps() else 'viridis'
            cmap = plt.get_cmap(cmap_name)
            ax.scatter(
                [x], [y], c=[color_val], cmap=cmap,
                vmin=0, vmax=1, s=n_size, marker=marker_sym,
                zorder=3, edgecolors='white', linewidths=0.8, alpha=alpha
            )
        else:
            ax.scatter(
                [x], [y], c=color_val, s=n_size, marker=marker_sym,
                zorder=3, edgecolors='white', linewidths=0.8, alpha=alpha
            )
    
    labels = pick_edge_labels(
        subgraph, pos, layout, intensities_are_log,
        max_labels=cfg['max_edge_labels'], min_sep=cfg['min_sep']
    )
    
    for u, v, data, mx, my in labels:
        if coherent_nodes and (u not in coherent_nodes or v not in coherent_nodes):
            continue
            
        short, _ = get_multicharged_labels(u, v, G, data['type'], edge_names)
        color = get_edge_color(data['type'])
        is_low = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
        
        ax.text(
            mx, my, short, fontsize=cfg['label_size'], 
            ha='center', va='center', color=color, 
            alpha=0.35 if is_low else 0.95, zorder=5,
            bbox=dict(
                boxstyle=f'round,pad={cfg["label_pad"]}',
                facecolor='white', edgecolor=color, 
                linewidth=0.8, alpha=(0.35 if is_low else 0.95) * 0.9
            )
        )
    
    # --- ADD THE MATPLOTLIB COLORBAR ---
    if use_prob_styling:
        cmap_name = prob_colorscale if prob_colorscale in plt.colormaps() else 'viridis'
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=cfg['legend_title_size'])
        cbar.ax.tick_params(labelsize=cfg['legend_size'])
        
    legend_handles = [mpatches.Patch(color=c, label=cat.title()) for cat, c in EDGE_COLORS.items()]
    ax.legend(
        handles=legend_handles, title='Edge type',
        fontsize=cfg['legend_size'], title_fontsize=cfg['legend_title_size'],
        loc='upper left', framealpha=0.9
    )
    
    if layout == 'spatial' and xlim:
        margin = (xlim[1] - xlim[0]) * 0.05
        ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
    
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.spines[['top','right']].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Static figure saved → {filepath}')



def save_static_combined(G, subgraph, nodes, pos, low_prob_nodes, node_data,
                         layout, intensities_are_log, title, edge_names,
                         use_prob_styling, low_prob_opacity, prob_colorscale,
                         highlight_color, normal_color, node_size, edge_width,
                         filepath, profiles, nodes_to_plot, style='article', 
                         xlim=None, highlight_nodes=None,
                         root_idx=None, coherent_nodes=None, incoherent_alpha=0.2):
    """Save a combined publication-ready static figure (Graph + Profiles)."""
    
    cfg = style_config(style)
    fig_width = 10.0 if style == 'article' else cfg['figsize'][0] * 1.5
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_width, cfg['figsize'][1]), 
        gridspec_kw={'width_ratios': [2, 1]}
    )
    
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax1.set_title('Pseudospectrum', fontsize=cfg['title_size'] + 2, fontweight='bold')
    
    if layout == 'spatial':
        ax1.set_yscale('log')
        ax1.set_ylabel('Intensity (log scale)', fontsize=cfg['axis_label_size'])
        ax1.set_xlabel('m/z', fontsize=cfg['axis_label_size'])
    else:
        ax1.set_xticks([])
        ax1.set_yticks([])

    for u, v, data in subgraph.edges(data=True):
        color = get_edge_color(data['type'])
        
        is_incoherent = coherent_nodes and (u not in coherent_nodes or v not in coherent_nodes)
        if is_incoherent:
            alpha = incoherent_alpha
        elif use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes):
            alpha = low_prob_opacity
        else:
            alpha = 1.0
            
        if alpha > 0:
            ax1.plot(
                [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                color=color, lw=max(0.6, edge_width * cfg['edge_scale']), 
                alpha=alpha, zorder=1, solid_capstyle='round'
            )

    for node, color_val, prob, is_low in node_data:
        x, y = pos[node]
        
        is_incoherent = coherent_nodes and node not in coherent_nodes
        alpha = incoherent_alpha if is_incoherent else 1.0
        
        if alpha <= 0:
            continue
            
        marker_sym = '*' if node == root_idx else 'o'
        n_size = (node_size * cfg['node_scale'] * (1.5 if node == root_idx else 1.0))**2
        
        if isinstance(color_val, float):
            cmap_name = prob_colorscale if prob_colorscale in plt.colormaps() else 'viridis'
            cmap = plt.get_cmap(cmap_name)
            ax1.scatter(
                [x], [y], c=[color_val], cmap=cmap, vmin=0, vmax=1,
                s=n_size, marker=marker_sym, zorder=3, 
                edgecolors='white', linewidths=0.8, alpha=alpha
            )
        else:
            ax1.scatter(
                [x], [y], c=color_val, s=n_size, marker=marker_sym, 
                zorder=3, edgecolors='white', linewidths=0.8, alpha=alpha
            )

        if alpha > 0.5:
            ax1.text(
                x, y, str(node), fontsize=cfg['label_size'],
                ha='center', va='center', color='black', fontweight='bold', 
                zorder=4, path_effects=[pe.withStroke(linewidth=2, foreground='white')]
            )

    labels = pick_edge_labels(
        subgraph, pos, layout, intensities_are_log, 
        max_labels=cfg['max_edge_labels'], min_sep=cfg['min_sep']
    )
    
    for u, v, data, mx, my in labels:
        if coherent_nodes and (u not in coherent_nodes or v not in coherent_nodes):
            continue
            
        short, _ = get_multicharged_labels(u, v, G, data['type'], edge_names)
        color = get_edge_color(data['type'])
        is_low = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
        
        ax1.text(
            mx, my, short, fontsize=cfg['label_size'], 
            ha='center', va='center', color=color,
            alpha=0.35 if is_low else 0.95, zorder=5, 
            bbox=dict(
                boxstyle=f'round,pad={cfg["label_pad"]}', 
                facecolor='white', edgecolor=color, 
                linewidth=0.8, alpha=(0.35 if is_low else 0.95) * 0.9
            )
        )

    # --- ADD THE MATPLOTLIB COLORBAR TO AX1 ---
    if use_prob_styling:
        cmap_name = prob_colorscale if prob_colorscale in plt.colormaps() else 'viridis'
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=cfg['legend_title_size'])
        cbar.ax.tick_params(labelsize=cfg['legend_size'])

    if layout == 'spatial' and xlim:
        margin = (xlim[1] - xlim[0]) * 0.05
        ax1.set_xlim(xlim[0] - margin, xlim[1] + margin)
        
    ax1.grid(True, lw=0.3, alpha=0.4)
    ax1.spines[['top','right']].set_visible(False)

    ax2.set_facecolor('white')
    ax2.set_title('Elution Profiles', fontsize=cfg['title_size'] + 2, fontweight='bold')
    plotted_any = False
    profile_colors = plt.get_cmap('tab10')
    
    for i, node in enumerate(nodes_to_plot):
        if node in profiles and (not coherent_nodes or node in coherent_nodes):
            prof = profiles[node]
            ls = '--' if prof.get('is_low', False) else '-'
            ax2.plot(
                prof['rt'], prof['intensity_raw'], 
                color=profile_colors(i % 10), linestyle=ls, 
                linewidth=max(1.5, 2.5 * cfg['edge_scale']), label=f'Node {node}'
            )
            plotted_any = True

    ax2.set_xlabel('Retention Time (min)', fontsize=cfg['axis_label_size'])
    ax2.set_ylabel('Raw Intensity', fontsize=cfg['axis_label_size'])
    ax2.tick_params(axis='both', which='major', labelsize=cfg['tick_size'])
    ax2.grid(True, lw=0.3, alpha=0.4)
    ax2.spines[['top', 'right']].set_visible(False)
    
    if plotted_any and len(nodes_to_plot) <= 15:
        ax2.legend(fontsize=cfg['legend_size'], framealpha=0.9, loc='best')

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Combined static figure saved → {filepath}')


def visualize_component(G, nodes, title='Component Visualization', layout='spatial', 
                       highlight_nodes=None, show_edge_labels=True, save_html=None, 
                       static_filepath=None, static_combined_filepath=None, 
                       node_probs=None, graph_name=None, ionization_mode='positive', 
                       use_prob_styling=False, prob_threshold=0.5, low_prob_opacity=0.2, 
                       normal_color='#484C48', highlight_color='#800080', node_size=24, 
                       edge_width=2, show_legend=True, intensities_are_log=False, 
                       prob_colorscale='RdYlGn', show_colorbar=True, show=True, 
                       style='article', xlim=None, node_features=None, show_profiles=False, 
                       annotation_results=None, root_node=None, fade_incoherent=True,
                       graph=None, remove_low_prob=False):
    """Interactive Plotly visualization with robust toggleable directional fading and profile swapping."""
    import json
    
    # dynamic graph
    if graph is not None:
        if node_features is None and 'node_features' in graph:
            node_features = graph['node_features'].numpy()
            
        if annotation_results is None and 'edge_index' in graph:
            try:
                from .annotate_component import annotate_component
            except ImportError:
                from annotate_component import annotate_component
                
            annotation_results = annotate_component(
                component_gidx_set=set(nodes),
                node_features=graph["node_features"],
                edge_index=graph["edge_index"],
                edge_attr=graph["edge_attr"],
                node_probs=node_probs or {},
                file_name=graph_name,
                roi_values=graph["roi_values"],
                prob_threshold=prob_threshold,
                temperature=3.0,
                ionization_mode=ionization_mode
            )

    edge_names = EDGE_TYPE_NAMES_POSITIVE if ionization_mode == 'positive' else EDGE_TYPE_NAMES_NEGATIVE
    
    if not nodes:
        return None
        
    nodes = list(nodes)

    # remove low probability nodes if requested
    if remove_low_prob and node_probs is not None:
        valid_nodes = []
        for n in nodes:
            roi = G.nodes[n]['roi']
            prob = node_probs.get((graph_name, roi), node_probs.get(n))
            if prob is None or prob >= prob_threshold:
                valid_nodes.append(n)
        nodes = valid_nodes
        
        if not nodes:
            print("Warning: All nodes removed due to probability threshold.")
            return None

    subgraph = G.subgraph(nodes)
    roi_to_node = {G.nodes[n]['roi']: n for n in nodes}
    
    if layout == 'spatial':
        pos = {n: [G.nodes[n]['mz'], max(G.nodes[n]['intensity'], 1e-6)] for n in nodes}
    elif layout == 'spring':
        pos = nx.spring_layout(subgraph, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.circular_layout(subgraph)

    root_idx = None
    coherent_nodes = None
    incoherent_alpha = 0.2 if fade_incoherent else 0.0

    if highlight_nodes and len(highlight_nodes) == 1:
        root_idx = list(highlight_nodes)[0]
    
    if root_node is not None:
        root_idx = root_node
        
    if root_idx is not None and root_idx in G.nodes:
        try:
            coherent_nodes = get_coherent_nodes(G, root_idx, edge_names)
        except NameError:
            pass

    if annotation_results and root_idx is not None:
        for res in annotation_results:
            if res['candidate_gidx'] == root_idx or res['candidate_roi'] == root_idx:
                root_idx = res['candidate_gidx']
                annotated_rois = set(res['node_annotations'].keys())
                coherent_nodes = {roi_to_node[r] for r in annotated_rois if r in roi_to_node}
                break

    node_x, node_y, hover_text = [], [], []
    node_colors, node_data = [], []
    low_prob_nodes = set()
    
    for node in nodes:
        roi = G.nodes[node]['roi']
        mz = G.nodes[node]['mz']
        rt = G.nodes[node]['rt']
        intensity = G.nodes[node]['intensity']
        
        prob = None
        is_low = False
        
        if use_prob_styling and node_probs is not None:
            prob = node_probs.get((graph_name, roi), node_probs.get(node))
            if prob is not None and prob < prob_threshold:
                is_low = True
                low_prob_nodes.add(node)
                
        color_val = float(prob) if prob is not None else normal_color
        node_colors.append(color_val)
        node_data.append((node, color_val, prob, is_low))
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        
        true_intensity = np.expm1(intensity) if intensities_are_log else intensity
        hover_info = f'<b>ROI: {roi}</b><br>Node {node}<br>m/z: {mz:.4f} | RT: {rt:.2f}<br>Intensity: {true_intensity:.2e}<br>'
        if prob is not None: hover_info += f'Probability: {prob:.3f}'
        hover_text.append(hover_info)

    edges_by_category = {}
    mid_x_list, mid_y_list, mid_hover_text = [], [], []
    
    for u, v, data in subgraph.edges(data=True):
        edge_type = data['type']
        edge_name = edge_names.get(edge_type, f'unknown_{edge_type}')
        category = get_edge_category(edge_type)
        
        if category not in edges_by_category:
            edges_by_category[category] = {'normal': {'x': [], 'y': []}, 'shadow': {'x': [], 'y': []}}
            
        is_low_prob = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
        is_incoherent = coherent_nodes and (u not in coherent_nodes or v not in coherent_nodes)
        is_shadow = is_low_prob or is_incoherent
        
        subtype = 'shadow' if is_shadow else 'normal'
        edges_by_category[category][subtype]['x'].extend([pos[u][0], pos[v][0], None])
        edges_by_category[category][subtype]['y'].extend([pos[u][1], pos[v][1], None])
        
        mx, my = get_midpoint(pos, u, v, layout, intensities_are_log)
        mid_x_list.append(mx)
        mid_y_list.append(my)
        
        short, hover_str = get_multicharged_labels(u, v, G, edge_type, edge_names)
        cosine_sim = data.get('cosine_similarity')
        if cosine_sim is not None:
            hover_str += f"<br>Cosine Similarity: {cosine_sim:.4f}"
        if is_shadow: hover_str += "<br><i>(Low Prob/Incoherent)</i>"
        mid_hover_text.append(hover_str)

    edge_traces = []
    for category, subtypes in edges_by_category.items():
        base_color = EDGE_COLORS.get(category, '#888')
        
        # Always append the normal trace
        edge_traces.append(go.Scatter(
            x=subtypes['normal']['x'] if subtypes['normal']['x'] else [], 
            y=subtypes['normal']['y'] if subtypes['normal']['y'] else [], 
            line=dict(width=edge_width, color=base_color), 
            mode='lines', hoverinfo='skip', 
            name=f'{category.title()} edges', opacity=1.0
        ))
        
        # Always append the shadow (fade) trace so JS can find it later
        edge_traces.append(go.Scatter(
            x=subtypes['shadow']['x'] if subtypes['shadow']['x'] else [], 
            y=subtypes['shadow']['y'] if subtypes['shadow']['y'] else [], 
            line=dict(width=edge_width, color=base_color), 
            mode='lines', hoverinfo='skip', 
            name=f'{category.title()} (fade)', opacity=incoherent_alpha, showlegend=False
        ))
    edge_hover_trace = go.Scatter(x=mid_x_list, y=mid_y_list, mode='markers', marker=dict(size=20, color='rgba(0,0,0,0)'), hovertext=mid_hover_text, hovertemplate='%{hovertext}<extra></extra>', showlegend=False)

    node_traces = []
    
    opacities = [incoherent_alpha if (coherent_nodes and n not in coherent_nodes) else 1.0 for n in nodes]
    text_colors = ['rgba(0,0,0,1)' if op > 0 else 'rgba(0,0,0,0)' for op in opacities]
    
    node_symbols = ['star' if n == root_idx else 'circle' for n in nodes]
    node_sizes = [node_size * 1.5 if n == root_idx else node_size for n in nodes]

    marker_dict = dict(
        size=node_sizes, color=node_colors, opacity=opacities, 
        symbol=node_symbols, line=dict(width=1, color='white')
    )
    
    if use_prob_styling and node_probs is not None:
        marker_dict.update(dict(colorscale=prob_colorscale, cmin=0, cmax=1, colorbar=dict(title='Probability', thickness=15, len=0.7, x=1.02) if show_colorbar else None, showscale=show_colorbar))

    node_traces.append(go.Scatter(
        x=node_x, y=node_y, mode='markers+text', 
        text=[str(n) for n in nodes], textposition='middle center', 
        textfont=dict(size=12, color=text_colors, family='Arial Black'), 
        marker=marker_dict, hovertemplate='%{hovertext}<extra></extra>', 
        hovertext=hover_text, customdata=nodes, name='Nodes', uid='main_nodes'
    ))

    build_profiles = show_profiles and node_features is not None
    graph_traces = edge_traces + [edge_hover_trace] + node_traces
    profiles = {}
    initial_nodes = []

    if build_profiles:
        from plotly.subplots import make_subplots
        import matplotlib.colors as mcolors
        
        profiles = compute_node_profiles(node_features, nodes)
        profile_cmap = plt.get_cmap('tab10')
        
        for i, n in enumerate(nodes):
            if n in profiles:
                profiles[n]['is_low'] = node_data[i][3]
                profiles[n]['color'] = mcolors.to_hex(profile_cmap(i % 10))
                
        start_set = set(highlight_nodes or [])
        if root_idx is not None: start_set.add(root_idx)
        initial_nodes = list(dict.fromkeys([n for n in nodes if n in start_set and n in profiles]))

        if static_combined_filepath:
            nodes_to_plot = list(profiles.keys())
            save_static_combined(G, subgraph, nodes, pos, low_prob_nodes, node_data, layout, intensities_are_log, title, edge_names, use_prob_styling, low_prob_opacity, prob_colorscale, highlight_color, normal_color, node_size, edge_width, static_combined_filepath, profiles, nodes_to_plot, style, xlim, highlight_nodes, root_idx, coherent_nodes, incoherent_alpha)

        fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=('Pseudospectrum', 'Elution Profiles'), horizontal_spacing=0.04)
        for trace in graph_traces: fig.add_trace(trace, row=1, col=1)
        for node in initial_nodes:
            line_dash = 'dash' if profiles[node].get('is_low', False) else 'solid'
            fig.add_trace(go.Scatter(x=profiles[node]['rt'], y=profiles[node]['intensity_raw'], mode='lines', name=f'Node {node}', uid=f'profile-{node}', meta=node, line=dict(color=profiles[node]['color'], width=2, dash=line_dash)), row=1, col=2)
            
        fig.update_xaxes(title_text='Retention Time (min)', row=1, col=2)
        fig.update_yaxes(title_text='Raw Intensity', row=1, col=2) 
    else:
        fig = go.Figure(data=graph_traces)
    
    if show_edge_labels:
        for u, v, data in subgraph.edges(data=True):
            if coherent_nodes and (u not in coherent_nodes or v not in coherent_nodes): continue
            short, _ = get_multicharged_labels(u, v, G, data['type'], edge_names)
            mx, my = get_midpoint(pos, u, v, layout, intensities_are_log)
            is_low = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
            fig.add_annotation(x=mx, y=my, text=short, showarrow=False, visible=True, font=dict(size=11, color='black'), bgcolor='rgba(255,255,255,0.85)', opacity=0.4 if is_low else 1.0, bordercolor='gray', borderwidth=0.8, borderpad=2)
    yaxis_cfg = dict(showgrid=True, gridcolor='#E5E5E5', zeroline=False, type='log' if layout=='spatial' else 'linear', showticklabels=layout=='spatial')
    fig.update_layout(uirevision='constant', title=None if build_profiles else title, showlegend=show_legend, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), plot_bgcolor='white', autosize=True, height=900, xaxis=dict(showgrid=True, gridcolor='#E5E5E5', zeroline=True, zerolinecolor='#E5E5E5', showticklabels=layout=='spatial', range=xlim if layout=='spatial' and xlim else None), yaxis=yaxis_cfg)
    if layout == 'spatial': fig.update_layout(xaxis_title='m/z', yaxis_title='Intensity (log scale)')
    
    if save_html:
        js_cands = {}
        top_cands = annotation_results[:5] if annotation_results else []
        for r in top_cands:
            c_node = str(int(r['candidate_gidx']))
            rois = r['node_annotations']
            js_cands[c_node] = {
                'name': str(rois.get(r['candidate_roi'], 'Unknown')),
                'nodes': [str(int(roi_to_node[roi])) for roi in rois if roi in roi_to_node],
                'annots': {str(int(roi_to_node[roi])): str(ann) for roi, ann in rois.items() if roi in roi_to_node}
            }
        
        js_edges = []
        for u, v, d in subgraph.edges(data=True):
            js_edges.append({
                'u': str(int(u)), 
                'v': str(int(v)), 
                'cat': get_edge_category(d['type']).title(),
                'cosine_sim': float(d.get('cosine_similarity', 0)) if d.get('cosine_similarity') is not None else None,
                'x': [float(pos[u][0]), float(pos[v][0]), None], 
                'y': [float(pos[u][1]), float(pos[v][1]), None]
            })

        js_coherent_nodes = [str(n) for n in coherent_nodes] if coherent_nodes else []
        js_low_prob = [str(n) for n in low_prob_nodes] if low_prob_nodes else []

        post_js = f"""
        var allProfiles = {json.dumps({str(k): v for k, v in profiles.items()})};
        var candData = {json.dumps(js_cands)};
        var allEdges = {json.dumps(js_edges)};
        var shownNodes = new Set({json.dumps([str(n) for n in initial_nodes])});
        var staticCoherentNodes = new Set({json.dumps(js_coherent_nodes)});
        var lowProbNodes = new Set({json.dumps(js_low_prob)});
        var fallbackRootId = "{str(root_idx) if root_idx is not None else ''}";
        var isNormalized = false;
        
        setTimeout(function() {{
            var graphDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (!graphDiv || !graphDiv.data) return;

            var controlDiv = document.createElement('div');
            controlDiv.style.cssText = 'position:absolute;left:10px;top:10px;z-index:1000;background:white;padding:10px;border:1px solid #ccc;border-radius:4px;box-shadow:0 2px 4px rgba(0,0,0,0.1);max-width:240px;';
            
            var toggleBtn = document.createElement('button');
            toggleBtn.innerHTML = '−'; 
            toggleBtn.style.cssText = 'width:24px;height:24px;padding:0;margin-bottom:5px;cursor:pointer;border:1px solid #999;border-radius:3px;background:#f0f0f0;';
            
            var controlContent = document.createElement('div');
            
            if (Object.keys(candData).length > 0) {{
                controlContent.innerHTML += '<label style="font-size:12px;display:block;margin-bottom:3px;font-weight:bold;">Filter by Annotation Root:</label>';
                var candSelect = document.createElement('select');
                candSelect.id = 'candSelect';
                candSelect.style.cssText = 'font-size:11px;width:100%;margin-bottom:10px;';
                candSelect.innerHTML = '<option value="">-- Show All --</option>';
                Object.keys(candData).forEach(k => {{ candSelect.innerHTML += `<option value="${{k}}">Node ${{k}} (${{candData[k].name}})</option>`; }});
                controlContent.appendChild(candSelect);
            }}

            controlContent.innerHTML += '<div style="margin-bottom:10px;"><input type="checkbox" id="incoherentToggle" ' + ({incoherent_alpha} > 0 ? 'checked' : '') + '><label for="incoherentToggle" style="font-size:12px;margin-left:5px;">Show All Nodes</label></div>';

            controlContent.innerHTML += '<label style="font-size:12px;display:block;margin-bottom:3px;">Profiles Intensity:</label>';
            var intSelect = document.createElement('select');
            intSelect.style.cssText = 'font-size:11px;width:100%;margin-bottom:8px;';
            intSelect.innerHTML = '<option value="raw">Raw</option><option value="norm">Normalized</option>';
            controlContent.appendChild(intSelect);

            controlDiv.appendChild(toggleBtn); controlDiv.appendChild(controlContent);
            graphDiv.parentNode.insertBefore(controlDiv, graphDiv);
            
            toggleBtn.onclick = function() {{
                var h = controlContent.style.display === 'none';
                controlContent.style.display = h ? 'block' : 'none';
                toggleBtn.innerHTML = h ? '−' : '+';
            }};

            function updateGraphState() {{
                var candEl = document.getElementById('candSelect');
                var rootId = candEl && candEl.value ? String(candEl.value) : fallbackRootId;
                var incToggle = document.getElementById('incoherentToggle');
                var showIncoherent = incToggle ? incToggle.checked : false;
                
                var nodeTraceIdx = graphDiv.data.findIndex(t => t.name === 'Nodes');
                if (nodeTraceIdx === -1) return;
                
                var trace = graphDiv.data[nodeTraceIdx];
                if (!trace.customdata) return; 
                
                if (!trace._origX) trace._origX = trace.x.slice();
                if (!trace._origY) trace._origY = trace.y.slice();
                
                var cNodes = null;
                var cDataAnnots = null;
                
                if (rootId) {{
                    if (candData[rootId]) {{
                        cNodes = new Set(candData[rootId].nodes.map(String));
                        cDataAnnots = candData[rootId].annots;
                    }} else if (rootId === fallbackRootId && staticCoherentNodes.size > 0) {{
                        cNodes = staticCoherentNodes;
                    }} else {{
                        cNodes = new Set([rootId]); 
                    }}
                }}
                
                var newX = [], newY = [], newOpacities = [], newSymbols = [], newSizes = [], newTextColors = [], newHover = [];
                
                for(var i=0; i < trace.customdata.length; i++) {{
                    var n = String(trace.customdata[i]);
                    if (Array.isArray(trace.customdata[i])) n = String(trace.customdata[i][0]);
                    
                    var isRoot = (n === rootId);
                    newSymbols.push(isRoot ? 'star' : 'circle');
                    newSizes.push(isRoot ? {node_size * 1.5} : {node_size});
                    
                    var hoverStr = trace.hovertext && trace.hovertext[i] ? String(trace.hovertext[i]) : '';
                    var baseHover = hoverStr.split('<br><b>Annotation:')[0];
                    var isLow = lowProbNodes.has(n);
                    
                    if (!cNodes) {{ 
                        newX.push(trace._origX[i]);
                        newY.push(trace._origY[i]);
                        newOpacities.push(isLow ? 0.35 : 1.0); 
                        newTextColors.push('rgba(0,0,0,1)');
                        newHover.push(baseHover);
                    }} else {{
                        if (cNodes.has(n)) {{
                            newX.push(trace._origX[i]);
                            newY.push(trace._origY[i]);
                            newOpacities.push(isLow ? 0.35 : 1.0);
                            newTextColors.push('rgba(0,0,0,1)');
                            var ann = cDataAnnots && cDataAnnots[n] ? cDataAnnots[n] : null;
                            newHover.push(ann ? baseHover + '<br><b>Annotation: ' + ann + '</b>' : baseHover);
                        }} else {{
                            if (showIncoherent) {{
                                newX.push(trace._origX[i]);
                                newY.push(trace._origY[i]);
                                // ONLY alter opacity. Do not push new colors. 
                                // This preserves the WebGL colorscale so the node stays the correct mapped color.
                                newOpacities.push(0.15); 
                                newTextColors.push('rgba(0,0,0,0)');
                                newHover.push(baseHover);
                            }} else {{
                                newX.push(null);
                                newY.push(null);
                                newOpacities.push(0);
                                newTextColors.push('rgba(0,0,0,0)');
                                newHover.push('');
                            }}
                        }}
                    }}
                }}
                
                var updates = {{
                    'x': [newX],
                    'y': [newY],
                    'marker.opacity': [newOpacities], 
                    'marker.symbol': [newSymbols], 
                    'marker.size': [newSizes], 
                    'textfont.color': [newTextColors], 
                    'hovertext': [newHover]
                }};
                
                var edgeUpdates = {{}};
                allEdges.forEach(edge => {{
                    if (!edgeUpdates[edge.cat]) edgeUpdates[edge.cat] = {{ activeX: [], activeY: [], shadowX: [], shadowY: [] }};
                    
                    var isLowProbEdge = lowProbNodes.has(String(edge.u)) || lowProbNodes.has(String(edge.v));
                    var isCoherentEdge = !cNodes || (cNodes.has(String(edge.u)) && cNodes.has(String(edge.v)));
                    
                    if (isCoherentEdge && !isLowProbEdge) {{
                        edgeUpdates[edge.cat].activeX.push(...edge.x);
                        edgeUpdates[edge.cat].activeY.push(...edge.y);
                    }} else {{
                        var shouldShadow = false;
                        if (isCoherentEdge && isLowProbEdge) shouldShadow = true; 
                        if (!isCoherentEdge && showIncoherent) shouldShadow = true; 
                        
                        if (shouldShadow) {{
                            edgeUpdates[edge.cat].shadowX.push(...edge.x);
                            edgeUpdates[edge.cat].shadowY.push(...edge.y);
                        }}
                    }}
                }});
                
                var traceIndices = [];
                var traceUpdatesX = [];
                var traceUpdatesY = [];
                var traceUpdatesOp = [];
                
                for(var i=0; i<graphDiv.data.length; i++) {{
                    var tr = graphDiv.data[i];
                    if (!tr.name) continue;
                    
                    if (tr.name.includes(' edges')) {{
                        var cat = tr.name.split(' ')[0];
                        if (edgeUpdates[cat]) {{
                            traceIndices.push(i);
                            traceUpdatesX.push(edgeUpdates[cat].activeX);
                            traceUpdatesY.push(edgeUpdates[cat].activeY);
                            traceUpdatesOp.push(1.0);
                        }}
                    }} else if (tr.name.includes(' (fade)')) {{
                        var cat = tr.name.split(' ')[0];
                        if (edgeUpdates[cat]) {{
                            traceIndices.push(i);
                            traceUpdatesX.push(edgeUpdates[cat].shadowX);
                            traceUpdatesY.push(edgeUpdates[cat].shadowY);
                            traceUpdatesOp.push(0.25); 
                        }}
                    }}
                }}
                
                if (traceIndices.length > 0) {{
                    Plotly.restyle(graphDiv, {{x: traceUpdatesX, y: traceUpdatesY, opacity: traceUpdatesOp}}, traceIndices);
                }}
                Plotly.restyle(graphDiv, updates, [nodeTraceIdx]);
            }}

            if(document.getElementById('candSelect')) {{
                document.getElementById('candSelect').addEventListener('change', function(e) {{
                    var rootId = e.target.value;
                    updateGraphState();
                    
                    if(rootId && allProfiles[rootId]) {{
                        var toDelete = [];
                        for(var j=0; j < graphDiv.data.length; j++) {{
                            if(graphDiv.data[j].xaxis === 'x2') toDelete.push(j);
                        }}
                        if(toDelete.length > 0) Plotly.deleteTraces(graphDiv, toDelete);
                        shownNodes.clear();
                        
                        var pr = allProfiles[rootId];
                        var yData = isNormalized ? pr.intensity_norm : pr.intensity_raw;
                        Plotly.addTraces(graphDiv, {{
                            x: pr.rt, y: yData, mode: 'lines', name: 'Node '+rootId, 
                            xaxis: 'x2', yaxis: 'y2', 
                            line: {{color: pr.color, width: 2, dash: pr.is_low ? 'dash' : 'solid'}}
                        }});
                        shownNodes.add(rootId);
                    }}
                }});
            }}

            var incToggle = document.getElementById('incoherentToggle');
            if (incToggle) incToggle.addEventListener('change', updateGraphState);

            var intSelectEl = document.querySelector('select:last-of-type');
            if (intSelectEl) {{
                intSelectEl.addEventListener('change', function(e) {{
                    isNormalized = (e.target.value === 'norm');
                    var uIdx=[], uY=[];
                    graphDiv.data.forEach((t, i) => {{
                        if (t.xaxis === 'x2') {{
                            var nId = t.name.replace('Node ', '');
                            if (allProfiles[nId]) {{
                                uIdx.push(i);
                                uY.push(isNormalized ? allProfiles[nId].intensity_norm : allProfiles[nId].intensity_raw);
                            }}
                        }}
                    }});
                    if (uIdx.length) Plotly.restyle(graphDiv, {{y: uY}}, uIdx);
                    Plotly.relayout(graphDiv, {{'yaxis2.title.text': isNormalized ? 'Normalized Intensity' : 'Raw Intensity'}});
                }});
            }}

            graphDiv.on('plotly_click', function(evt) {{
                var pt = evt.points[0];
                if (pt.curveNumber === undefined || pt.customdata === undefined) return;
                
                var nId = String(pt.customdata);
                if (Array.isArray(pt.customdata)) nId = String(pt.customdata[0]);
                
                if (!(nId in allProfiles)) return;
                
                if (shownNodes.has(nId)) {{
                    var idx = graphDiv.data.findIndex(t => t.name === 'Node ' + nId && t.xaxis === 'x2');
                    if (idx >= 0) {{ Plotly.deleteTraces(graphDiv, [idx]); shownNodes.delete(nId); }}
                }} else {{
                    var pr = allProfiles[nId];
                    Plotly.addTraces(graphDiv, {{
                        x: pr.rt, y: isNormalized ? pr.intensity_norm : pr.intensity_raw, 
                        mode: 'lines', name: 'Node '+nId, xaxis: 'x2', yaxis: 'y2', 
                        line: {{color: pr.color, width: 2, dash: pr.is_low ? 'dash' : 'solid'}}
                    }});
                    shownNodes.add(nId);
                }}
            }});
            
            updateGraphState();
        }}, 500);
        """
        fig.write_html(save_html, post_script=post_js)
        print(f'Interactive HTML saved → {save_html}')
        
    if static_filepath: save_static(G, subgraph, nodes, pos, low_prob_nodes, node_data, layout, intensities_are_log, title, edge_names, use_prob_styling, low_prob_opacity, prob_colorscale, highlight_color, normal_color, node_size, edge_width, static_filepath, style, xlim, root_idx, coherent_nodes, incoherent_alpha)
    if show: fig.show()
    return fig


if __name__ == "__main__":
    print("Config adduct diffs:", EDGE_TYPE_NAMES_POSITIVE)
    print('\n \n \n')
    print("Config adduct diffs (negative):", EDGE_TYPE_NAMES_NEGATIVE)