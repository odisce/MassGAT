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
def save_static(G, subgraph, nodes, pos, low_prob_nodes, node_data, 
                layout, intensities_are_log, title, edge_names,
                use_prob_styling, low_prob_opacity, prob_colorscale,
                highlight_color, normal_color, node_size, edge_width,
                filepath, style='article', xlim=None):
    """Save figure as PDF/PNG/SVG."""
    
    cfg = style_config(style)
    fig, ax = plt.subplots(figsize=cfg['figsize'])
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Axis setup
    if layout == 'spatial':
        ax.set_yscale('log')
        ax.set_ylabel('Intensity (log scale)', fontsize=cfg['axis_label_size'])
        ax.set_xlabel('m/z', fontsize=cfg['axis_label_size'])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    # Draw edges
    for u, v, data in subgraph.edges(data=True):
        color = get_edge_color(data['type'])
        alpha = low_prob_opacity if (use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)) else 1.0
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, lw=max(0.6, edge_width * cfg['edge_scale']),
                alpha=alpha, zorder=1, solid_capstyle='round')
    
    # Draw nodes
    grad_x, grad_y, grad_vals = [], [], []
    fixed_colors = {}
    
    for node, color_val, prob, is_low in node_data:
        x, y = pos[node]
        if isinstance(color_val, float):
            grad_x.append(x); grad_y.append(y); grad_vals.append(color_val)
        else:
            fixed_colors.setdefault(color_val, ([], []))[0].append(x)
            fixed_colors.setdefault(color_val, ([], []))[1].append(y)
    
    # Gradient nodes
    if grad_x:
        cmap = {'RdYlGn': 'RdYlGn', 'Viridis': 'viridis'}.get(prob_colorscale, 'RdYlGn')
        sc = ax.scatter(grad_x, grad_y, c=grad_vals, cmap=cmap,
                       vmin=0, vmax=1, s=(node_size * cfg['node_scale'])**2,
                       zorder=3, edgecolors='white', linewidths=0.8)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.015, pad=0.01, shrink=0.6)
        cbar.set_label('Probability', fontsize=max(7, cfg['legend_size']))
    
    # Fixed color nodes
    for color, (xs, ys) in fixed_colors.items():
        ax.scatter(xs, ys, c=color, s=(node_size * cfg['node_scale'])**2,
                  zorder=3, edgecolors='white', linewidths=0.8)
    
    # Edge labels
    labels = pick_edge_labels(subgraph, pos, layout, intensities_are_log,
                             max_labels=cfg['max_edge_labels'], min_sep=cfg['min_sep'])
    
    for u, v, data, mx, my in labels:
        edge_name = edge_names.get(data['type'], f'unknown_{data["type"]}')
        short = short_label(edge_name)
        color = get_edge_color(data['type'])
        is_low = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
        
        ax.text(mx, my, short, fontsize=cfg['label_size'], ha='center', va='center',
                color=color, alpha=0.35 if is_low else 0.95, zorder=5,
                bbox=dict(boxstyle=f'round,pad={cfg["label_pad"]}',
                         facecolor='white', edgecolor=color, linewidth=0.8,
                         alpha=(0.35 if is_low else 0.95) * 0.9))
    
    # Legend
    legend_handles = [mpatches.Patch(color=c, label=cat.title())
                     for cat, c in EDGE_COLORS.items()]
    ax.legend(handles=legend_handles, title='Edge type',
             fontsize=cfg['legend_size'], title_fontsize=cfg['legend_title_size'],
             loc='upper left', framealpha=0.9)
    
    # Axis limits
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
                         xlim=None, highlight_nodes=None):
    """Save a combined publication-ready static figure (Graph + Profiles)."""
    cfg = style_config(style)
    # Widen the figure to accommodate the side-by-side layout
    fig_width = 10.0 if style == 'article' else cfg['figsize'][0] * 1.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, cfg['figsize'][1]), gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor('white')

    # --- LEFT PANEL: NETWORK GRAPH ---
    ax1.set_facecolor('white')
    ax1.set_title('Pseudospectrum', fontsize=cfg['title_size'] + 2, fontweight='bold')
    
    if layout == 'spatial':
        ax1.set_yscale('log')
        ax1.set_ylabel('Intensity (log scale)', fontsize=cfg['axis_label_size'])
        ax1.set_xlabel('m/z', fontsize=cfg['axis_label_size'])
    else:
        ax1.set_xticks([]); ax1.set_yticks([])

    # Edges
    for u, v, data in subgraph.edges(data=True):
        color = get_edge_color(data['type'])
        alpha = low_prob_opacity if (use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)) else 1.0
        ax1.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color,
                 lw=max(0.6, edge_width * cfg['edge_scale']), alpha=alpha, zorder=1, solid_capstyle='round')

    # Nodes
    grad_x, grad_y, grad_vals, fixed_colors = [], [], [], {}
    for node, color_val, prob, is_low in node_data:
        x, y = pos[node]
        if isinstance(color_val, float):
            grad_x.append(x); grad_y.append(y); grad_vals.append(color_val)
        else:
            fixed_colors.setdefault(color_val, ([], []))[0].append(x)
            fixed_colors.setdefault(color_val, ([], []))[1].append(y)

    if grad_x:
        try: 
            cmap = plt.get_cmap(prob_colorscale)
        except ValueError: 
            cmap = plt.get_cmap('viridis')
        sc = ax1.scatter(grad_x, grad_y, c=grad_vals, cmap=cmap, vmin=0, vmax=1,
                         s=(node_size * cfg['node_scale'])**2, zorder=3, edgecolors='white', linewidths=0.8)
        cbar = plt.colorbar(sc, ax=ax1, fraction=0.015, pad=0.01, shrink=0.6)
        cbar.set_label('Probability', fontsize=max(7, cfg['legend_size']))


    for color, (xs, ys) in fixed_colors.items():
        ax1.scatter(xs, ys, c=color, s=(node_size * cfg['node_scale'])**2, zorder=3, edgecolors='white', linewidths=0.8)

    # Node IDs with White Halo for perfect legibility
    for n, pos_xy in pos.items():
        ax1.text(pos_xy[0], pos_xy[1], str(n), fontsize=cfg['label_size'],
                 ha='center', va='center', color='black', fontweight='bold', zorder=4,
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Edge Labels
    labels = pick_edge_labels(subgraph, pos, layout, intensities_are_log, max_labels=cfg['max_edge_labels'], min_sep=cfg['min_sep'])
    for u, v, data, mx, my in labels:
        short = short_label(edge_names.get(data['type'], f'unknown_{data["type"]}'))
        color = get_edge_color(data['type'])
        is_low = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
        ax1.text(mx, my, short, fontsize=cfg['label_size'], ha='center', va='center', color=color,
                 alpha=0.35 if is_low else 0.95, zorder=5, bbox=dict(boxstyle=f'round,pad={cfg["label_pad"]}', 
                 facecolor='white', edgecolor=color, linewidth=0.8, alpha=(0.35 if is_low else 0.95) * 0.9))

    if layout == 'spatial' and xlim:
        margin = (xlim[1] - xlim[0]) * 0.05
        ax1.set_xlim(xlim[0] - margin, xlim[1] + margin)
    ax1.grid(True, lw=0.3, alpha=0.4); ax1.spines[['top','right']].set_visible(False)

    # --- RIGHT PANEL: PROFILES ---
    ax2.set_facecolor('white')
    ax2.set_title('Elution Profiles', fontsize=cfg['title_size'] + 2, fontweight='bold')
    plotted_any = False
    
    # NEW: Use a high-contrast categorical palette specifically for the lines.
    # 'tab10' provides dark, distinct colors perfect for white backgrounds.
    profile_colors = plt.get_cmap('tab10')
    
    for i, node in enumerate(nodes_to_plot):
        if node in profiles:
            prof = profiles[node]
            ls = '--' if prof.get('is_low', False) else '-'
            
            # Assign color from the categorical palette based on index
            line_color = profile_colors(i % 10)
            
            # Increased minimum linewidth to 1.5 to make the lines punchier
            ax2.plot(prof['rt'], prof['intensity_raw'], color=line_color,
                     linestyle=ls, linewidth=max(1.5, 2.5 * cfg['edge_scale']), label=f'Node {node}')
            plotted_any = True

    ax2.set_xlabel('Retention Time (min)', fontsize=cfg['axis_label_size'])
    ax2.set_ylabel('Raw Intensity', fontsize=cfg['axis_label_size'])
    ax2.tick_params(axis='both', which='major', labelsize=cfg['tick_size'])
    ax2.grid(True, lw=0.3, alpha=0.4); ax2.spines[['top', 'right']].set_visible(False)
    
    # Show legend if it won't clutter the graph too much
    if plotted_any and len(nodes_to_plot) <= 15:
        ax2.legend(fontsize=cfg['legend_size'], framealpha=0.9, loc='best')

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Combined static figure saved → {filepath}')

def visualize_component(G, nodes, title='Component Visualization',
                       layout='spatial', highlight_nodes=None,
                       show_edge_labels=True, save_html=None, static_filepath=None,
                       static_combined_filepath=None, 
                       node_probs=None, graph_name=None, ionization_mode='positive',
                       use_prob_styling=False, prob_threshold=0.5,
                       low_prob_opacity=0.2, normal_color='#484C48',
                       highlight_color='#800080', node_size=24, edge_width=2,
                       show_legend=True, intensities_are_log=False,
                       prob_colorscale='RdYlGn', show_colorbar=True,
                       show=True, style='article', xlim=None,
                       node_features=None, show_profiles=False):
    """Interactive Plotly visualization of a spectral network component.

    node_features, show_profiles: when show_profiles=True and node_features
        is provided, the saved HTML gets a second panel on the right showing
        RT/intensity profiles. Profiles for nodes in `highlight_nodes` are
        shown by default; clicking any node in the graph toggles its profile
        on/off. node_features should be indexable by the graph-level node
        ids in `nodes` (i.e. graph['node_features'], not pre-sliced to the
        component). Has no effect on the static export.
    """
    
    edge_names = EDGE_TYPE_NAMES_POSITIVE if ionization_mode == 'positive' else EDGE_TYPE_NAMES_NEGATIVE
    
    if not nodes:
        print('No nodes to visualize')
        return None
    
    nodes = list(nodes)
    subgraph = G.subgraph(nodes)
    
    # Calculate positions
    if layout == 'spatial':
        pos = {n: [G.nodes[n]['mz'], max(G.nodes[n]['intensity'], 1e-6)]
               for n in nodes}
    elif layout == 'spring':
        pos = nx.spring_layout(subgraph, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.circular_layout(subgraph)
    
    # Prepare node data
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    manual_highlights = set(highlight_nodes) if highlight_nodes else set()
    low_prob_nodes = set()
    
    hover_text = []
    node_colors = []
    node_data = []  # (node, color_val, prob, is_low)
    
    for node in nodes:
        roi = G.nodes[node]['roi']
        mz = G.nodes[node]['mz']
        rt = G.nodes[node]['rt']
        intensity = G.nodes[node]['intensity']
        degree = G.degree[node]
        true_intensity = np.expm1(intensity) if intensities_are_log else intensity
        
        # Determine color
        prob = None
        is_low = False
        
        if use_prob_styling and node_probs is not None:
            prob = node_probs.get((graph_name, roi), node_probs.get(node))
            if prob is None:
                prob = 0.0
            is_low = prob < prob_threshold
        

        if use_prob_styling and node_probs is not None and prob is not None:
            color_val = float(prob)
        else:
            color_val = normal_color
        
        if is_low:
            low_prob_nodes.add(node)
        
        node_colors.append(color_val)
        node_data.append((node, color_val, prob, is_low))
        
        # Hover text
        hover_info = f'<b>ROI: {roi}</b><br>Node {node}<br>'
        hover_info += f'm/z: {mz:.4f} | RT: {rt:.2f}<br>'
        hover_info += f'Intensity: {true_intensity:.2e}<br>Degree: {degree}<br>'
        if prob is not None:
            hover_info += f'Probability: {prob:.3f}<br>'
        hover_text.append(hover_info)
    
    # BUILD EDGE TRACES
    # Group edges by category
    edges_by_category = {}
    mid_x_list = []
    mid_y_list = []
    mid_hover_text = []
    
    for u, v, data in subgraph.edges(data=True):
        edge_type = data['type']
        edge_name = edge_names.get(edge_type, f'unknown_{edge_type}')
        category = get_edge_category(edge_type)
        
        # Initialize bucket for this category
        if category not in edges_by_category:
            edges_by_category[category] = {'normal': {'x': [], 'y': []}, 'shadow': {'x': [], 'y': []}}
        
        # Check if this is a shadow edge (low probability)
        is_shadow = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
        subtype = 'shadow' if is_shadow else 'normal'
        
        # Add edge coordinates
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edges_by_category[category][subtype]['x'].extend([x0, x1, None])
        edges_by_category[category][subtype]['y'].extend([y0, y1, None])
        
        # Store midpoint for hover
        mx, my = get_midpoint(pos, u, v, layout, intensities_are_log)
        mid_x_list.append(mx)
        mid_y_list.append(my)
        
        # Build hover text (matching original exactly)
        hover_str = f"{edge_name} (type {edge_type})"
        if is_shadow:
            hover_str += "<br><i>(Low Probability Link)</i>"
        mid_hover_text.append(hover_str)
    
    # Create edge traces (one per category)
    edge_traces = []
    for category, subtypes in edges_by_category.items():
        base_color = EDGE_COLORS.get(category, '#888')
        
        # Normal edges
        if subtypes['normal']['x']:
            edge_traces.append(go.Scatter(
                x=subtypes['normal']['x'], 
                y=subtypes['normal']['y'],
                line=dict(width=edge_width, color=base_color),
                mode='lines', 
                hoverinfo='skip',
                name=f'{category.title()} edges', 
                opacity=1.0
            ))
        
        # Shadow edges (low probability)
        if use_prob_styling and subtypes['shadow']['x']:
            edge_traces.append(go.Scatter(
                x=subtypes['shadow']['x'], 
                y=subtypes['shadow']['y'],
                line=dict(width=edge_width, color=base_color),
                mode='lines', 
                hoverinfo='skip',
                name=f'{category.title()} (low prob)',
                opacity=low_prob_opacity, 
                showlegend=False
            ))
    
    # EDGE HOVER TRACE 
    edge_hover_trace = go.Scatter(
        x=mid_x_list, 
        y=mid_y_list,
        mode='markers',
        marker=dict(size=20, color='rgba(0,0,0,0)'), 
        hovertext=mid_hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        showlegend=False, 
        name='Edge Info'
    )
    
    # BUILD NODE TRACES
    node_traces = []
    text_cfg = dict(size=12, color='black', family='Arial Black') # Bold, highly visible text
    
    if use_prob_styling and node_probs is not None:
        grad_idx = [i for i, n in enumerate(nodes) if n not in manual_highlights]
        if grad_idx:
            node_traces.append(go.Scatter(
                x=[node_x[i] for i in grad_idx], y=[node_y[i] for i in grad_idx],
                mode='markers+text',
                text=[str(nodes[i]) for i in grad_idx],
                textposition='middle center', textfont=text_cfg,
                marker=dict(
                    size=node_size, color=[node_colors[i] for i in grad_idx],
                    colorscale=prob_colorscale, cmin=0, cmax=1,
                    colorbar=dict(title='Probability', thickness=15, len=0.7, x=1.02) if show_colorbar else None,
                    showscale=show_colorbar, line=dict(width=1, color='white')
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[hover_text[i] for i in grad_idx],
                customdata=[nodes[i] for i in grad_idx], name='Nodes'
            ))
        
        hi_idx = [i for i, n in enumerate(nodes) if n in manual_highlights]
        if hi_idx:
            node_traces.append(go.Scatter(
                x=[node_x[i] for i in hi_idx], y=[node_y[i] for i in hi_idx],
                mode='markers+text',
                text=[str(nodes[i]) for i in hi_idx],
                textposition='middle center', textfont=text_cfg,
                marker=dict(
                    size=node_size * 1.2, # Make it slightly larger
                    color=[node_colors[i] for i in hi_idx],
                    colorscale=prob_colorscale, cmin=0, cmax=1,
                    line=dict(width=4, color='#FF00FF') # Distinct Magenta Halo
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[hover_text[i] for i in hi_idx],
                customdata=[nodes[i] for i in hi_idx], name='Target'
            ))
    else:
        node_traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(n) for n in nodes],
            textposition='middle center', textfont=text_cfg,
            marker=dict(size=node_size, color=node_colors, line=dict(width=1, color='white')),
            hovertemplate='%{hovertext}<extra></extra>', hovertext=hover_text,
            customdata=list(nodes), name='Nodes'
        ))
    
    # Build edge_traces, then edge_hover, then node_traces
    build_profiles = show_profiles and node_features is not None
    graph_traces = edge_traces + [edge_hover_trace] + node_traces

    if build_profiles:
        from plotly.subplots import make_subplots
        import matplotlib.colors as mcolors

        profiles = compute_node_profiles(node_features, nodes)
        profile_cmap = plt.get_cmap('tab10')
        
        for i, n in enumerate(nodes):
            if n in profiles:
                profiles[n]['is_low'] = node_data[i][3]
                profiles[n]['color'] = mcolors.to_hex(profile_cmap(i % 10))
        
        initial_nodes = [n for n in nodes if n in manual_highlights and n in profiles]


        if static_combined_filepath:
            nodes_to_plot = list(profiles.keys())
            save_static_combined(G, subgraph, nodes, pos, low_prob_nodes, node_data,
                                 layout, intensities_are_log, title, edge_names,
                                 use_prob_styling, low_prob_opacity, prob_colorscale,
                                 highlight_color, normal_color, node_size, edge_width,
                                 static_combined_filepath, profiles, nodes_to_plot, 
                                 style, xlim, highlight_nodes=manual_highlights)

        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.65, 0.35],
            subplot_titles=('Pseudospectrum', 'Elution Profiles'),
            horizontal_spacing=0.04
        )
        for trace in graph_traces:
            fig.add_trace(trace, row=1, col=1)

        initial_nodes = [n for n in nodes if n in manual_highlights and n in profiles]
        profile_trace_ids = {}
        for node in initial_nodes:
            uid = f'profile-{node}'
            line_dash = 'dash' if profiles[node].get('is_low', False) else 'solid'
            
            fig.add_trace(go.Scatter(
                x=profiles[node]['rt'], 
                y=profiles[node]['intensity_raw'],
                mode='lines', name=f'Node {node}', uid=uid, meta=node,
                line=dict(color=profiles[node]['color'], width=2, dash=line_dash) 
            ), row=1, col=2)
            profile_trace_ids[node] = uid

        fig.update_xaxes(title_text='Retention Time (min)', row=1, col=2)
        fig.update_yaxes(title_text='Raw Intensity', row=1, col=2) 
    else:
        fig = go.Figure(data=graph_traces)
    
    # Edge labels
    if show_edge_labels:
        for u, v, data in subgraph.edges(data=True):
            edge_type = data['type']
            edge_name = edge_names.get(edge_type, f'unknown_{edge_type}')
            short = short_label(edge_name)
            mx, my = get_midpoint(pos, u, v, layout, intensities_are_log)
            is_low = use_prob_styling and (u in low_prob_nodes or v in low_prob_nodes)
            
            fig.add_annotation(
                x=mx, y=my, text=short, showarrow=False,
                visible=show_edge_labels, # Link to python state
                font=dict(size=11, color='black'), # Increased size
                bgcolor='rgba(255,255,255,0.85)', opacity=0.4 if is_low else 1.0,
                bordercolor='gray', borderwidth=0.8, borderpad=2
            )

    # Layout configuration
    yaxis_cfg = dict(showgrid=True, gridcolor='#E5E5E5', zeroline=False)
    yaxis_title = None
    
    if layout == 'spatial':
        yaxis_cfg['type'] = 'log'
        yaxis_title = 'Intensity (log scale)'
    else:
        yaxis_cfg['showticklabels'] = False
        yaxis_cfg['type'] = 'linear'
    
    fig.update_layout(
        title=None if build_profiles else title,
        showlegend=show_legend,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        plot_bgcolor='white',
        autosize=True,
        height=900,
        xaxis=dict(
            showgrid=True, 
            gridcolor='#E5E5E5',
            zeroline=True, 
            zerolinecolor='#E5E5E5',
            showticklabels=layout == 'spatial',
            range=xlim if layout == 'spatial' and xlim else None
        ),
        yaxis=yaxis_cfg
    )
    
    if layout == 'spatial':
        fig.update_layout(xaxis_title='m/z', yaxis_title=yaxis_title)
    
    # Save outputs
    if save_html:
        if build_profiles:
            import json
            profiles_json = json.dumps({str(k): v for k, v in profiles.items()})
            initial_json = json.dumps([str(n) for n in initial_nodes])
            click_js = f"""
            var allProfiles = {profiles_json};
            var shownNodes = new Set({initial_json});
            var isNormalized = false; // Default to raw
            var currentColorscale = '{prob_colorscale}';
            
            setTimeout(function() {{
                var graphDiv = document.getElementsByClassName('plotly-graph-div')[0];
                if (!graphDiv) return;

                // Create collapsible control panel
                var controlDiv = document.createElement('div');
                controlDiv.id = 'controlPanel';
                controlDiv.style.position = 'absolute';
                controlDiv.style.left = '10px'; controlDiv.style.top = '10px';
                controlDiv.style.zIndex = '1000'; controlDiv.style.background = 'white';
                controlDiv.style.padding = '10px'; controlDiv.style.border = '1px solid #ccc';
                controlDiv.style.borderRadius = '4px'; controlDiv.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                controlDiv.style.maxWidth = '220px';

                // Collapse/Expand Button
                var toggleBtn = document.createElement('button');
                toggleBtn.innerText = '−'; toggleBtn.style.width = '24px'; toggleBtn.style.height = '24px';
                toggleBtn.style.padding = '0'; toggleBtn.style.marginBottom = '5px'; toggleBtn.style.cursor = 'pointer';
                toggleBtn.style.border = '1px solid #999'; toggleBtn.style.borderRadius = '3px'; toggleBtn.style.background = '#f0f0f0';
                
                var controlContent = document.createElement('div');
                controlContent.id = 'controlContent'; controlContent.style.display = 'block';

                // Intensity Dropdown
                var intensityLabel = document.createElement('label');
                intensityLabel.innerText = 'Intensity: '; intensityLabel.style.marginRight = '5px'; intensityLabel.style.fontSize = '12px';
                intensityLabel.style.display = 'block'; intensityLabel.style.marginBottom = '3px';
                var selectBox = document.createElement('select');
                selectBox.style.fontSize = '11px'; selectBox.style.width = '100%'; selectBox.style.marginBottom = '8px';
                var optRaw = document.createElement('option');
                optRaw.value = 'raw'; optRaw.text = 'Raw';
                var optNorm = document.createElement('option');
                optNorm.value = 'norm'; optNorm.text = 'Normalized';
                selectBox.appendChild(optRaw); selectBox.appendChild(optNorm);
                selectBox.value = 'raw';
                
                // Colorscale Dropdown
                var colorLabel = document.createElement('label');
                colorLabel.innerText = 'Colorscale: '; colorLabel.style.marginRight = '5px'; colorLabel.style.fontSize = '12px';
                colorLabel.style.display = 'block'; colorLabel.style.marginBottom = '3px';
                var colorSelect = document.createElement('select');
                colorSelect.style.fontSize = '11px'; colorSelect.style.width = '100%'; colorSelect.style.marginBottom = '8px';
                var colorOptions = [['viridis', 'Viridis'], ['RdYlGn ', 'RdYlGn '], ['plasma', 'Plasma']];
                colorOptions.forEach(function(opt) {{
                    var el = document.createElement('option');
                    el.value = opt[0]; el.text = opt[1];
                    colorSelect.appendChild(el);
                }});
                colorSelect.value = '{prob_colorscale}';

                // Edge Label Toggle
                var labelCheckbox = document.createElement('input');
                labelCheckbox.type = 'checkbox'; labelCheckbox.id = 'edgeLabelToggle';
                labelCheckbox.checked = {str(show_edge_labels).lower()};
                labelCheckbox.style.marginRight = '5px';
                var labelText = document.createElement('label');
                labelText.htmlFor = 'edgeLabelToggle'; labelText.innerText = 'Edge Labels';
                labelText.style.fontSize = '12px';

                controlContent.appendChild(intensityLabel); controlContent.appendChild(selectBox);
                controlContent.appendChild(colorLabel); controlContent.appendChild(colorSelect);
                controlContent.appendChild(document.createElement('br'));
                controlContent.appendChild(labelCheckbox); controlContent.appendChild(labelText);
                
                controlDiv.appendChild(toggleBtn); controlDiv.appendChild(controlContent);
                graphDiv.parentNode.insertBefore(controlDiv, graphDiv);

                // Toggle button functionality
                toggleBtn.addEventListener('click', function(e) {{
                    var isHidden = controlContent.style.display === 'none';
                    controlContent.style.display = isHidden ? 'block' : 'none';
                    toggleBtn.innerText = isHidden ? '−' : '+';
                }});

                // Colorscale Change Event - only update gradient nodes (not highlighted)
                colorSelect.addEventListener('change', function(e) {{
                    currentColorscale = e.target.value;
                    var colorscaleMap = {{
                        'viridis': 'Viridis',
                        'red_green': [[0, 'rgb(220,20,60)'], [1, 'rgb(34,139,34)']],  // Crimson to Forest Green
                        'plasma': 'Plasma'
                    }};
                    
                    // Find and update only the gradient node trace (first node trace that has colorscale)
                    for (var i = 0; i < graphDiv.data.length; i++) {{
                        var trace = graphDiv.data[i];
                        if (trace.marker && trace.marker.colorscale && trace.name === 'Nodes') {{
                            trace.marker.colorscale = colorscaleMap[currentColorscale];
                            Plotly.restyle(graphDiv, {{'marker.colorscale': [colorscaleMap[currentColorscale]]}}, [i]);
                            break; // Only update the first gradient node trace
                        }}
                    }}
                }});

                // Edge Toggle Event
                labelCheckbox.addEventListener('change', function(e) {{
                    var isVisible = e.target.checked;
                    var annots = graphDiv.layout.annotations || [];
                    var newAnnots = annots.map(function(a) {{
                        // Subplot titles don't have bordercolor, only edge labels do in this fig
                        if (a.bordercolor) {{ a.visible = isVisible; }}
                        return a;
                    }});
                    Plotly.relayout(graphDiv, {{'annotations': newAnnots}});
                }});

                // Profile Toggle Event
                selectBox.addEventListener('change', function(e) {{
                    isNormalized = (e.target.value === 'norm');
                    var updateIndices = [], updateY = [];
                    for (var i = 0; i < graphDiv.data.length; i++) {{
                        if (graphDiv.data[i].xaxis === 'x2') {{
                            var nId = graphDiv.data[i].name.replace('Node ', '');
                            if (allProfiles[nId]) {{
                                updateIndices.push(i);
                                updateY.push(isNormalized ? allProfiles[nId].intensity_norm : allProfiles[nId].intensity_raw);
                            }}
                        }}
                    }}
                    if (updateIndices.length > 0) {{ Plotly.restyle(graphDiv, {{y: updateY}}, updateIndices); }}
                    Plotly.relayout(graphDiv, {{'yaxis2.title.text': isNormalized ? 'Normalized Intensity' : 'Raw Intensity'}});
                }});

                // Map Clicks to Line Draws
                graphDiv.on('plotly_click', function(evt) {{
                    var pt = evt.points[0];
                    if (pt.curveNumber === undefined || pt.customdata === undefined) return;
                    var nodeId = String(pt.customdata);
                    if (!(nodeId in allProfiles)) return;

                    if (shownNodes.has(nodeId)) {{
                        var idx = -1;
                        for (var i = 0; i < graphDiv.data.length; i++) {{
                            if (graphDiv.data[i].name === ('Node ' + nodeId) && graphDiv.data[i].xaxis === 'x2') {{
                                idx = i; break;
                            }}
                        }}
                        if (idx >= 0) {{ Plotly.deleteTraces(graphDiv, idx); shownNodes.delete(nodeId); }}
                    }} else {{
                        var profile = allProfiles[nodeId];
                        var yData = isNormalized ? profile.intensity_norm : profile.intensity_raw;
                        Plotly.addTraces(graphDiv, {{
                            x: profile.rt, y: yData, mode: 'lines', name: 'Node ' + nodeId, 
                            xaxis: 'x2', yaxis: 'y2',
                            line: {{ color: profile.color, width: 2, dash: profile.is_low ? 'dash' : 'solid' }} 
                        }});
                        shownNodes.add(nodeId);
                    }}
                }});
            }}, 500); 
            """
            fig.write_html(save_html, post_script=click_js)
        else:
            fig.write_html(save_html)
        print(f'Interactive HTML saved → {save_html}')
    
    if static_filepath:
        save_static(G, subgraph, nodes, pos, low_prob_nodes, node_data,
                   layout, intensities_are_log, title, edge_names,
                   use_prob_styling, low_prob_opacity, prob_colorscale,
                   highlight_color, normal_color, node_size, edge_width,
                   static_filepath, style, xlim)
    
    if show:
        fig.show()
    
    return fig


if __name__ == "__main__":
    print("Config adduct diffs:", EDGE_TYPE_NAMES_POSITIVE)
    print('\n \n \n')
    print("Config adduct diffs (negative):", EDGE_TYPE_NAMES_NEGATIVE)