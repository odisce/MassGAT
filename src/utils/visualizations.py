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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Build edge type name mappings (number -> name)
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
    'loss': '#F44336',         # red
    'mod': '#9C27B0',          # purple
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
        return 'adduct'
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



def create_nx_graph_from_torch(graph, ionization_mode = "positive"):
    if ionization_mode == "positive":
        edge_type_names = EDGE_TYPE_NAMES_POSITIVE
    elif ionization_mode == "negative":
        edge_type_names = EDGE_TYPE_NAMES_NEGATIVE

    G = nx.Graph()

    roi_values = graph["roi_values"]
    # Create mapping from ROI values to node indices
    roi_to_idx = {roi_values[i].item(): i for i in range(len(roi_values))}
    print(f"ROI to index mapping created: {len(roi_to_idx)} ROIs")

    node_features = graph["node_features"]

    # Add nodes with features (using node indices as NetworkX node IDs)
    for i in range(len(roi_values)):
        roi = roi_values[i].item()
        features = node_features[i, :8].numpy()  # Basic features
        G.add_node(i,  # Use node index as NetworkX node ID
                roi=roi,
                mz=features[0], 
                rt=features[1], 
                intensity=features[2],
                rt_min=features[3],
                rt_max=features[4],
                max_intensity=features[5],
                mzmean=features[6],
                scan_length=features[7])

    # Add edges with types (converting ROI values to node indices)
    if "edge_index" in graph and "edge_attr" in graph:
        edge_index = graph["edge_index"].numpy()
        edge_attr = graph["edge_attr"].numpy()
        for i in range(edge_index.shape[1]):
            source_roi, target_roi = edge_index[0, i].item(), edge_index[1, i].item()
            # Convert ROI values to node indices
            source_idx = roi_to_idx.get(source_roi)
            target_idx = roi_to_idx.get(target_roi)
            
            if source_idx is not None and target_idx is not None:
                edge_type = edge_attr[i, 1].item() if edge_attr.shape[1] > 1 else 0  # Column 1 = edge types
                edge_name = edge_type_names.get(edge_type, f'unknown_{edge_type}')
                G.add_edge(source_idx, target_idx, type=edge_type, name=edge_name)
            else:
                print(f"Warning: ROI not found in mapping - Source: {source_roi}, Target: {target_roi}")
    
    return G


def get_midpoint(pos, u, v, layout, intensities_are_log):
    """Calculate midpoint of an edge."""
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    mid_x = (x0 + x1) / 2
    
    if layout == 'spatial' and intensities_are_log:
        mid_y = math.sqrt(max(y0, 0.1) * max(y1, 0.1))  # geometric mean in log space
    else:
        mid_y = (y0 + y1) / 2
    return mid_x, mid_y
 
 
def _node_color_value(
    node: Any,
    G: nx.Graph,
    graph_name: str | None,
    node_probs: dict | None,
    manual_highlight_set: set,
    use_probability_styling: bool,
    prob_threshold: float,
    highlight_color: str,
    normal_node_color: str,
) -> tuple[Any, float | None, bool]:
    """
    Return (color_or_value, probability, is_low_prob).

    color_or_value is either:
      - a hex string  (fixed colour)
      - a float [0,1] (probability → fed to a colorscale)
    """
    roi = G.nodes[node]["roi"]
    prob: float | None = None
    is_low_prob = False

    if use_probability_styling and node_probs is not None:
        prob = node_probs.get((graph_name, roi), node_probs.get(node))
        is_low_prob = prob is None or prob < prob_threshold

    if node in manual_highlight_set:
        return highlight_color, prob, is_low_prob

    if use_probability_styling and node_probs is not None:
        if prob is not None:
            return float(prob), prob, is_low_prob
        return 0.0, None, True   # unknown → treat as 0

    return normal_node_color, None, False
 
 
# ---------------------------------------------------------------------------
# Plotly edge-trace builder
# ---------------------------------------------------------------------------
 
def _build_edge_traces(
    subgraph: nx.Graph,
    pos: dict,
    low_prob_nodes: set,
    use_probability_styling: bool,
    low_prob_edge_opacity: float,
    edge_width: float,
    layout: str,
    intensities_are_log: bool,
    edge_type_names: dict,
) -> tuple[list[go.Scatter], list[float], list[float], list[str]]:
    """
    Build per-category edge traces and hover-midpoint data.
 
    Returns
    -------
    traces, mid_x_list, mid_y_list, mid_hover_text
    """
    edges_by_category: dict[str, dict] = {}
    mid_x_list:   list[float] = []
    mid_y_list:   list[float] = []
    mid_hover_text: list[str] = []
 
    for u, v, data in subgraph.edges(data=True):
        edge_type = data["type"]
        edge_name = edge_type_names.get(edge_type, f"unknown_{edge_type}")
        category  = _edge_category(edge_type)
 
        bucket = edges_by_category.setdefault(
            category, {"normal": {"x": [], "y": []}, "shadow": {"x": [], "y": []}}
        )
 
        is_shadow = (
            use_probability_styling
            and (u in low_prob_nodes or v in low_prob_nodes)
        )
        subtype = "shadow" if is_shadow else "normal"
 
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        bucket[subtype]["x"].extend([x0, x1, None])
        bucket[subtype]["y"].extend([y0, y1, None])
 
        mx, my = _get_midpoint(pos, u, v, layout, intensities_are_log)
        mid_x_list.append(mx)
        mid_y_list.append(my)
        hover_str = f"{edge_name} (type {edge_type})"
        if is_shadow:
            hover_str += "<br><i>(Low Probability Link)</i>"
        mid_hover_text.append(hover_str)
 
    traces: list[go.Scatter] = []
    for category, subtypes in edges_by_category.items():
        base_color = EDGE_COLORS.get(category, "#888")
        if subtypes["normal"]["x"]:
            traces.append(go.Scatter(
                x=subtypes["normal"]["x"], y=subtypes["normal"]["y"],
                line=dict(width=edge_width, color=base_color),
                mode="lines", hoverinfo="skip",
                name=f"{category.title()} edges", opacity=1.0,
            ))
        if use_probability_styling and subtypes["shadow"]["x"]:
            traces.append(go.Scatter(
                x=subtypes["shadow"]["x"], y=subtypes["shadow"]["y"],
                line=dict(width=edge_width, color=base_color),
                mode="lines", hoverinfo="skip",
                name=f"{category.title()} (low prob)",
                opacity=low_prob_edge_opacity, showlegend=False,
            ))
 
    return traces, mid_x_list, mid_y_list, mid_hover_text
 
 
# ---------------------------------------------------------------------------
# Static (Matplotlib) export
# ---------------------------------------------------------------------------
 
def _save_static_figure(
    G: nx.Graph,
    subgraph: nx.Graph,
    component_nodes: list,
    pos: dict,
    low_prob_nodes: set,
    node_probs_resolved: list,
    layout: str,
    intensities_are_log: bool,
    title: str,
    edge_type_names: dict,
    use_probability_styling: bool,
    low_prob_edge_opacity: float,
    probability_colorscale: str,
    highlight_color: str,
    normal_node_color: str,
    node_size: float,
    edge_width: float,
    filepath: str,
    static_style: str = "article",
    xlim: tuple | None = None,
) -> None:


    style_cfg = _static_style_config(static_style)
    fig, ax = plt.subplots(figsize=style_cfg["figsize"])
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
 
    # ── Compute layout-dependent y-axis behaviour ───────────────────────────
    # Positions are always in original intensity space for spatial layout
    if layout == "spatial":
        ax.set_yscale("log")
        ax.set_ylabel("Intensity (log scale)", fontsize=style_cfg["axis_label_size"])
        ax.set_xlabel("m/z", fontsize=style_cfg["axis_label_size"])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
 
    # ── Draw edges ──────────────────────────────────────────────────────────
    for u, v, data in subgraph.edges(data=True):
        edge_type = data["type"]
        category  = _edge_category(edge_type)
        color     = EDGE_COLORS.get(category, "#888")
        alpha     = low_prob_edge_opacity if (
            use_probability_styling and (u in low_prob_nodes or v in low_prob_nodes)
        ) else 1.0
 
        x_vals = [pos[u][0], pos[v][0]]
        y_vals = [pos[u][1], pos[v][1]]
        ax.plot(x_vals, y_vals, color=color,
            lw=max(0.6, edge_width * style_cfg["edge_scale"]),
                alpha=alpha, zorder=1, solid_capstyle="round")
 
    # ── Draw nodes ──────────────────────────────────────────────────────────
    # Separate gradient nodes from fixed-colour nodes
    grad_x, grad_y, grad_vals = [], [], []
    fixed_colors_map: dict[str, tuple[list, list]] = {}   # color → (xs, ys)
 
    for (node, color_val, prob, is_low) in node_probs_resolved:
        x, y = pos[node]
        if isinstance(color_val, float):
            grad_x.append(x); grad_y.append(y); grad_vals.append(color_val)
        else:
            bucket = fixed_colors_map.setdefault(color_val, ([], []))
            bucket[0].append(x); bucket[1].append(y)
 
    if grad_x:
        mpl_cmap = _plotly_to_mpl_cmap(probability_colorscale)
        sc = ax.scatter(grad_x, grad_y, c=grad_vals, cmap=mpl_cmap,
                        vmin=0, vmax=1,
                        s=(node_size * style_cfg["node_scale"]) ** 2,
                        zorder=3, edgecolors="white", linewidths=0.8)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.015, pad=0.01, shrink=0.6)
        cbar.set_label("Probability", fontsize=max(7, style_cfg["legend_size"]))
        cbar.ax.tick_params(labelsize=style_cfg["tick_size"])
 
    for color_val, (xs, ys) in fixed_colors_map.items():
        ax.scatter(xs, ys, c=color_val,
                   s=(node_size * style_cfg["node_scale"]) ** 2,
                   zorder=3, edgecolors="white", linewidths=0.8)
 
    # ── Edge labels ─────────────────────────────────────────────────────────
    selected_labels = _pick_edge_labels(
        subgraph=subgraph,
        pos=pos,
        layout=layout,
        intensities_are_log=intensities_are_log,
        max_labels=style_cfg["max_edge_labels"],
        min_sep=style_cfg["label_min_separation"],
    )

    for u, v, data, mx, my in selected_labels:
        edge_type = data["type"]
        edge_name = edge_type_names.get(edge_type, f"unknown_{edge_type}")
        short     = _short_label(edge_name)
        category  = _edge_category(edge_type)
        text_color = EDGE_COLORS.get(category, "#555")
 
        is_low = use_probability_styling and (
            u in low_prob_nodes or v in low_prob_nodes
        )
        alpha = 0.35 if is_low else 0.95
 
        txt = ax.text(mx, my, short,
            fontsize=style_cfg["label_size"], ha="center", va="center",
            color=text_color, alpha=alpha, zorder=5,
            bbox=dict(
                boxstyle=f"round,pad={style_cfg['label_pad']}",
                facecolor="white",
                edgecolor=text_color,
                linewidth=0.8,
                alpha=alpha * 0.9,
            )
        )
        txt.set_path_effects([
            pe.withStroke(linewidth=1.5, foreground="white", alpha=0.95)
        ])
 
    # ── Legend ──────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=c, label=cat.title())
        for cat, c in EDGE_COLORS.items()
    ]
    ax.legend(handles=legend_handles,
        title="Edge type",
        fontsize=style_cfg["legend_size"],
        title_fontsize=style_cfg["legend_title_size"],
        loc="upper left", framealpha=0.9,
        edgecolor="0.8",         # light grey border
        handlelength=1.2,        # shorter colour swatch
        handleheight=0.8,
        borderpad=0.5
    )
 
    # ── Apply fixed m/z scale with margin ──────────────────────────────────
    if layout == "spatial" and xlim:
        margin = (xlim[1] - xlim[0]) * 0.05  # 5% margin on each side
        ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
    
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_linewidth(0.5)
    ax.tick_params(axis="both", labelsize=style_cfg["tick_size"], width=0.5, length=3)

    fig.tight_layout()

    suffix = filepath.rsplit(".", 1)[-1].lower()
    if suffix not in {"pdf", "svg", "png"}:
        filepath += ".pdf"
 
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Static figure saved → {filepath}")
 
 
def _plotly_to_mpl_cmap(colorscale_name: str) -> str:
    """Map common Plotly colorscale names to Matplotlib equivalents."""
    mapping = {
        "RdYlGn": "RdYlGn",
        "Viridis": "viridis",
        "Plasma":  "plasma",
        "Blues":   "Blues",
        "Reds":    "Reds",
        "Hot":     "hot",
        "YlOrRd":  "YlOrRd",
    }
    return mapping.get(colorscale_name, "RdYlGn")


def _static_style_config(style: str) -> dict[str, Any]:
    """Return rendering settings for static exports."""
    style_name = (style or "presentation").lower()
    if style_name == "article":
        return {
            "figsize": (7.2, 4.6),
            "title_size": 9,
            "axis_label_size": 8,
            "tick_size": 7,
            "legend_size": 7,
            "legend_title_size": 7.5,
            "label_size": 7,
            "label_pad": 0.12,
            "node_scale": 1.0,
            "edge_scale": 0.55,
            "max_edge_labels": 24,
            "label_min_separation": 0.075,
        }
    return {
        "figsize": (14, 8),
        "title_size": 14,
        "axis_label_size": 11,
        "tick_size": 10,
        "legend_size": 9,
        "legend_title_size": 10,
        "label_size": 9,
        "label_pad": 0.18,
        "node_scale": 1.25,
        "edge_scale": 0.75,
        "max_edge_labels": 80,
        "label_min_separation": 0.045,
    }


def _edge_label_candidates(
    subgraph: nx.Graph,
    pos: dict,
    layout: str,
    intensities_are_log: bool,
) -> list[tuple[float, tuple[Any, Any, dict], tuple[float, float]]]:
    """Build sortable edge-label candidates: (score, edge_tuple, midpoint)."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(max(xs) - min(xs), 1e-9)
    y_span = max(max(ys) - min(ys), 1e-9)

    candidates: list[tuple[float, tuple[Any, Any, dict], tuple[float, float]]] = []
    for u, v, data in subgraph.edges(data=True):
        edge_type = data["type"]
        category = _edge_category(edge_type)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx = (x1 - x0) / x_span
        dy = (y1 - y0) / y_span
        length_score = math.hypot(dx, dy)
        category_bonus = 0.12 if category != "profile_correlation" else 0.0
        mx, my = _get_midpoint(pos, u, v, layout, intensities_are_log)
        candidates.append((length_score + category_bonus, (u, v, data), (mx, my)))

    return sorted(candidates, key=lambda x: x[0], reverse=True)


def _pick_edge_labels(
    subgraph: nx.Graph,
    pos: dict,
    layout: str,
    intensities_are_log: bool,
    max_labels: int,
    min_sep: float,
) -> list[tuple[Any, Any, dict, float, float]]:
    """Greedy label picker to reduce overlaps while preserving edge-type diversity."""
    candidates = _edge_label_candidates(subgraph, pos, layout, intensities_are_log)
    if max_labels <= 0 or not candidates:
        return []

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(max(xs) - min(xs), 1e-9)
    y_span = max(max(ys) - min(ys), 1e-9)

    selected: list[tuple[Any, Any, dict, float, float]] = []
    selected_midpoints: list[tuple[float, float]] = []
    seen_types: set[int] = set()

    for _, (u, v, data), (mx, my) in candidates:
        edge_type = data["type"]
        keep_for_type_coverage = edge_type not in seen_types

        too_close = False
        if not keep_for_type_coverage:
            for sx, sy in selected_midpoints:
                d = math.hypot((mx - sx) / x_span, (my - sy) / y_span)
                if d < min_sep:
                    too_close = True
                    break

        if too_close:
            continue

        selected.append((u, v, data, mx, my))
        selected_midpoints.append((mx, my))
        seen_types.add(edge_type)

        if len(selected) >= max_labels:
            break

    return selected
 
 
# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
 
def visualize_component_interactive(
    G: nx.Graph,
    component_nodes,
    title:                  str   = "Component Visualization",
    layout:                 str   = "spatial",       # 'spatial' | 'spring' | 'circular'
    highlight_nodes               = None,
    show_edge_labels:       bool  = False,
    save_html:              str | None = None,
    save_static:            str | None = None,       # path for PDF/PNG/SVG, e.g. "out.pdf"
    node_probs:             dict | None = None,
    graph_name:             str | None = None,
    ionization_mode:        str  = "positive",       # 'positive' or 'negative' (for edge label naming)
    use_probability_styling: bool = False,
    prob_threshold:         float = 0.5,
    low_prob_edge_opacity:  float = 0.2,
    normal_node_color:      str   = "#484C48",
    highlight_color:        str   = "#800080",
    node_size:              int   = 12,
    edge_width:             float = 2,
    show_legend:            bool  = True,
    intensities_are_log:    bool  = False,
    probability_colorscale: str   = "RdYlGn",
    show_colorbar:          bool  = True,
    show:                   bool  = True,
    edge_type_names:        dict | None = None,
    static_style:           str   = "presentation",   # 'presentation' | 'article'
    xlim:                   tuple | None = None,     # (xmin, xmax) for spatial layout
) -> go.Figure | None:
    """
    Interactive Plotly visualization of a spectral network component.
 
    Parameters
    ----------
    save_static : str, optional
        If given, also saves a static Matplotlib figure (PDF recommended).
        Supports .pdf, .svg, .png extensions.
    edge_type_names : dict, optional
        Mapping {edge_type_int: str}. If omitted, uses the module default mapping.
    static_style : str, optional
        Static export profile. Use "presentation" for slides, or "article"
        for one-column compact figures.
    xlim : tuple, optional
        (xmin, xmax) to fix the x-axis range for spatial layout. 
        Useful for consistent m/z scaling across multiple components of the same compound.
    """
    if edge_type_names is None:
        if ionization_mode == "positive":
            edge_type_names = EDGE_TYPE_NAMES_POSITIVE
        else:
            edge_type_names = EDGE_TYPE_NAMES_NEGATIVE
 
    if not component_nodes:
        print("No nodes to visualize")
        return None
 
    component_nodes = list(component_nodes)
    subgraph = G.subgraph(component_nodes)
 
    # ── Positions ─────────────────────────────────────────────────────────
    if layout == "spatial":
        if intensities_are_log:
            pos = {n: [G.nodes[n]["mz"], max(float(np.expm1(G.nodes[n]["intensity"])), 1e-6)]
                   for n in component_nodes}
        else:
            pos = {n: [G.nodes[n]["mz"], max(G.nodes[n]["intensity"], 1e-6)]
                   for n in component_nodes}
    elif layout == "spring":
        pos = nx.spring_layout(subgraph, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.circular_layout(subgraph)
 
    node_x = [pos[n][0] for n in component_nodes]
    node_y = [pos[n][1] for n in component_nodes]
 
    # ── Node colours & probabilities ──────────────────────────────────────
    manual_highlight_set = set(highlight_nodes) if highlight_nodes else set()
    low_prob_nodes: set = set()
 
    hover_text:         list[str] = []
    node_colors:        list      = []
    node_probs_resolved: list     = []   # (node, color_val, prob, is_low)
 
    for node in component_nodes:
        roi       = G.nodes[node]["roi"]
        mz        = G.nodes[node]["mz"]
        rt        = G.nodes[node]["rt"]
        intensity = G.nodes[node]["intensity"]
        degree    = G.degree[node]
        true_intensity = np.expm1(intensity) if intensities_are_log else intensity
 
        color_val, prob, is_low = _node_color_value(
            node, G, graph_name, node_probs,
            manual_highlight_set, use_probability_styling, prob_threshold,
            highlight_color, normal_node_color,
        )
        if is_low:
            low_prob_nodes.add(node)
 
        node_colors.append(color_val)
        node_probs_resolved.append((node, color_val, prob, is_low))
 
        hover_info  = f"<b>ROI: {roi}</b><br>Node {node}<br>"
        hover_info += f"m/z: {mz:.4f} | RT: {rt:.2f}<br>"
        hover_info += f"Intensity: {true_intensity:.2e}<br>Degree: {degree}<br>"
        if prob is not None:
            hover_info += f"Probability: {prob:.3f}<br>"
        elif use_probability_styling:
            hover_info += "Probability: N/A<br>"
        hover_text.append(hover_info)
 
    # ── Plotly node traces ─────────────────────────────────────────────────
    node_traces: list[go.Scatter] = []
 
    if use_probability_styling and node_probs is not None:
        grad_idx = [i for i, n in enumerate(component_nodes)
                    if n not in manual_highlight_set]
        if grad_idx:
            node_traces.append(go.Scatter(
                x=[node_x[i] for i in grad_idx],
                y=[node_y[i] for i in grad_idx],
                mode="markers",
                marker=dict(
                    size=node_size,
                    color=[node_colors[i] for i in grad_idx],
                    colorscale=probability_colorscale,
                    cmin=0, cmax=1,
                    colorbar=dict(title="Probability", thickness=15,
                                  len=0.7, x=1.02) if show_colorbar else None,
                    showscale=show_colorbar,
                    line=dict(width=1, color="white"),
                ),
                text=[f"ROI {G.nodes[component_nodes[i]]['roi']}" for i in grad_idx],
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=[hover_text[i] for i in grad_idx],
                name="Nodes", showlegend=True,
            ))
 
        hi_idx = [i for i, n in enumerate(component_nodes)
                  if n in manual_highlight_set]
        if hi_idx:
            node_traces.append(go.Scatter(
                x=[node_x[i] for i in hi_idx],
                y=[node_y[i] for i in hi_idx],
                mode="markers",
                marker=dict(size=node_size, color=highlight_color,
                            line=dict(width=2, color="white")),
                text=[f"ROI {G.nodes[component_nodes[i]]['roi']}" for i in hi_idx],
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=[hover_text[i] for i in hi_idx],
                name="Highlighted", showlegend=True,
            ))
    else:
        node_traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(size=node_size, color=node_colors,
                        line=dict(width=1, color="white")),
            text=[f"ROI {G.nodes[n]['roi']}" for n in component_nodes],
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text,
            name="Nodes",
        ))
 
    # ── Edge traces + hover midpoints ──────────────────────────────────────
    edge_traces, mid_x, mid_y, mid_hover = _build_edge_traces(
        subgraph, pos, low_prob_nodes,
        use_probability_styling, low_prob_edge_opacity,
        edge_width, layout, intensities_are_log, edge_type_names,
    )
 
    edge_hover_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=20, color="white", opacity=0),
        hovertext=mid_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False, name="Edge Info",
    )
 
    fig = go.Figure(data=edge_traces + [edge_hover_trace] + node_traces)
 
    # ── Edge label annotations (Plotly) ────────────────────────────────────
    if show_edge_labels:
        for u, v, data in subgraph.edges(data=True):
            edge_type = data["type"]
            edge_name = edge_type_names.get(edge_type, f"unknown_{edge_type}")
            short     = _short_label(edge_name)
            mx, my    = _get_midpoint(pos, u, v, layout, intensities_are_log)
            text_opacity = (
                0.4 if (use_probability_styling and
                        (u in low_prob_nodes or v in low_prob_nodes))
                else 1.0
            )
            fig.add_annotation(
                x=mx, y=my, text=short,
                showarrow=False,
                font=dict(size=8, color="black"),
                bgcolor="rgba(255,255,255,0.85)",
                opacity=text_opacity,
                bordercolor="gray",
                borderwidth=0.8,
                borderpad=2,
            )
 
    # ── Y-axis configuration ───────────────────────────────────────────────
    if layout == "spatial":
        yaxis_cfg   = dict(showgrid=True, gridcolor="#E5E5E5",
                           zeroline=False,
                           type="log")
        yaxis_title = "Intensity (log scale)"
    else:
        yaxis_cfg   = dict(showgrid=True, gridcolor="#E5E5E5",
                           zeroline=True, zerolinecolor="#E5E5E5",
                           showticklabels=False, type="linear")
        yaxis_title = None
 
    fig.update_layout(
        title=title,
        showlegend=show_legend,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        plot_bgcolor="white",
        width=1800, height=900,
        xaxis=dict(
            showgrid=True, gridcolor="#E5E5E5",
            zeroline=True, zerolinecolor="#E5E5E5",
            showticklabels=layout == "spatial",
            range=xlim if layout == "spatial" and xlim else None,
        ),
        yaxis=yaxis_cfg,
    )
    if layout == "spatial":
        fig.update_layout(xaxis_title="m/z", yaxis_title=yaxis_title)
 
    # ── Outputs ────────────────────────────────────────────────────────────
    if save_html:
        fig.write_html(save_html)
        print(f"Interactive HTML saved → {save_html}")
 
    if save_static:
        # Ensure positions passed to static figure are always in original intensity space
        _save_static_figure(
            G=G,
            subgraph=subgraph,
            component_nodes=component_nodes,
            pos=pos,
            low_prob_nodes=low_prob_nodes,
            node_probs_resolved=node_probs_resolved,
            layout=layout,
            intensities_are_log=intensities_are_log,
            title=title,
            edge_type_names=edge_type_names,
            use_probability_styling=use_probability_styling,
            low_prob_edge_opacity=low_prob_edge_opacity,
            probability_colorscale=probability_colorscale,
            highlight_color=highlight_color,
            normal_node_color=normal_node_color,
            node_size=node_size,
            edge_width=edge_width,
            filepath=save_static,
            static_style=static_style,
            xlim=xlim,
        )
 
    if show:
        fig.show()
 
    return fig

if __name__ == "__main__":
    print("Config adduct diffs:", EDGE_TYPE_NAMES_POSITIVE)
    print('\n \n \n')
    print("Config adduct diffs (negative):", EDGE_TYPE_NAMES_NEGATIVE)