import os
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from pathlib import Path
import sys
import time

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Function to reindex graph nodes
def reindex_graph(data_dict):
    """Reindex graph nodes using roi_values as the mapping."""
    if data_dict['edge_index'].numel() == 0:
        return data_dict['edge_index'], data_dict.get('edge_attr', None)

    # roi_values contains the original node IDs
    roi_values = data_dict['roi_values']
    num_nodes = roi_values.size(0)

    # Create inverse mapping: original_id -> new_index (0 to num_nodes-1)
    max_id = max(int(data_dict['edge_index'].max().item()), int(roi_values.max().item()))
    node_mapping = torch.full((max_id + 1,), -1, dtype=torch.long)
    node_mapping[roi_values] = torch.arange(num_nodes)

    # Apply mapping to edge_index
    data_dict['edge_index'] = node_mapping[data_dict['edge_index']]

    # Remove invalid edges
    valid_mask = (data_dict['edge_index'] >= 0).all(dim=0)
    data_dict['edge_index'] = data_dict['edge_index'][:, valid_mask]

    if 'edge_attr' in data_dict:
        data_dict['edge_attr'] = data_dict['edge_attr'][valid_mask]

    return data_dict['edge_index'], data_dict.get('edge_attr', None)



# Function to load a single synthetic file (for training)
def load_single_synthetic_file(path):
    """
    Load a single synthetic graph file, perform feature engineering, and return a PyG Data object.
    
    Args:
        path: Path to the .pt file
    
    Returns:
        Data: PyTorch Geometric Data object
    """

    start_time = time.time() 

    data_dict = torch.load(path, map_location='cpu')
    torch_time = time.time() - start_time

    # 1. Reindex: Maps Global IDs to Local Indices (0..N)
    edge_index, edge_attr = reindex_graph(data_dict)

    # Identify connected nodes
    # If there are no edges, keep a single random node to represent the empty component
    if edge_index.numel() == 0:
        num_total_nodes = data_dict['node_features'].size(0)
        if num_total_nodes == 0:
            return None, None
        random_node_local = torch.randint(0, num_total_nodes, (1,))
        connected_nodes = random_node_local
    else:
        connected_nodes = torch.unique(edge_index)
    num_connected = connected_nodes.size(0)
    
    #Filter Node Features
    raw_x = data_dict['node_features'][connected_nodes]
    
    #Filter Metadata (ROI Values)
    subset_roi_values = data_dict['roi_values'][connected_nodes]
    
    # Filter Labels & Mask
    all_y = data_dict.get('ground_truth_labels', torch.zeros(data_dict['node_features'].size(0), dtype=torch.long))
    
    # Handle cases where labels might be larger or same size
    if all_y.size(0) >= data_dict['node_features'].size(0):
        y = all_y[connected_nodes]
    else:
        y = all_y[:num_connected]

    all_mask = data_dict.get('positive_mask', torch.ones(data_dict['node_features'].size(0), dtype=torch.bool))
    
    if all_mask.size(0) >= data_dict['node_features'].size(0):
        positive_mask = all_mask[connected_nodes]
    else:
        positive_mask = all_mask[:num_connected]


    # Remap Edge Index to Compact Range (0..N_connected) only if there are edges
    if edge_index.numel() > 0:
        node_mapper = torch.full((edge_index.max() + 1,), -1, dtype=torch.long)
        node_mapper[connected_nodes] = torch.arange(num_connected)
        edge_index = node_mapper[edge_index]
    indexing_time = time.time() - start_time - torch_time

    # 2. Feature Engineering
    mz = raw_x[:, 0]       
    into = raw_x[:, 2]     
    rtmin = raw_x[:, 3]    
    imax = raw_x[:, 5]
    sc_length = raw_x[:, 7]     
    rt_seq = raw_x[:, 8:40]
    int_seq = raw_x[:, 40:72]

    log_into = torch.log1p(into)
    log_imax = torch.log1p(imax)
    mz_int, mz_float = torch.floor(mz), mz - torch.floor(mz)
    rt_norm = raw_x[:, 1] / 30.0 # max 30 minutes in the data
    mz_norm = mz_int / 1000.0

    # Substract RT Min
    rt_relative_seq = rt_seq - rtmin.unsqueeze(1)
    rt_relative_seq[rt_seq == 0] = 0.0

    # Intensity normalization by max
    int_max = int_seq.max(dim=1, keepdim=True).values
    int_relative_seq = int_seq / (int_max + 1e-6)

    # Construct X
    # Structure: [mz_norm, rt_norm, log_into, rtmin, rtmax, log_imax, mz_float, sc_len] + Profile
    x = raw_x.clone()
    x[:, 0] = mz_norm
    x[:, 1] = rt_norm
    x[:, 6] = mz_float
    x[:, 2] = log_into
    x[:, 5] = log_imax
    x[:, 7] = sc_length
    x[:, 8:40] = rt_relative_seq
    x[:, 40:72] = int_relative_seq

    # 3. Edge Attribute Engineering
    if edge_index.numel() > 0:
        row, col = edge_index
        log_into_ratio = log_into[row] - log_into[col]
        log_imax_ratio = log_imax[row] - log_imax[col]
        rt_diff = torch.abs(raw_x[row, 1] - raw_x[col, 1])
        new_edge_feats = torch.stack([log_into_ratio, log_imax_ratio, rt_diff], dim=1)
        if edge_attr is not None:
            final_edge_attr = torch.cat([edge_attr, new_edge_feats], dim=1)
        else:
            final_edge_attr = new_edge_feats
    else:
        edge_attr_dim = (data_dict['edge_attr'].size(1) + 3) if 'edge_attr' in data_dict else 3
        final_edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

    # Check lengths
    if x.size(0) != y.size(0) or x.size(0) != positive_mask.size(0):
        print(f"CRITICAL ERROR in {path}: x({x.size(0)}), y({y.size(0)}), mask({positive_mask.size(0)}) mismatch.")
        # Fix by truncating to minimum length to avoid crashing training
        min_len = min(x.size(0), y.size(0), positive_mask.size(0))
        x = x[:min_len]
        y = y[:min_len]
        positive_mask = positive_mask[:min_len]

    data_kwargs = dict(
        x=x,
        edge_index=edge_index,
        edge_attr=final_edge_attr,
        y=y,
        positive_mask=positive_mask,
        file_id=data_dict.get('file_name', None),
        roi_values=subset_roi_values
    )

    data = Data(**data_kwargs)
    
    data_creation_time = time.time() - start_time - torch_time - indexing_time

    return data, [torch_time, indexing_time, data_creation_time]



def load_mzml_file(path, min_component_size=1):
    """
    Loads a single .pt file (Real Data).
    Returns:
        data_list: list of Data objects (one per connected component)
        times: [torch_time, indexing_time, data_creation_time]
    """
    t0 = time.time()
    metrics = {}

    # 1. LOAD THE FILE
    data_dict = torch.load(path, map_location='cpu')
    
    for key in ['roi_values', 'edge_index', 'ground_truth_labels']:
        if key in data_dict and data_dict[key] is not None:
            data_dict[key] = data_dict[key].long()
            
    t1 = time.time()
    metrics['1_load_file_time'] = t1 - t0

    # 2. REINDEX GRAPH
    edge_index, global_edge_attr = reindex_graph(data_dict)
    t2 = time.time()
    metrics['2_reindex_time'] = t2 - t1

    # 3. FEATURE ENGINEERING
    raw_x = data_dict['node_features']
    mz = raw_x[:, 0]       
    into = raw_x[:, 2]     
    rtmin = raw_x[:, 3]    
    rtmax = raw_x[:, 4]
    imax = raw_x[:, 5]     
    rt_seq = raw_x[:, 8:40]
    int_seq = raw_x[:, 40:72]

    log_into = torch.log1p(into)
    log_imax = torch.log1p(imax)
    mz_int, mz_float = torch.floor(mz), mz - torch.floor(mz)
    rt_norm = raw_x[:, 1] / 30.0 
    mz_norm = mz_int / 1000.0
    rt_relative_seq = rt_seq - rtmin.unsqueeze(1)
    rt_relative_seq[rt_seq == 0] = 0.0

    int_max = int_seq.max(dim=1, keepdim=True).values
    int_relative_seq = int_seq / (int_max + 1e-6)

    rtmin = rtmin / 60.0
    rtmax = rtmax / 60.0
    rt_norm = rt_norm / 60.0
    rt_relative_seq = rt_relative_seq / 60.0

    x = raw_x.clone()
    x[:, 0] = mz_norm
    x[:, 1] = rt_norm
    x[:, 3] = rtmin
    x[:, 4] = rtmax
    x[:, 6] = mz_float
    x[:, 2] = log_into
    x[:, 5] = log_imax
    x[:, 8:40] = rt_relative_seq
    x[:, 40:72] = int_relative_seq

    # Edge Features
    if edge_index.numel() > 0:
        row, col = edge_index
        log_into_ratio = log_into[row] - log_into[col]
        log_imax_ratio = log_imax[row] - log_imax[col]
        rt_diff = torch.abs(raw_x[row, 1] - raw_x[col, 1])
        new_edge_feats = torch.stack([log_into_ratio, log_imax_ratio, rt_diff], dim=1)
        final_edge_attr = torch.cat([global_edge_attr, new_edge_feats], dim=1) if global_edge_attr is not None else new_edge_feats
    else:
        final_edge_attr = None

    t3 = time.time()
    metrics['3_feature_engineering_time'] = t3 - t2

    # 4. CONNECTED COMPONENTS
    num_nodes = x.size(0)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    num_components, component_labels = sp.csgraph.connected_components(adj, connection='weak')
    component_labels = torch.from_numpy(component_labels).long()
    
    t4 = time.time()
    metrics['4_connected_components_time'] = t4 - t3

    # Handle defaults
    all_y = data_dict.get('ground_truth_labels')
    if all_y is None: all_y = torch.zeros(num_nodes, dtype=torch.long)
    all_mask = data_dict.get('positive_mask')
    if all_mask is None: all_mask = torch.ones(num_nodes, dtype=torch.bool)
    all_rois = data_dict['roi_values']

    # 5. OPTIMIZED GRAPH SPLITTING
    # Pre-sort nodes by component label to avoid O(N) boolean masks in the loop
    node_sort_idx = torch.argsort(component_labels)
    sorted_node_labels = component_labels[node_sort_idx]
    unique_node_labels, node_counts = torch.unique_consecutive(sorted_node_labels, return_counts=True)
    node_groups = torch.split(node_sort_idx, node_counts.tolist())

    # Map each component label to its nodes
    label_to_nodes = {label.item(): nodes for label, nodes in zip(unique_node_labels, node_groups)}

    # Pre-sort edges by component label
    # An edge belongs to the same component as its source node (row)
    if edge_index.numel() > 0:
        edge_component_labels = component_labels[edge_index[0]]
        edge_sort_idx = torch.argsort(edge_component_labels)
        sorted_edge_labels = edge_component_labels[edge_sort_idx]
        unique_edge_labels, edge_counts = torch.unique_consecutive(sorted_edge_labels, return_counts=True)
        edge_groups = torch.split(edge_sort_idx, edge_counts.tolist())
        label_to_edges = {label.item(): edges for label, edges in zip(unique_edge_labels, edge_groups)}
    else:
        label_to_edges = {}

    # Global array used for ultra-fast local relabeling (avoids using pyg's slow subgraph)
    local_mapping = torch.empty(num_nodes, dtype=torch.long)

    data_list = []
    
    # Filter by minimum component size
    valid_labels = unique_node_labels[node_counts >= min_component_size]

    for label in valid_labels:
        label_item = label.item()
        
        # O(1) fetch instead of O(N) mask
        sub_nodes = label_to_nodes[label_item] 
        
        sub_x = x[sub_nodes]
        sub_y = all_y[sub_nodes]
        sub_pos_mask = all_mask[sub_nodes]
        sub_rois = all_rois[sub_nodes]
        is_true_component = (sub_y != -1).any().long()

        # Edge handling
        if label_item in label_to_edges:
            sub_edge_indices = label_to_edges[label_item]
            sub_edge_index = edge_index[:, sub_edge_indices]
            sub_edge_attr = final_edge_attr[sub_edge_indices] if final_edge_attr is not None else None
            
            # Fast relabeling: map original global IDs to local 0...N range
            local_mapping[sub_nodes] = torch.arange(len(sub_nodes))
            sub_edge_index = local_mapping[sub_edge_index]
        else:
            sub_edge_index = torch.empty((2, 0), dtype=torch.long)
            sub_edge_attr = torch.empty((0, final_edge_attr.size(1) if final_edge_attr is not None else 0))

        data = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            y=sub_y,
            positive_mask=sub_pos_mask,
            roi_values=sub_rois,
            graph_label=torch.tensor([is_true_component])
        )
        data_list.append(data)

    t5 = time.time()
    metrics['5_graph_splitting_loop_time'] = t5 - t4
    metrics['TOTAL_TIME'] = t5 - t0

    print(metrics)

    return data_list, metrics


# GraphDataset class
class GraphDataset(Dataset):
    """
    Dataset for the constructed chemical graphs.
    """
    def __init__(self, root_path, verbose=False, cache_file="synthetic_dataset.pt", load_from_cache=True, load_synthetic=True):
        """
        Args:
            root_path: Path to the root directory containing subfolders
            verbose: If True, print detailed loading information
            cache_file: Path to the cached .pt file for storing aggregated data
        """
        super().__init__()
        self.root_path = Path(root_path)
        self.verbose = verbose
        self.cache_file = Path(cache_file)
        self.load_synthetic = load_synthetic

        self.graphs = []
        self.graph_labels = []  # Graph-level labels (true/false)

        # Profiling measures
        self.torch_times = []
        self.indexing_times = []
        self.data_creation_times = []


        # Load graphs (from cache or raw files)
        if self.cache_file.exists() and load_from_cache:
            self._load_from_cache()
            print(f"Loaded dataset from cache {self.cache_file}, total graphs: {len(self.graphs)}")
        else:
            self._load_graphs()
            self._save_to_cache()


        # Print example graph info
        if len(self.graphs) > 0:
            self._print_example_graph(0)

        # Plot distributions after loading
        self._plot_feature_distributions()
    def _plot_feature_distributions(self):
        """Plot and save distributions of log_into, log_imax, sc_length for all graphs."""
        if len(self.graphs) == 0:
            print("No graphs to plot feature distributions.")
            return

        log_into_all = []
        log_imax_all = []
        sc_length_all = []

        for data in self.graphs:
            x = data.x
            # Defensive: check shape
            if x.size(1) < 8:
                continue
            log_into_all.append(x[:, 2].cpu().numpy())
            log_imax_all.append(x[:, 5].cpu().numpy())
            sc_length_all.append(x[:, 7].cpu().numpy())

        import numpy as np
        log_into_all = np.concatenate(log_into_all)
        log_imax_all = np.concatenate(log_imax_all)
        sc_length_all = np.concatenate(sc_length_all)

        # Plot
        plt.figure(figsize=(15, 4))
        for i, (arr, name) in enumerate(zip([log_into_all, log_imax_all, sc_length_all],
                                            ["log_into", "log_imax", "sc_length"])):
            plt.subplot(1, 3, i+1)
            sns.histplot(arr, bins=50, kde=True)
            plt.title(f"Distribution of {name}")
            plt.xlabel(name)
            plt.ylabel("Count")
        plt.tight_layout()

        # Determine output directory
        if self.cache_file is not None and self.cache_file.parent.exists():
            out_dir = self.cache_file.parent
        else:
            out_dir = self.root_path if self.root_path.exists() else Path(".")
        out_path = out_dir / "feature_distributions.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved feature distributions plot to {out_path}")


    def _load_from_cache(self):
        """Load dataset from a cached .pt file."""
        if self.verbose:
            print(f"Loading dataset from cache: {self.cache_file}")
        cache_data = torch.load(self.cache_file, weights_only=False)
        self.graphs = cache_data['graphs']
        self.graph_labels = cache_data['graph_labels']

    def _save_to_cache(self):
        """Save the processed dataset to a cached .pt file."""
        if self.verbose:
            print(f"Saving dataset to cache: {self.cache_file}")
        torch.save({'graphs': self.graphs, 'graph_labels': self.graph_labels}, self.cache_file)

    def _load_graphs(self):
        """Load all graphs from subdirectories."""
        total_skipped = 0
        pt_files = list(self.root_path.rglob("*.pt"))
        print(f"Found {len(pt_files)} .pt files in {self.root_path}")
        n_files = len(pt_files)
        progress_step = max(1, n_files // 20)  # 5% steps
        for idx, pt_file in enumerate(pt_files):
            if idx % progress_step == 0 or idx == n_files - 1:
                print(f"Loading progress: {idx+1}/{n_files} files ({100*(idx+1)//n_files}%)")
            try:
                if not self.load_synthetic:
                    subgraphs, times = load_mzml_file(pt_file)
                    if not subgraphs:
                        total_skipped += 1
                        continue
                    for data in subgraphs:
                        data.graph_label = torch.tensor([is_true_graph], dtype=torch.long)
                        self.graphs.append(data)
                        self.graph_labels.append(data.graph_label.item())
                else:
                    data, times = load_single_synthetic_file(pt_file)
                    if data is None:
                        total_skipped += 1
                        continue
                    self.graphs.append(data)
                    is_true_graph = (data.y != -1).any().long().item()
                    self.graph_labels.append(is_true_graph)
                if times is not None:
                    self.torch_times.append(times[0])
                    self.indexing_times.append(times[1])
                    self.data_creation_times.append(times[2])
            except Exception as e:
                print(f"Error loading {pt_file.name}: {e}")
                total_skipped += 1
        print(f"Loaded {len(self.graphs)} graphs ({total_skipped} skipped)")
        # display profiling summary
        if len(self.torch_times) > 0:
            mean_torch_time = sum(self.torch_times) / len(self.torch_times)
            mean_indexing_time = sum(self.indexing_times) / len(self.indexing_times)
            mean_data_creation_time = sum(self.data_creation_times) / len(self.data_creation_times)
            print(f"[Profiling Summary] Mean time to load file with torch.load: {mean_torch_time:.4f} seconds")
            print(f"[Profiling Summary] Mean time for reindexing graph: {mean_indexing_time:.4f} seconds")
            print(f"[Profiling Summary] Mean time to create PyG Data object: {mean_data_creation_time:.4f} seconds")
            print(f"[Profiling Summary] Total time to load all files: {sum(self.torch_times) + sum(self.indexing_times) + sum(self.data_creation_times):.4f} seconds")
        
    def _print_example_graph(self, idx):
        """Print information about an example graph."""
        data = self.graphs[idx]
        print("\n=== Example Graph Content ===")
        print(f"Node Features (x): {data.x}")
        print(f"Edge Index: {data.edge_index.shape}")
        print(f"Edge Attributes: {data.edge_attr}")
        print(f"Node Labels (y): {data.y.shape}")
        print(f"Positive Mask: {data.positive_mask.shape}")
        print(f"Edge index range: [{data.edge_index.min()}, {data.edge_index.max()}]")
        print(f"Number of nodes: {data.num_nodes}")
        print("=" * 30 + "\n")
        sys.stdout.flush()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        Get a graph by index.
        
        Returns:
            Data: PyG Data object with both node-level labels (y) and graph-level label
        """
        return self.graphs[idx]

    def get_num_classes(self):
        """Get number of unique graph-level classes."""
        return len(set(self.graph_labels))
    
    def get_num_node_classes(self):
        """Get number of unique node-level classes."""
        # Assuming ground_truth_labels use -1 for unlabeled nodes
        all_labels = []
        for data in self.graphs:
            valid_labels = data.y[data.y != -1]
            all_labels.extend(valid_labels.tolist())
        
        if len(all_labels) == 0:
            return 0
        
        return len(set(all_labels))



def load_real_dataset(root_path, verbose=False, cache_file=None, load_from_cache=True):
    """
    Load a real dataset.
    
    Args:
        root_path: Path to the root directory
        verbose: If True, print detailed information
        cache_file: Filename for caching the dataset
        load_from_cache: If True, attempt to load dataset from cache
    
    Returns:
        GraphDataset: Dataset instance
    """    
    start_time = time.time()  # Start profiling

    # for cache name, take the directory name of root_path and add .pt extension
    if cache_file is None:
        cache_file = os.path.splitext(root_path.rstrip('/\\'))[0] + '.pt'

    dataset = GraphDataset(root_path, verbose=verbose, cache_file=cache_file, load_from_cache=load_from_cache, load_synthetic=False)
    end_time = time.time()  # End profiling

    print(f"[Profiling] Total time to load real dataset: {end_time - start_time:.2f} seconds")
    return dataset


def load_synthetic_dataset(root_path, verbose=False, cache_file=None, load_from_cache=True):
    """
    Load the synthetic dataset.
    
    Args:
        root_path: Path to the root directory
        verbose: If True, print detailed information
        cache_file: Filename for caching the dataset
        load_from_cache: If True, attempt to load dataset from cache
    
    Returns:
        GraphDataset: Dataset instance
    """    
    start_time = time.time()  # Start profiling

    # add a .pt to the root_path for caching if not provided
    if cache_file is None:
        cache_file = os.path.splitext(root_path.rstrip('/\\'))[0] + '.pt'

    dataset = GraphDataset(root_path, verbose=verbose, cache_file=cache_file, load_from_cache=load_from_cache)
    end_time = time.time()  # End profiling

    print(f"[Profiling] Total time to load synthetic dataset: {end_time - start_time:.2f} seconds")
    return dataset