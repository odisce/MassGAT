import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import LayerNorm


class FourierScalarEncoder(nn.Module):
    """
    Fourier feature encoding for scalar inputs (e.g. into, imax, etc.)
    """
    def __init__(self, input_dim=8, embed_dim=64, sigma=1.0):
        super().__init__()
        # Random Fourier Features projection
        # B mapping input_dim -> embed_dim // 2
        self.register_buffer('B', torch.randn(input_dim, embed_dim // 2) * sigma)
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: [Batch, input_dim]
        # Fourier Projection:
        proj = 2 * math.pi * (x @ self.B) 
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.output_mlp(ff)

class ProfileTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=2, num_layers=1, dropout_seq = 0.3):
        super().__init__()
        
        # Project Intensity (scalar) to Vector
        self.input_proj = nn.Linear(1, d_model)
        
        # Project Time (scalar) to Vector (This replaces fixed PositionalEncoding)
        self.time_proj = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            batch_first=True, 
            dropout=dropout_seq,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Linear(d_model, 1)

    def forward(self, x_intensity, x_time):
        """
        x_intensity: [Batch, 32] - Normalized Intensities
        x_time:      [Batch, 32] - Relative RT (in seconds or normalized units)
        """
        # embed intensity and time separately
        h_int = self.input_proj(x_intensity.unsqueeze(-1))
        h_time = self.time_proj(x_time.unsqueeze(-1))
        
        # combine
        h = h_int + h_time
        
        # padding mask
        padding_mask = (x_intensity == 0)
        
        # transformer Pass
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        
        # attention pooling
        scores = self.attention_pool(h).squeeze(-1) # [Batch, 32]
        scores = scores.masked_fill(padding_mask, -1e9) # Ignore padding
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        
        # Weighted Sum
        return torch.sum(h * weights, dim=1) # [Batch, d_model]


class CNNPeakEncoder(nn.Module):
    """
    Standard 1D-CNN baseline for peak profile encoding.
    """
    def __init__(self, output_dim=64, dropout_seq = 0.3):
        super().__init__()
        self.dropout_seq = dropout_seq

        # layer 1
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # layer 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # global max pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # final projection
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x_intensity, x_time):
        """
        x_intensity: [Batch, 32]
        x_time:      [Batch, 32]
        """

        x = torch.stack([x_intensity, x_time], dim=1) # [Batch, 2, 32]
        
        # convolutional Layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, p=self.dropout_seq, training=self.training)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=self.dropout_seq, training=self.training)

        x = F.relu(self.bn3(self.conv3(x))) # Output: [Batch, 64, 32]
        
        # global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # final projection
        return self.fc(x)
    

class EdgeHead(nn.Module):
    """
    Predicts edge properties given node embeddings and edge features.
    """
    def __init__(self, node_embed_dim, edge_feat_dim, hidden_dim=64, num_classes=1, dropout_ffn = 0.2):
        super().__init__()
        # Input: [Node_i_Embed, Node_j_Embed, Edge_Attr]
        self.input_dim = (node_embed_dim * 2) + edge_feat_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(hidden_dim // 2, num_classes) # Output 1 logit
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, embed_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim]
        """
        row, col = edge_index
        
        # get embeddings for source and target nodes
        x_i = x[row]
        x_j = x[col]
        
        # concatenate node_i + node_j + edge_features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # predict
        return self.mlp(edge_input).squeeze(-1)
    

class MassGAT(nn.Module):
    """
    Main MassGAT model
    """
    
    def __init__(self, num_features, hidden_dim=32, embedding_dim=32, 
                 num_iterations=2, num_layers_per_iter=2, encoder_type='transformer', edge_dim=12, block_type='gate',
                 dropout_seq = 0.3, dropout_ffn=0.2, dropout_gnn=0.3, zero_metadata_all_nodes=False):
        super().__init__()
        self.use_attention = True
        self.num_iterations = num_iterations
        self.num_layers_per_iter = num_layers_per_iter
        self.encoder_type = encoder_type
        self.block_type = block_type
        self.zero_metadata_all_nodes = zero_metadata_all_nodes
        self.edge_type_embedding = nn.Embedding(36, 8)

        # Input projection components
        self.signal_len = 32
        self.scalar_dim = num_features - self.signal_len
        self.signal_embed_dim = 64 # Dimension coming out of Transformer

        self.scalar_encoder = FourierScalarEncoder(input_dim=8, embed_dim=self.signal_embed_dim)
        self.signal_encoder = ProfileTransformerEncoder(d_model=self.signal_embed_dim, dropout_seq=dropout_seq)
        self.fusion_proj = nn.Linear(self.signal_embed_dim * 2, hidden_dim)
        self.signal_encoder_cnn = CNNPeakEncoder(output_dim=self.signal_embed_dim, dropout_seq=dropout_seq)
        self.fusion_proj_cnn = nn.Linear(self.signal_embed_dim * 2, hidden_dim)
        self.input_proj = nn.Linear(num_features, hidden_dim)

        
        # Iterative blocks (ablation: gate, gat, gcn)
        self.blocks = nn.ModuleList()
        for _ in range(num_iterations):
            if block_type == 'gate':
                self.blocks.append(GatedGNNBlock(hidden_dim=hidden_dim, num_layers=num_layers_per_iter,edge_dim=edge_dim, dropout_gnn=dropout_gnn))
            elif block_type == 'gat':
                self.blocks.append(GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout_gnn, edge_dim=edge_dim))
            elif block_type == 'gcn':
                self.blocks.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown block_type: {block_type}")
        
        # Final embedding projection
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.Dropout(0.1)
        )
        
        # Graph-level classification head
        self.graph_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Node-level classification head
        self.node_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Edge-level classification head
        self.edge_head = EdgeHead(node_embed_dim=embedding_dim, 
                                  edge_feat_dim=edge_dim, 
                                  hidden_dim=hidden_dim//2,
                                  dropout_ffn=dropout_ffn)
        
        self.singleton_node_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, 1)
        )

        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _input_projection(self, x, encoder_type='transformer', is_singleton=None):
        """
        Encodes scalar features and profile separately, then fuses them.
        Uses CNN encoder for singleton nodes, encoder_type for others.
        """
        scalars = x[:, 0:8]
        rt_seq = x[:, 8:40]
        int_seq = x[:, 40:72]

        if is_singleton is None:
            is_singleton = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        out = torch.zeros(x.size(0), self.fusion_proj_cnn.out_features, device=x.device)

        # Process singleton nodes with CNN encoder
        if is_singleton.any():
            emb_scalar_s = self.scalar_encoder(scalars[is_singleton])
            emb_profile_s = self.signal_encoder_cnn(int_seq[is_singleton], rt_seq[is_singleton])
            combined_s = torch.cat([emb_scalar_s, emb_profile_s], dim=1)
            out[is_singleton] = self.fusion_proj_cnn(combined_s)

        # Process non-singleton nodes with specified encoder_type
        if (~is_singleton).any():
            emb_scalar_n = self.scalar_encoder(scalars[~is_singleton])
            if encoder_type == 'transformer':
                emb_profile_n = self.signal_encoder(int_seq[~is_singleton], rt_seq[~is_singleton])
                combined_n = torch.cat([emb_scalar_n, emb_profile_n], dim=1)
                out[~is_singleton] = self.fusion_proj(combined_n)
            elif encoder_type == 'cnn':
                emb_profile_n = self.signal_encoder_cnn(int_seq[~is_singleton], rt_seq[~is_singleton])
                combined_n = torch.cat([emb_scalar_n, emb_profile_n], dim=1)
                out[~is_singleton] = self.fusion_proj_cnn(combined_n)
            else:
                out[~is_singleton] = self.input_proj(x[~is_singleton])

        return out


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is not None:
            edge_types = edge_attr[:, 1].long()
            edge_type_emb = self.edge_type_embedding(edge_types)
            edge_continuous = torch.cat([edge_attr[:, 0:1], edge_attr[:, 2:]], dim=1)
            edge_attr = torch.cat([edge_type_emb, edge_continuous], dim=1)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Singleton mask computed once
        if edge_index.numel() > 0:
            connected = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            connected[edge_index[0]] = True
            connected[edge_index[1]] = True
            is_singleton = ~connected
        else:
            is_singleton = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        # Set first 8 scalar values to 0 (singleton-only by default, or all nodes if zero_metadata_all_nodes=True)
        if self.zero_metadata_all_nodes:
            x = x.clone()
            x[:, 0:8] = 0
        elif is_singleton.any():
            x = x.clone()
            x[is_singleton, 0:8] = 0

        x0 = self._input_projection(x, encoder_type=self.encoder_type, is_singleton=is_singleton)
        x0 = F.relu(x0)
        x = x0.clone()

        all_gates = []
        for block in self.blocks:
            if self.block_type == 'gate':
                residual = x
                x_new, gates = block(x, edge_index, edge_attr, batch)
                x_new = x_new + residual
                # singletons keep their x0 representation
                x = torch.where(is_singleton.unsqueeze(-1), x0, x_new)
                all_gates.append(gates)
            elif self.block_type == 'gat':
                x_new = block(x, edge_index, edge_attr=edge_attr)
                x = torch.where(is_singleton.unsqueeze(-1), x0, x_new)
            elif self.block_type == 'gcn':
                x_new = block(x, edge_index)
                x = torch.where(is_singleton.unsqueeze(-1), x0, x_new)
            else:
                raise ValueError(f"Unknown block_type: {self.block_type}")

        x_combined = x + x0
        node_embeddings = self.final_proj(x_combined)

        # Separate classification heads for singletons vs connected
        node_logits = torch.zeros(x.size(0), device=x.device)
        if (~is_singleton).any():
            node_logits[~is_singleton] = self.node_head(
                node_embeddings[~is_singleton]).squeeze(-1)
        if is_singleton.any():
            node_logits[is_singleton] = self.singleton_node_head(
                node_embeddings[is_singleton]).squeeze(-1)

        edge_logits = self.edge_head(node_embeddings, edge_index, edge_attr)
        graph_mean = global_mean_pool(node_embeddings, batch)
        graph_max = global_max_pool_alt(node_embeddings, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        graph_logits = self.graph_head(graph_repr).squeeze(-1)

        return {
            "node_embeddings": node_embeddings,
            "node_logits": node_logits,
            "graph_logits": graph_logits,
            "edge_logits": edge_logits,
            "edge_index": edge_index,
            "is_singleton": is_singleton,  # pass through for loss
        }


class IterativeCNNIndependentClassifier(nn.Module):
    """
    Independent node classifier that reuses trained weights from MassGAT.
    """
    def __init__(self, iterative_model, use_singleton_head=True):
        super().__init__()
        self.scalar_encoder = iterative_model.scalar_encoder
        self.signal_encoder_cnn = iterative_model.signal_encoder_cnn
        self.fusion_proj_cnn = iterative_model.fusion_proj_cnn
        self.final_proj = iterative_model.final_proj
        if use_singleton_head and hasattr(iterative_model, 'singleton_node_head'):
            self.node_head = iterative_model.singleton_node_head
        else:
            self.node_head = iterative_model.node_head

    def forward(self, x):
        scalars = x[:, 0:8]
        rt_seq = x[:, 8:40]
        int_seq = x[:, 40:72]

        emb_scalar = self.scalar_encoder(scalars)
        emb_profile = self.signal_encoder_cnn(int_seq, rt_seq)
        combined = torch.cat([emb_scalar, emb_profile], dim=1)
        x0 = F.relu(self.fusion_proj_cnn(combined))

        # Mirror the iterative terminal projection while keeping nodes independent.
        node_embeddings = self.final_proj(x0 + x0)
        logits = self.node_head(node_embeddings).squeeze(-1)
        return logits


class GatedGNNBlock(nn.Module):
    """
    Performs gated GNN message passing within an iterative block.
    """
    def __init__(self, hidden_dim, num_layers=2, edge_dim=5, dropout_gnn=0.3):
        super().__init__()
        self.use_attention = True
        self.edge_dim = edge_dim
        self.dropout_gnn = dropout_gnn
        
        # Gate: Predicts reliability (0..1) for every edge
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

        self.convs = nn.ModuleList([
                GATv2Conv(
                    hidden_dim, hidden_dim // 4, heads=4, concat=True,
                    dropout=self.dropout_gnn, edge_dim=edge_dim
                )
                for _ in range(num_layers)
            ])

        
        self.layer_norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(self.dropout_gnn)
    
    def forward(self, x, edge_index, edge_attr, batch):
        row, col = edge_index
        edge_feat_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        gates = self.gate_mlp(edge_feat_input)
        gated_edge_attr = edge_attr * gates
        
        for i, conv in enumerate(self.convs):
            residual = x 
            
            x_new = conv(x, edge_index, edge_attr=gated_edge_attr)
            x_new = F.relu(x_new)
            x_new = self.layer_norms[i](x_new)
            x_new = self.dropout(x_new)
            x = x_new + residual
            
        return x, gates


def global_max_pool_alt(x, batch, size=None):
    """
    Alternative to global_max_pool that doesn't use torch_scatter
    
    Args:
        x: Node features [num_nodes, feature_dim]
        batch: Batch assignment [num_nodes]
        size: Number of graphs (optional)
    
    Returns:
        Pooled features [num_graphs, feature_dim]
    """
    if size is None:
        size = int(batch.max().item()) + 1
    
    # Initialize output with very negative values
    out = x.new_full((size, x.size(1)), float('-inf'))
    
    # Manual max pooling
    for i in range(size):
        mask = (batch == i)
        if mask.any():
            out[i] = x[mask].max(dim=0)[0]
    
    return out