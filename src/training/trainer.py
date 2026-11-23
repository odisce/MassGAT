"""
Main trainer class for MassGAT
"""
import torch
import torch.optim as optim
import torch.nn as nn
import time
import pandas as pd
from torch_geometric.loader import DataLoader

from ..data.augmentation import augment_graph
from .losses import full_hierarchical_loss
from ..data import load_synthetic_dataset


class ChroMSTrainer:
    """Main trainer class"""
    def __init__(self, model, learning_rate=0.0001, device='cpu',
                 alpha=1.0, beta=1.0, delta=1.0, gamma=0.5,
                 model_type="hierarchical", loss_fn=full_hierarchical_loss):
        self.model = model.to(device)
        self.device = device

        # set loss function parameters
        self.alpha = alpha # graph
        self.beta = beta # node
        self.delta = delta # edge
        self.gamma = gamma # consistency

        # set training parameters
        self.model_type = model_type
        self.hierarchical_consistency = True
        self.node_loss_only_on_true_graphs = True
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.9, patience=5
        )
        self.loss_fn = loss_fn
    
    def train_epoch(self, train_loader, augmentation="mixed"):
        """Train for one epoch"""
        self.model.train()
        total_train_loss = 0
        total_graph_loss = 0
        total_node_loss = 0
        batch_count = 0
        
        # Metrics
        graph_correct = 0
        graph_total = 0
        graph_correct_nonsingle = 0
        graph_total_nonsingle = 0
        node_correct = 0
        node_total = 0
        connected_node_correct = 0
        connected_node_total = 0
        singleton_node_correct = 0
        singleton_node_total = 0

        # Initialize accumulators for additional metrics
        total_avg_true_graph_prob = 0
        total_avg_false_graph_prob = 0
        total_consistency_loss = 0
        total_avg_edge_prob = 0
        total_edge_loss = 0

        for batch in train_loader:

            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            if augmentation != "none":
                batch = augment_graph(batch, augmentation)
                        
            # Forward pass
            model_output = self.model(batch)

            node_logits = model_output['node_logits']
            graph_logits = model_output['graph_logits']
            edge_logits = model_output.get('edge_logits', None)
            edge_index = model_output.get('edge_index', None)
            node_embeddings = model_output.get('node_embeddings', None)
            is_singleton = model_output.get('is_singleton', None)
        
            # Determine model type and compute loss
            if self.model_type == "hierarchical":
                loss_dict = self.loss_fn(
                        node_logits = node_logits, 
                        graph_logits = graph_logits, 
                        edge_logits = edge_logits,
                        edge_index = edge_index,
                        node_embeddings = node_embeddings,
                        labels = batch.y,
                        positive_mask = batch.positive_mask, 
                        batch = batch.batch,
                        alpha=self.alpha, beta=self.beta, gamma=self.gamma, delta=self.delta, use_focal=True,
                        is_singleton=is_singleton)
                
                loss = loss_dict['total_loss']

                # Accumulate additional metrics
                total_avg_true_graph_prob += loss_dict['avg_true_graph_prob']
                total_avg_false_graph_prob += loss_dict['avg_false_graph_prob']
                total_consistency_loss += loss_dict['consistency_loss'].item()
                total_avg_edge_prob += loss_dict.get('avg_edge_prob', 0)
                total_edge_loss += loss_dict.get('edge_loss', 0)
                total_graph_loss += loss_dict['graph_loss'].item()
                total_node_loss += loss_dict['node_loss'].item()
                
                # Calculate accuracies
                with torch.no_grad():
                    graph_preds = (torch.sigmoid(graph_logits) > 0.5).long()
                    graph_labels = []
                    for i in range(batch.num_graphs):
                        node_mask = (batch.batch == i)
                        # A graph is True if it has any node with label >= 0
                        is_true = (batch.y[node_mask] >= 0).any().long()
                        graph_labels.append(is_true)
                    graph_labels = torch.tensor(graph_labels, device=self.device)
                    
                    graph_correct += (graph_preds == graph_labels).sum().item()
                    graph_total += batch.num_graphs

                    # Non-singleton graph accuracy
                    edge_index_cpu = edge_index
                    graph_has_edges = torch.zeros(batch.num_graphs, dtype=torch.bool, device=self.device)
                    if edge_index_cpu is not None and edge_index_cpu.numel() > 0:
                        ei = edge_index_cpu
                        graph_has_edges.scatter_(0, batch.batch[ei[0]], torch.ones(ei.shape[1], dtype=torch.bool, device=self.device))
                    nonsingle_mask_g = graph_has_edges
                    if nonsingle_mask_g.sum() > 0:
                        graph_correct_nonsingle += (graph_preds[nonsingle_mask_g] == graph_labels[nonsingle_mask_g]).sum().item()
                        graph_total_nonsingle += nonsingle_mask_g.sum().item()

                    # Valid nodes are everything except -2
                    valid_node_mask = (batch.y != -2)

                    # Determine connectivity per node
                    is_conn = torch.zeros(node_logits.size(0), dtype=torch.bool, device=self.device)
                    if edge_index is not None and edge_index.numel() > 0:
                        is_conn[edge_index[0]] = True
                        is_conn[edge_index[1]] = True

                    if valid_node_mask.sum() > 0:
                        node_preds = (torch.sigmoid(node_logits[valid_node_mask]) > 0.5).long()
                        node_targets = (batch.y[valid_node_mask] >= 0).long()
                        node_correct += (node_preds == node_targets).sum().item()
                        node_total += valid_node_mask.sum().item()

                        conn_valid = valid_node_mask & is_conn
                        if conn_valid.sum() > 0:
                            cp = (torch.sigmoid(node_logits[conn_valid]) > 0.5).long()
                            ct = (batch.y[conn_valid] >= 0).long()
                            connected_node_correct += (cp == ct).sum().item()
                            connected_node_total += conn_valid.sum().item()

                        sing_valid = valid_node_mask & ~is_conn
                        if sing_valid.sum() > 0:
                            sp = (torch.sigmoid(node_logits[sing_valid]) > 0.5).long()
                            st = (batch.y[sing_valid] >= 0).long()
                            singleton_node_correct += (sp == st).sum().item()
                            singleton_node_total += sing_valid.sum().item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            if batch_count % 50 == 0:
                if self.model_type == "hierarchical":
                    print(f"  Batch {batch_count}: loss={loss.item():.4f}, "
                          f"graph_acc(non-single)={graph_correct_nonsingle/max(graph_total_nonsingle,1):.3f}, "
                          f"conn_node_acc={connected_node_correct/max(connected_node_total,1):.3f}, "
                          f"single_node_acc={singleton_node_correct/max(singleton_node_total,1):.3f}")
                else:
                    print(f"  Batch {batch_count}: loss={loss.item():.4f}")
        
        result = {
            'train_loss': total_train_loss / max(len(train_loader), 1),
            'graph_loss': total_graph_loss / max(len(train_loader), 1),
            'node_loss': total_node_loss / max(len(train_loader), 1),
            'edge_loss': total_edge_loss / max(len(train_loader), 1),
            'avg_true_graph_prob': total_avg_true_graph_prob / max(len(train_loader), 1),
            'avg_false_graph_prob': total_avg_false_graph_prob / max(len(train_loader), 1),
            'consistency_loss': total_consistency_loss / max(len(train_loader), 1),
            'avg_edge_prob': total_avg_edge_prob / max(len(train_loader), 1),
        }
        
        if self.model_type == "hierarchical":
            result.update({
                'graph_loss': total_graph_loss / max(len(train_loader), 1),
                'node_loss': total_node_loss / max(len(train_loader), 1),
                'graph_accuracy': graph_correct_nonsingle / max(graph_total_nonsingle, 1),  # non-singleton only
                'node_accuracy': connected_node_correct / max(connected_node_total, 1),  # connected only
                'singleton_node_accuracy': singleton_node_correct / max(singleton_node_total, 1),
            })
        
        return result

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_val_loss = 0
        total_graph_loss = 0
        total_node_loss = 0

        total_avg_edge_prob = 0

        # Metrics
        graph_correct = 0
        graph_total = 0
        graph_correct_nonsingle = 0
        graph_total_nonsingle = 0
        node_correct = 0
        node_total = 0
        connected_node_correct = 0
        connected_node_total = 0
        singleton_node_correct = 0
        singleton_node_total = 0
        graph_true_positives = 0
        graph_false_positives = 0
        graph_false_negatives = 0
        node_true_positives = 0
        node_false_positives = 0
        node_false_negatives = 0

        # Initialize accumulators for additional metrics
        total_avg_true_graph_prob = 0
        total_avg_false_graph_prob = 0
        total_consistency_loss = 0
        total_edge_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                model_output = self.model(batch)
                node_logits = model_output['node_logits']
                graph_logits = model_output['graph_logits']
                edge_logits = model_output.get('edge_logits', None)
                edge_index = model_output.get('edge_index', None)
                node_embeddings = model_output.get('node_embeddings', None)

                if self.model_type == "hierarchical":
                    loss_dict = self.loss_fn(
                        node_logits = node_logits, 
                        graph_logits = graph_logits, 
                        edge_logits = edge_logits,
                        edge_index = edge_index,
                        node_embeddings = node_embeddings,
                        labels = batch.y,
                        positive_mask = batch.positive_mask, 
                        batch = batch.batch,
                        alpha=self.alpha, beta=self.beta, gamma=self.gamma, delta=self.delta, use_focal=True
                    )
                    val_loss = loss_dict['total_loss']

                    # Accumulate additional metrics
                    total_avg_true_graph_prob += loss_dict['avg_true_graph_prob']
                    total_avg_false_graph_prob += loss_dict['avg_false_graph_prob']
                    total_consistency_loss += loss_dict['consistency_loss'].item()
                    total_avg_edge_prob += loss_dict.get('avg_edge_prob', 0)
                    total_edge_loss += loss_dict.get('edge_loss', 0)

                    total_graph_loss += loss_dict['graph_loss'].item()
                    total_node_loss += loss_dict['node_loss'].item()

                    # Graph accuracy, precision, recall
                    graph_preds = (torch.sigmoid(graph_logits) > 0.5).long()
                    graph_labels = []
                    for i in range(batch.num_graphs):
                        node_mask = (batch.batch == i)
                        # A graph is True if it has any node with label >= 0
                        is_true = (batch.y[node_mask] >= 0).any().long()
                        graph_labels.append(is_true)
                    graph_labels = torch.tensor(graph_labels, device=self.device)
                    
                    graph_correct += (graph_preds == graph_labels).sum().item()
                    graph_total += batch.num_graphs
                    graph_true_positives += ((graph_preds == 1) & (graph_labels == 1)).sum().item()
                    graph_false_positives += ((graph_preds == 1) & (graph_labels == 0)).sum().item()
                    graph_false_negatives += ((graph_preds == 0) & (graph_labels == 1)).sum().item()

                    # Non-singleton graph accuracy
                    graph_has_edges = torch.zeros(batch.num_graphs, dtype=torch.bool, device=self.device)
                    if edge_index is not None and edge_index.numel() > 0:
                        graph_has_edges.scatter_(0, batch.batch[edge_index[0]], torch.ones(edge_index.shape[1], dtype=torch.bool, device=self.device))
                    nonsingle_mask_g = graph_has_edges
                    if nonsingle_mask_g.sum() > 0:
                        graph_correct_nonsingle += (graph_preds[nonsingle_mask_g] == graph_labels[nonsingle_mask_g]).sum().item()
                        graph_total_nonsingle += nonsingle_mask_g.sum().item()

                    # Valid nodes are everything except -2
                    valid_node_mask = (batch.y != -2)

                    # Determine connectivity per node
                    is_conn = torch.zeros(node_logits.size(0), dtype=torch.bool, device=self.device)
                    if edge_index is not None and edge_index.numel() > 0:
                        is_conn[edge_index[0]] = True
                        is_conn[edge_index[1]] = True

                    if valid_node_mask.sum() > 0:
                        node_preds = (torch.sigmoid(node_logits[valid_node_mask]) > 0.5).long()
                        node_targets = (batch.y[valid_node_mask] >= 0).long()
                        node_correct += (node_preds == node_targets).sum().item()
                        node_total += valid_node_mask.sum().item()
                        node_true_positives += ((node_preds == 1) & (node_targets == 1)).sum().item()
                        node_false_positives += ((node_preds == 1) & (node_targets == 0)).sum().item()
                        node_false_negatives += ((node_preds == 0) & (node_targets == 1)).sum().item()

                        conn_valid = valid_node_mask & is_conn
                        if conn_valid.sum() > 0:
                            cp = (torch.sigmoid(node_logits[conn_valid]) > 0.5).long()
                            ct = (batch.y[conn_valid] >= 0).long()
                            connected_node_correct += (cp == ct).sum().item()
                            connected_node_total += conn_valid.sum().item()

                        sing_valid = valid_node_mask & ~is_conn
                        if sing_valid.sum() > 0:
                            sp = (torch.sigmoid(node_logits[sing_valid]) > 0.5).long()
                            st = (batch.y[sing_valid] >= 0).long()
                            singleton_node_correct += (sp == st).sum().item()
                            singleton_node_total += sing_valid.sum().item()

                total_val_loss += val_loss.item()

        result = {
            'val_loss': total_val_loss / max(len(val_loader), 1),
            'graph_loss': total_graph_loss / max(len(val_loader), 1),
            'node_loss': total_node_loss / max(len(val_loader), 1),
            'edge_loss': total_edge_loss / max(len(val_loader), 1),
            'avg_true_graph_prob': total_avg_true_graph_prob / max(len(val_loader), 1),
            'avg_false_graph_prob': total_avg_false_graph_prob / max(len(val_loader), 1),
            'consistency_loss': total_consistency_loss / max(len(val_loader), 1),
            'avg_edge_prob': total_avg_edge_prob / max(len(val_loader), 1)
        }

        if self.model_type == "hierarchical":
            graph_accuracy = graph_correct / max(graph_total, 1)
            node_accuracy = node_correct / max(node_total, 1)
            graph_precision = graph_true_positives / max(graph_true_positives + graph_false_positives, 1)
            graph_recall = graph_true_positives / max(graph_true_positives + graph_false_negatives, 1)
            node_precision = node_true_positives / max(node_true_positives + node_false_positives, 1)
            node_recall = node_true_positives / max(node_true_positives + node_false_negatives, 1)
            result.update({
                'graph_loss': total_graph_loss / max(len(val_loader), 1),
                'node_loss': total_node_loss / max(len(val_loader), 1),
                'graph_accuracy': graph_correct_nonsingle / max(graph_total_nonsingle, 1),  # non-singleton only
                'node_accuracy': connected_node_correct / max(connected_node_total, 1),  # connected only
                'singleton_node_accuracy': singleton_node_correct / max(singleton_node_total, 1),
                'graph_precision': graph_precision,
                'graph_recall': graph_recall,
                'node_precision': node_precision,
                'node_recall': node_recall,
            })
        return result


    def train(self, train_graphs, val_graphs, num_epochs=15, batch_size=1, 
              augmentation="mixed", early_stopping_patience=5):
        """Main training loop"""
        # Create dataloaders with limited workers
        train_loader = DataLoader(
            train_graphs, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0 
        )
        val_loader = DataLoader(
            val_graphs, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )



        # Store additional metrics in training history
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'graph_loss': [],
            'node_loss': [],
            'edge_loss': [],
            'avg_true_graph_prob': [],
            'avg_false_graph_prob': [],
            'consistency_loss': [],
            'avg_edge_prob': []
        }

        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training phase
            avg_train_loss = self.train_epoch(train_loader, augmentation)

            # Validation phase
            avg_val_loss = self.validate(val_loader)

            epoch_time = time.time() - start_time

            # Extract the train and validation loss values from the dictionaries
            train_loss = avg_train_loss['train_loss']
            val_loss = avg_val_loss['val_loss']

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}, Time = {epoch_time:.1f}s")

            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['learning_rate'].append(current_lr)
            # Store additional metrics
            training_history['graph_loss'].append(avg_val_loss['graph_loss'])
            training_history['node_loss'].append(avg_val_loss['node_loss'])
            training_history['edge_loss'].append(avg_val_loss['edge_loss'])
            training_history['avg_true_graph_prob'].append(avg_val_loss['avg_true_graph_prob'])
            training_history['avg_false_graph_prob'].append(avg_val_loss['avg_false_graph_prob'])
            training_history['consistency_loss'].append(avg_val_loss['consistency_loss'])
            training_history['avg_edge_prob'].append(avg_val_loss['avg_edge_prob'])

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs).")
                    break

        print("Training completed!")

        return training_history

    def test(self, test_graphs):
        """
        Test the model on a test dataset using the same logic as validation.
        """
        # Create a DataLoader for the test graphs
        test_loader = DataLoader(
            test_graphs,
            batch_size=1,  # Assuming batch size of 1 for testing
            shuffle=False,
            num_workers=0
        )

        # Reuse the validate function to perform testing
        test_results = self.validate(test_loader)

        return test_results

    @staticmethod
    def load_dataset(config):
        """Load dataset based on configuration"""
        if config.use_synthetic:
            dataset = load_synthetic_dataset(config.synthetic_path)
        else:
            raise ValueError("Unknown dataset type")
        return dataset

    def predict(self, graphs, batch_size=64):
        """
        Compute predictions for a list of graphs and return probabilities.
        """
        import numpy as np
        start_time = time.time()
        self.model.eval()
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
        predictions = {}
        graph_counter = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)

                # Normalize access to logits and indices (support dict or tuple outputs)
                if isinstance(output, dict):
                    node_logits = output.get("node_logits", None)
                    graph_logits = output.get("graph_logits", None)
                    edge_logits = output.get("edge_logits", None)
                    edge_index = output.get("edge_index", None)
                elif isinstance(output, tuple):
                    # assume (node_logits, graph_logits, ...) ordering
                    try:
                        node_logits = output[0]
                        graph_logits = output[1]
                    except Exception:
                        node_logits = None
                        graph_logits = None
                    # No edge outputs in tuple form by default
                    edge_logits = None
                    edge_index = None
                else:
                    node_logits = None
                    graph_logits = None
                    edge_logits = None
                    edge_index = None

                # Compute probabilities
                node_probs = torch.sigmoid(node_logits).cpu().numpy() if node_logits is not None else None
                graph_probs = torch.sigmoid(graph_logits).cpu().numpy() if graph_logits is not None else None
                edge_probs_all = torch.sigmoid(edge_logits).cpu().numpy().flatten() if edge_logits is not None else None

                # Fallback: if model didn't return edge_index, use batch.edge_index
                if edge_index is None and hasattr(batch, 'edge_index'):
                    edge_index = batch.edge_index

                # Metadata for splitting
                batch_indices = batch.batch.cpu().numpy() if hasattr(batch, 'batch') else None
                roi_values = batch.roi_values.cpu().numpy() if hasattr(batch, 'roi_values') else None
                batch_size_actual = graph_probs.shape[0] if graph_probs is not None else batch.num_graphs

                # For each graph in the batch, slice node probs and edges (use ROI values for edges)
                for j in range(batch_size_actual):
                    # Node probs per graph
                    if batch_indices is not None and node_probs is not None:
                        node_mask = (batch_indices == j)
                        node_probs_j = node_probs[node_mask]
                    else:
                        node_probs_j = None

                    # Edge ROI pairs and probs per graph
                    edge_index_j = None
                    edge_probs_j = None
                    if edge_index is not None and edge_probs_all is not None and batch_indices is not None and roi_values is not None:
                        ei = edge_index.cpu().numpy()
                        rows = ei[0]
                        cols = ei[1]
                        edge_graph_mask = (batch_indices[rows] == j) & (batch_indices[cols] == j)
                        if edge_graph_mask.any():
                            u_rois = roi_values[rows[edge_graph_mask]]
                            v_rois = roi_values[cols[edge_graph_mask]]
                            # store as list of [u_roi, v_roi]
                            edge_index_j = np.vstack([u_rois, v_rois]).T.tolist()
                            edge_probs_j = edge_probs_all[edge_graph_mask].tolist()
                        else:
                            edge_index_j = []
                            edge_probs_j = []

                    predictions[graph_counter] = {
                        'file_id': batch.file_id[j] if hasattr(batch, 'file_id') else None,
                        'node_probs': node_probs_j,
                        'graph_probs': graph_probs[j:j+1] if graph_probs is not None else None,
                        'edge_index': edge_index_j,
                        'edge_probs': edge_probs_j
                    }
                    graph_counter += 1
        end_time = time.time()
        print(f'Prediction completed in {end_time - start_time:.2f} seconds for {len(graphs)} graphs.')
        return predictions