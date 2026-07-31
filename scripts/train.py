#!/usr/bin/env python3
"""
Refactored training script for MassGAT using modular components
"""
import torch
import argparse
import glob
import fnmatch
from datetime import datetime
import os
import time
import mlflow
import pandas as pd

# Local imports
from src.data import load_synthetic_dataset, load_mzml_file
from src.models import MassGAT
from src.training import ChroMSTrainer, full_hierarchical_loss

from src.utils import (setup_mlflow, log_training_params, log_training_metrics, 
                      end_mlflow_run, set_seed)


def save_trained_model(model, config, metrics, description="", output_dir="weights"):
    """Save model checkpoint and metadata to a timestamped .pth file."""
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = hex(abs(hash(str(config))) % (16**8))[2:].zfill(8)
    path = os.path.join(output_dir, f"model_{model_id}_{stamp}.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
            "description": description,
            "created_at": stamp,
        },
        path,
    )
    return path

def train_val_test_split(graphs, seed=42):
    import random
    random.seed(seed)  # Use provided seed for reproducibility
    random.shuffle(graphs)
    split1 = int(0.7 * len(graphs))
    split2 = int(0.9 * len(graphs))
    train_graphs = graphs[:split1]
    val_graphs = graphs[split1:split2]
    test_graphs = graphs[split2:]
    print(f"Random split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    
    return train_graphs, val_graphs, test_graphs

def main(args):
    """Main training function with comprehensive seeding"""
    start_time = time.time()  # Start tracking total execution time

    # Set reproducibility seed FIRST
    set_seed(args.seed)
    seed_time = time.time()
    print(f"Reproducibility seed set to: {args.seed} (Time taken: {seed_time - start_time:.2f}s)")

    # Limit CPU threads to prevent overuse
    torch.set_num_threads(4)  # Limit PyTorch threads
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Limit Intel MKL threads

    print("Starting MassGAT training with modular architecture...")

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_check_time = time.time()
    print(f"Using device: {device} (Time taken: {gpu_check_time - seed_time:.2f}s)")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Setup MLflow tracking
    run_name = setup_mlflow(
        experiment_name="MassGAT-Training"
    )
    mlflow_setup_time = time.time()
    print(f"MLflow setup completed (Time taken: {mlflow_setup_time - gpu_check_time:.2f}s)")

    try:
        # Load data with seeded split
        print("\n=== Data Loading ===")
        data_loading_start = time.time()
        # Load data based on configuration
        if args.use_synthetic:
            dataset = load_synthetic_dataset(args.synthetic_path, load_from_cache=args.load_from_cache)
            graphs = [dataset[i] for i in range(len(dataset))]  # Convert to list of Data objects
            num_classes = dataset.get_num_classes()

        data_loading_end = time.time()
        print(f"Data loading completed (Time taken: {data_loading_end - data_loading_start:.2f}s)")

        if len(graphs) < 1000:
            print("Need at least 1000 graphs for training")
            return

        # Split data with seed for reproducible fallback
        split_start = time.time()
        train_graphs, val_graphs, test_graphs = train_val_test_split(graphs, seed=args.seed)
        split_end = time.time()
        print(f"Data splitting completed (Time taken: {split_end - split_start:.2f}s)")

        print(f"\nDataset Statistics:")
        print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
        print(f"  Node features: 72 dims")
        print(f"  Number of classes: {num_classes}")
        print(f"  Example graph: {train_graphs[0].x.size(0)} nodes, {train_graphs[0].edge_index.size(1)} edges")

        # Training configuration
        config = {
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
            'embedding_dim': args.embedding_dim,
            'alpha': args.alpha,
            'beta': args.beta,
            'augmentation': args.augmentation,
            'num_classes': num_classes,
            'num_train_graphs': len(train_graphs),
            'num_val_graphs': len(val_graphs),
            'num_test_graphs': len(test_graphs),
            'node_features': 72,
            'total_graphs': len(graphs),
            'split_method': 'pool_based',
            'model_type': args.model_type,
            'model_choice': args.model_choice,
            'encoder_type': args.encoder_type,
            'synthetic_path': args.synthetic_path,
            'block_type': args.block_type,  # Added for ablation
            'zero_metadata_all_nodes': args.zero_metadata_all_nodes,
        }



        # Log configuration
        log_training_params(config)

        # Create model
        print("\n=== Model Creation ===")
        model_creation_start = time.time()

        print(f"Using Iterative Hierarchical GNN model for synthetic data (block_type={args.block_type})")
        model = MassGAT(
            num_features=72,
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            encoder_type=config['encoder_type'],
            block_type=args.block_type,  # Pass block_type for ablation
            zero_metadata_all_nodes=args.zero_metadata_all_nodes,
        )


        model_creation_end = time.time()
        print(f"Model creation completed (Time taken: {model_creation_end - model_creation_start:.2f}s)")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {total_params:,} parameters")

        # Create trainer
        trainer_creation_start = time.time()
        trainer = ChroMSTrainer(
            model, 
            config['learning_rate'], 
            device, 
            alpha=config['alpha'], 
            beta=config['beta'],
            model_type=config['model_type'],
            loss_fn = full_hierarchical_loss
        )
        trainer_creation_end = time.time()
        print(f"Trainer creation completed (Time taken: {trainer_creation_end - trainer_creation_start:.2f}s)")

        # Train the model
        print("\n=== Training ===")
        training_start = time.time()
        training_history = trainer.train(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            augmentation=config['augmentation'],
        )
        training_end = time.time()
        print(f"Training completed (Time taken: {training_end - training_start:.2f}s)")

        # Log training metrics to MLflow
        for epoch in range(len(training_history['train_loss'])):
            metrics = {key: values[epoch] for key, values in training_history.items()}
            metrics['epoch'] = epoch + 1  # Add epoch number to metrics
            log_training_metrics(metrics, step=epoch)


        # Get results from the test set
        print("\n=== Testing ===")
        test_results = trainer.test(test_graphs)
        print(f"Testing completed. Test Results: {test_results}")
        # Log test results with a distinct prefix to avoid colliding with per-epoch val metrics
        prefixed_test_results = {f"test_{k}": v for k, v in test_results.items()}
        log_training_metrics(prefixed_test_results)


        # Save and log model (organized weights system only)
        print("\n=== Model Saving ===")
        model_saving_start = time.time()
        model_config = {
            'num_features': 72,
            'hidden_dim': config['hidden_dim'],
            'embedding_dim': config['embedding_dim'],
            'num_layers': 4
        }

        # Ensure metrics dictionary includes total_loss
        weights_metrics = {
            'total_loss': training_history['val_loss'][-1],
            'train_loss': training_history['train_loss'][-1],
            'val_loss': training_history['val_loss'][-1],
            'epochs_trained': config['num_epochs'],
            'total_parameters': total_params
        }

        description = f"Model- (α={config['alpha']}, β={config['beta']})"

        # Save weights to disk first
        organized_path = save_trained_model(
            model=model,
            config=model_config,
            metrics=weights_metrics,
            description=description
        )

        try:
            mlflow.log_artifact(str(organized_path), artifact_path="model_weights")
            # Also save the state dict explicitly for easy reloading
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                state_dict_path = os.path.join(tmp, "state_dict.pth")
                torch.save(model.state_dict(), state_dict_path)
                mlflow.log_artifact(state_dict_path, artifact_path="model_weights")
            print("Model weights logged to MLflow as artifact")
        except Exception as e:
            print(f"MLflow artifact logging failed (weights already saved locally): {e}")

        model_saving_end = time.time()
        print(f"Model saving completed (Time taken: {model_saving_end - model_saving_start:.2f}s)")
        print(f"Organized weights saved to: {organized_path}")
        

        # Log final metrics
        final_metrics = {
            "final_train_loss": training_history['train_loss'][-1],
            "final_val_loss": training_history['val_loss'][-1],
            "total_epochs": config['num_epochs'],
            "total_parameters": total_params
        }
        log_training_metrics(final_metrics)

        print("Training completed successfully!")
        print(f" Organized weights saved to: {organized_path}")
        print(" Metrics logged to MLflow")

        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f}s")

        return model, organized_path, final_metrics

    except Exception as e:
        print(f"Training error: {e}")
        log_training_params({"error": str(e)})
        raise

    finally:
        end_mlflow_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MassGAT Modular Training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=48, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=24, help='Embedding dimension')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for binary classification loss')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for contrastive clustering loss')
    parser.add_argument('--augmentation', type=str, default='none', 
                       choices=['none', 'node_drop', 'edge_drop', 'feature_noise', 'feature_mask', 'mixed'],
                       help='Augmentation strategy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_synthetic', action='store_true', default=True, help='Use synthetic data')
    parser.add_argument('--synthetic_path', type=str, default='data/synthetic/synthetic_dataset', help='Path to synthetic data')
    parser.add_argument('--load_from_cache', action='store_true', default=False, help='Load data from cache if available')
    parser.add_argument('--model_type', type=str, default='hierarchical', choices=['hierarchical', 'contrastive'], help='Type of model to train')
    parser.add_argument('--model_choice', type=str, default='baseline', choices=['baseline', 'iterative', 'no_interaction'], help='GNN model choice')
    parser.add_argument('--encoder_type', type=str, default='transformer', choices=['transformer', 'cnn'], help='Type of signal encoder to use')
    parser.add_argument('--block_type', type=str, default='gate', choices=['gate', 'gat', 'gcn'], help='Type of GNN block for ablation (gate, gat, gcn)')
    parser.add_argument('--zero_metadata_all_nodes', action='store_true', default=False, help='Zero out first 8 metadata features for all nodes')
    args = parser.parse_args()
    main(args)
