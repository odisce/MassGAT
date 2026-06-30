"""
Enhanced MLflow utilities for comprehensive experiment tracking
"""
import mlflow
import mlflow.pytorch
import torch
import os
from datetime import datetime
from pathlib import Path


def setup_mlflow(experiment_name="MassGAT-Training", run_name=None):
    """Initialize MLflow experiment and start a run"""
    # Resolve project root (3 levels up from this file: src/utils/mlflow_utils.py)
    project_root = Path(__file__).parent.parent.parent
    mlruns_path = project_root / "mlruns"
    mlruns_path.mkdir(exist_ok=True)

    # SQLite backend — required since MLflow 3.x deprecated the filesystem store.
    # The DB lives next to mlruns/ so the project stays self-contained.
    # Artifacts continue to be written into the existing mlruns/ directory so
    # previously logged artifacts remain accessible.
    db_path = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.absolute()}")

    # Keep artifacts in the familiar mlruns/ folder
    artifact_root = f"file://{mlruns_path.absolute()}"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow initialized: DB={db_path}, artifacts={mlruns_path}")
    
    if run_name is None:
        run_name = f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    mlflow.start_run(run_name=run_name)  # Updated to use `run_name` instead of `artifact_path`
    print(f"MLflow run started: {run_name}")
    
    # Log system info
    mlflow.log_params({
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "python_executable": os.sys.executable
    })
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1e9
        mlflow.log_params({
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory
        })
    
    return run_name


def log_training_params(params_dict):
    """Log training parameters to MLflow"""
    mlflow.log_params(params_dict)
    print(f"Logged {len(params_dict)} parameters to MLflow")


def log_training_metrics(metrics_dict, step=None):
    """Log training metrics to MLflow"""
    mlflow.log_metrics(metrics_dict, step=step)
    if step is not None:
        print(f"Logged metrics for step {step}")


def log_epoch_metrics(epoch, train_loss, val_loss, learning_rate, additional_metrics=None):
    """Log comprehensive epoch metrics"""
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    mlflow.log_metrics(metrics, step=epoch)


def save_and_log_model(model, model_id, checkpoint_path=None):
    """Save model locally and log to MLflow with better organization"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model state dict
    model_path = models_dir / f"{model_id}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Log model to MLflow
    try:
        mlflow.pytorch.log_model(model, "model", registered_model_name=f"MassGAT-{model_id}")
        mlflow.log_artifact(str(model_path), "model_files")
        
        # Log checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            mlflow.log_artifact(str(checkpoint_path), "checkpoints")
        
        print("Model logged to MLflow")
    except Exception as e:
        print(f"MLflow model logging failed: {e}")
    
    return model_path


def log_artifact_safe(file_path, artifact_path=None):
    """Safely log artifact to MLflow"""
    try:
        if Path(file_path).exists():
            mlflow.log_artifact(str(file_path), artifact_path)
            print(f"Logged artifact: {file_path}")
        else:
            print(f"Artifact not found: {file_path}")
    except Exception as e:
        print(f"Failed to log artifact {file_path}: {e}")


def end_mlflow_run():
    """End the current MLflow run safely"""
    try:
        mlflow.end_run()
        print("MLflow run ended successfully")
    except Exception as e:
        print(f"Error ending MLflow run: {e}")