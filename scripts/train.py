#!/usr/bin/env python
"""Training script for adaptive quantization-aware serving."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from adaptive_quantization_aware_serving_with_dynamic_precision.data.loader import get_dataloaders
from adaptive_quantization_aware_serving_with_dynamic_precision.models.model import create_model
from adaptive_quantization_aware_serving_with_dynamic_precision.training.trainer import Trainer
from adaptive_quantization_aware_serving_with_dynamic_precision.utils.config import (
    get_device,
    load_config,
    save_config,
    set_seed,
)
from adaptive_quantization_aware_serving_with_dynamic_precision.evaluation.analysis import (
    plot_training_curves,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train adaptive quantization-aware serving model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    return parser.parse_args()


def setup_mlflow(config):
    """Setup MLflow tracking if enabled."""
    if not config['tracking'].get('use_mlflow', False):
        return None

    try:
        import mlflow

        mlflow.set_experiment(config['tracking']['experiment_name'])
        mlflow.start_run()

        # Log config parameters
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, (dict, list)):
                        mlflow.log_param(f"{key}.{sub_key}", sub_value)
            elif not isinstance(value, (dict, list)):
                mlflow.log_param(key, value)

        logger.info("MLflow tracking enabled")
        return mlflow
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}")
        return None


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(config['logging']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, results_dir / 'config.yaml')

    # Setup MLflow
    mlflow_client = setup_mlflow(config)

    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = get_dataloaders(config)

        # Create model
        logger.info("Creating model...")
        num_classes = config['model']['num_classes']
        model = create_model(config, num_classes=num_classes)

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(model, config, device)

        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # Train model
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader, mlflow_client)

        # Save training history
        history_path = results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Plot training curves
        plot_path = plots_dir / 'training_curves.png'
        plot_training_curves(history, save_path=str(plot_path))

        # Save final model
        final_model_path = output_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        # Log final metrics
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")

        # Log model to MLflow
        if mlflow_client is not None:
            try:
                mlflow_client.log_artifact(str(final_model_path))
                mlflow_client.log_artifact(str(plot_path))
                mlflow_client.log_artifact(str(history_path))
            except Exception as e:
                logger.warning(f"Failed to log artifacts to MLflow: {e}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # End MLflow run
        if mlflow_client is not None:
            try:
                mlflow_client.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")


if __name__ == "__main__":
    main()
