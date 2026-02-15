#!/usr/bin/env python
"""Evaluation script for adaptive quantization-aware serving."""

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
from adaptive_quantization_aware_serving_with_dynamic_precision.utils.config import (
    get_device,
    load_config,
    set_seed,
)
from adaptive_quantization_aware_serving_with_dynamic_precision.evaluation.metrics import (
    evaluate_model,
    evaluate_precision_switching,
)
from adaptive_quantization_aware_serving_with_dynamic_precision.evaluation.analysis import (
    save_confusion_matrix,
    plot_precision_distribution,
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
        description='Evaluate adaptive quantization-aware serving model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (if not in checkpoint)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--test-precision-switching',
        action='store_true',
        help='Test precision switching with different latency budgets'
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config_path: Optional path to config file.

    Returns:
        Tuple of (model, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load config from checkpoint or file
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path:
        config = load_config(config_path)
    else:
        raise ValueError("Config not found in checkpoint and no config file provided")

    # Create model
    num_classes = config['model']['num_classes']
    model = create_model(config, num_classes=num_classes)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded model from {checkpoint_path}")
    return model, config


def print_metrics_table(metrics: dict):
    """Print metrics in a formatted table."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']*100:6.2f}%")
    print(f"  Precision (macro):  {metrics['precision_macro']*100:6.2f}%")
    print(f"  Recall (macro):     {metrics['recall_macro']*100:6.2f}%")
    print(f"  F1 Score (macro):   {metrics['f1_macro']*100:6.2f}%")

    # Latency metrics
    if 'avg_inference_time_ms' in metrics:
        print(f"\nLatency Metrics:")
        print(f"  Avg inference time: {metrics['avg_inference_time_ms']:6.2f} ms")
        print(f"  Throughput:         {metrics['throughput_samples_per_sec']:6.0f} samples/sec")

    # Per-class metrics
    num_classes = sum(1 for k in metrics if k.startswith('class_') and k.endswith('_f1'))
    if num_classes > 0:
        print(f"\nPer-Class Metrics:")
        print(f"  {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("  " + "-"*48)
        for i in range(num_classes):
            prec = metrics[f'class_{i}_precision'] * 100
            rec = metrics[f'class_{i}_recall'] * 100
            f1 = metrics[f'class_{i}_f1'] * 100
            print(f"  {i:<8} {prec:>10.2f}%  {rec:>10.2f}%  {f1:>10.2f}%")

    print("="*60 + "\n")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please train a model first using: python scripts/train.py")
        return

    # Load model and config
    logger.info("Loading model...")
    model, config = load_model_from_checkpoint(args.checkpoint, args.config)

    # Set seed
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device()
    model = model.to(device)
    model.eval()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading test data...")
    _, _, test_loader = get_dataloaders(config)

    # Evaluate model
    logger.info("Evaluating model on test set...")
    num_classes = config['model']['num_classes']
    metrics, y_true, y_pred = evaluate_model(model, test_loader, device, num_classes)

    # Print results
    print_metrics_table(metrics)

    # Save metrics to JSON
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save confusion matrix
    cm_path = plots_dir / 'confusion_matrix.png'
    save_confusion_matrix(y_true, y_pred, save_path=str(cm_path), normalize=True)

    # Test precision switching if requested
    if args.test_precision_switching:
        logger.info("Testing precision switching with different latency budgets...")

        latency_budgets = [5.0, 10.0, 15.0, 20.0]
        precision_results = evaluate_precision_switching(
            model, test_loader, device, num_classes, latency_budgets
        )

        # Save precision switching results
        precision_path = output_dir / 'precision_switching_results.json'
        with open(precision_path, 'w') as f:
            json.dump(precision_results, f, indent=2)
        logger.info(f"Saved precision switching results to {precision_path}")

        # Plot precision distribution
        dist_path = plots_dir / 'precision_distribution.png'
        plot_precision_distribution(precision_results, save_path=str(dist_path))

        # Print summary
        print("\nPrecision Switching Results:")
        print("="*60)
        for budget_key, budget_metrics in precision_results.items():
            budget = budget_key.replace('budget_', '').replace('ms', '')
            print(f"\nLatency Budget: {budget} ms")
            print(f"  Accuracy: {budget_metrics['accuracy']*100:.2f}%")
            if 'int8_ratio' in budget_metrics:
                print(f"  INT8:  {budget_metrics['int8_ratio']*100:.1f}%")
                print(f"  FP16:  {budget_metrics['fp16_ratio']*100:.1f}%")
                print(f"  FP32:  {budget_metrics['fp32_ratio']*100:.1f}%")
        print("="*60 + "\n")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
