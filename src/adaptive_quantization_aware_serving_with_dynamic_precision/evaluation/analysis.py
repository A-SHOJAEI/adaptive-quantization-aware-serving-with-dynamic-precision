"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot training and validation curves.

    Args:
        history: Dictionary containing training history with keys:
            'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Optional path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names.
        save_path: Optional path to save the plot.
        normalize: Whether to normalize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm)),
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_distribution(
    precision_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """Plot precision distribution across different latency budgets.

    Args:
        precision_results: Results from evaluate_precision_switching.
        save_path: Optional path to save the plot.
    """
    budgets = []
    int8_ratios = []
    fp16_ratios = []
    fp32_ratios = []
    accuracies = []

    for budget_key, metrics in precision_results.items():
        budget = float(budget_key.replace('budget_', '').replace('ms', ''))
        budgets.append(budget)
        int8_ratios.append(metrics.get('int8_ratio', 0) * 100)
        fp16_ratios.append(metrics.get('fp16_ratio', 0) * 100)
        fp32_ratios.append(metrics.get('fp32_ratio', 0) * 100)
        accuracies.append(metrics['accuracy'] * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Stacked bar chart for precision distribution
    x = np.arange(len(budgets))
    width = 0.6

    ax1.bar(x, int8_ratios, width, label='INT8', color='#3498db')
    ax1.bar(x, fp16_ratios, width, bottom=int8_ratios, label='FP16', color='#2ecc71')
    ax1.bar(
        x, fp32_ratios, width,
        bottom=[i + j for i, j in zip(int8_ratios, fp16_ratios)],
        label='FP32', color='#e74c3c'
    )

    ax1.set_xlabel('Latency Budget (ms)', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Precision Distribution by Latency Budget', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{b:.1f}' for b in budgets])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Accuracy vs latency budget
    ax2.plot(budgets, accuracies, 'o-', linewidth=2, markersize=8, color='#9b59b6')
    ax2.set_xlabel('Latency Budget (ms)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Latency Budget', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision distribution plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pareto_frontier(
    results: List[Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """Plot throughput-accuracy Pareto frontier.

    Args:
        results: List of dictionaries with 'throughput' and 'accuracy' keys.
        save_path: Optional path to save the plot.
    """
    throughputs = [r['throughput_samples_per_sec'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    labels = [r.get('label', '') for r in results]

    plt.figure(figsize=(10, 6))

    # Plot points
    plt.scatter(throughputs, accuracies, s=100, alpha=0.6, c='#3498db')

    # Add labels
    for i, label in enumerate(labels):
        if label:
            plt.annotate(
                label,
                (throughputs[i], accuracies[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )

    plt.xlabel('Throughput (samples/sec)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Throughput-Accuracy Pareto Frontier', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Pareto frontier plot to {save_path}")
    else:
        plt.show()

    plt.close()
