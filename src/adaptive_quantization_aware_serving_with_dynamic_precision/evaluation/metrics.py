"""Evaluation metrics computation."""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels of shape (num_samples,).
        y_pred: Predicted labels of shape (num_samples,).
        num_classes: Number of classes.

    Returns:
        Dictionary containing various metrics.
    """
    metrics = {}

    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted metrics
    metrics['precision_weighted'] = precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['recall_weighted'] = recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )

    for i in range(num_classes):
        metrics[f'class_{i}_precision'] = per_class_precision[i]
        metrics[f'class_{i}_recall'] = per_class_recall[i]
        metrics[f'class_{i}_f1'] = per_class_f1[i]

    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: Data loader for evaluation.
        device: Device to run evaluation on.
        num_classes: Number of classes.

    Returns:
        Tuple of:
            - Dictionary of metrics
            - Array of true labels
            - Array of predicted labels
    """
    model.eval()

    all_preds = []
    all_targets = []
    total_time = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            start_time = time.time()
            outputs, _, _ = model(inputs)
            batch_time = time.time() - start_time

            total_time += batch_time
            num_batches += 1

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, num_classes)

    # Add latency metrics
    avg_batch_time = total_time / num_batches
    batch_size = len(inputs)
    avg_sample_time = (avg_batch_time / batch_size) * 1000  # Convert to ms

    metrics['avg_inference_time_ms'] = avg_sample_time
    metrics['throughput_samples_per_sec'] = 1000.0 / avg_sample_time

    logger.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}")

    return metrics, y_true, y_pred


def evaluate_precision_switching(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    latency_budgets: List[float] = [5.0, 10.0, 15.0, 20.0]
) -> Dict[str, Dict[str, float]]:
    """Evaluate model with different latency budgets for precision switching.

    Args:
        model: Adaptive quantized model to evaluate.
        dataloader: Data loader for evaluation.
        device: Device to run evaluation on.
        num_classes: Number of classes.
        latency_budgets: List of latency budgets in milliseconds to test.

    Returns:
        Dictionary mapping latency budget to metrics.
    """
    results = {}

    for budget in latency_budgets:
        logger.info(f"Evaluating with latency budget: {budget}ms")

        model.eval()
        all_preds = []
        all_targets = []
        precision_counts = {0: 0, 1: 0, 2: 0}  # INT8, FP16, FP32

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Predict optimal precision
                if hasattr(model, 'predict_precision'):
                    precision_indices = model.predict_precision(inputs, latency_budget=budget)
                    for idx in precision_indices.cpu().numpy():
                        precision_counts[int(idx)] += 1

                # Standard inference
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        metrics = compute_metrics(y_true, y_pred, num_classes)

        # Add precision distribution
        total_samples = sum(precision_counts.values())
        if total_samples > 0:
            metrics['int8_ratio'] = precision_counts[0] / total_samples
            metrics['fp16_ratio'] = precision_counts[1] / total_samples
            metrics['fp32_ratio'] = precision_counts[2] / total_samples

        results[f'budget_{budget}ms'] = metrics

        logger.info(
            f"Budget {budget}ms - Accuracy: {metrics['accuracy']:.4f}, "
            f"INT8: {metrics.get('int8_ratio', 0):.2%}, "
            f"FP16: {metrics.get('fp16_ratio', 0):.2%}, "
            f"FP32: {metrics.get('fp32_ratio', 0):.2%}"
        )

    return results
