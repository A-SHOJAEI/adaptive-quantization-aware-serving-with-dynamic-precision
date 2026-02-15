"""Tests for data loading and preprocessing."""

import pytest
import torch

from adaptive_quantization_aware_serving_with_dynamic_precision.data.loader import (
    get_dataloaders,
    get_calibration_loader,
)
from adaptive_quantization_aware_serving_with_dynamic_precision.data.preprocessing import (
    get_transforms,
)


def test_get_transforms():
    """Test transform creation."""
    transforms = get_transforms('cifar10', augmentation=True)

    assert 'train' in transforms
    assert 'val' in transforms
    assert transforms['train'] is not None
    assert transforms['val'] is not None


def test_get_transforms_no_augmentation():
    """Test transforms without augmentation."""
    transforms = get_transforms('cifar10', augmentation=False)

    assert 'train' in transforms
    assert 'val' in transforms


def test_get_dataloaders(sample_config):
    """Test dataloader creation."""
    train_loader, val_loader, test_loader = get_dataloaders(sample_config)

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check batch size
    batch = next(iter(train_loader))
    images, labels = batch

    assert images.shape[0] <= sample_config['training']['batch_size']
    assert labels.shape[0] <= sample_config['training']['batch_size']
    assert images.shape[1:] == (3, 32, 32)


def test_get_calibration_loader(sample_config):
    """Test calibration loader creation."""
    calib_loader = get_calibration_loader(sample_config, num_samples=100)

    assert calib_loader is not None

    # Check that we get batches
    batch = next(iter(calib_loader))
    images, labels = batch

    assert images.shape[1:] == (3, 32, 32)
    assert len(labels) > 0


def test_dataloader_reproducibility(sample_config):
    """Test that dataloaders are reproducible with same seed."""
    from adaptive_quantization_aware_serving_with_dynamic_precision.utils.config import set_seed

    set_seed(42)
    train_loader1, _, _ = get_dataloaders(sample_config)
    batch1 = next(iter(train_loader1))

    set_seed(42)
    train_loader2, _, _ = get_dataloaders(sample_config)
    batch2 = next(iter(train_loader2))

    # Check that we get the same samples
    assert torch.allclose(batch1[0], batch2[0])
    assert torch.all(batch1[1] == batch2[1])
